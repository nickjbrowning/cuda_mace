#include <cuda.h>
#include <mma.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cuda_pipeline_primitives.h>
#include <stdio.h>
#include <stdlib.h>

#include "cuda_utils.cuh"
#include "linear_wmma_impl.cuh"

using namespace nvcuda;
using namespace std;
using namespace torch::autograd;
using namespace torch::indexing;

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define M_BATCH 16
#define N_BATCH 32
#define K_BATCH 32

template <const int NWARPS>
__global__ void __launch_bounds__(NWARPS *WARP_SIZE)
    linear_wmma_kernel(const float *__restrict__ X, const float *__restrict__ W,
                       float *__restrict__ OUT, const int NNODES, const uint M,
                       const uint N, const uint K, const uint L) {
#if __CUDA_ARCH__ >= 800
  const uint cCol = blockIdx.y;

  extern __shared__ char buffer[];

  void *sptr = buffer;
  size_t space = 0;

  float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
  float *buffer_out =
      shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

  /*
      for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < K_BATCH *
     (K_BATCH + 1); tid += blockDim.x * blockDim.y)
      {
          Xs[tid] = 0.0;
      }

      for (int tid = threadIdx.y * blockDim.x + threadIdx.x; tid < M *
     blockDim.y * WMMA_N; tid += blockDim.x * blockDim.y)
      {
          buffer_out[tid] = 0.0;
      }
  */

  // X += node * M * K; // move pointer to start of X

  // OUT += node * M * N + cCol * BN; // move pointer to start of OUT

  const float path_weight = 1.0f / sqrt((float)K);

  const uint lstart = L * L;
  const uint nl = 2 * L + 1;

  W += L * K * N; // move W to the correct weights sub-matrix

  const uint threadCol = threadIdx.x; // [0-32]
  const uint threadRow = threadIdx.y; //  128 / 32 = [0-4]
  const uint rowStrideB = blockDim.y;
  const uint nmiter = find_integer_divisor(M, rowStrideB);
  // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
  // wmma::precision::tf32, wmma::row_major> a_frag, delta_a_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::col_major>
      a_frag, delta_a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::row_major>
      b_frag, delta_b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

  wmma::fill_fragment(ab_frag, 0.0f);

  // const uint aRow = 0;

  const uint bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

  for (uint bkIdx = 0; bkIdx < K; bkIdx += WMMA_K) // 0, 16, 32
  {
    __syncthreads();

    if (bkIdx % 32 == 0) {
      for (int m = 0; m < nmiter; m++) {
        // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node
        // ((m * rowStrideB + threadRow) % nl -> index into the current L
        // channel
        if (m * rowStrideB + threadRow < 16) {
          int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;
          int lidx = lstart + (gid / nl) * 16 + gid % nl;

          if (lidx < NNODES * 16) // bounds checking
          {
            Xs[threadCol * 33 + m * rowStrideB + threadRow] =
                X[lidx * K + bkIdx + threadCol];
          } else {
            Xs[threadCol * 33 + m * rowStrideB + threadRow] = 0.0f;
          }
        }
      }
    }

    __syncthreads();

    if (bCol < N) {
      wmma::load_matrix_sync(a_frag, Xs + (bkIdx % 32) * 33, 33);
      wmma::load_matrix_sync(b_frag, W + bCol, N);

      for (int l = 0; l < a_frag.num_elements; l++) {
        float curr = a_frag.x[l];
        float tf32 = wmma::__float_to_tf32(curr);
        delta_a_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
        a_frag.x[l] = tf32;
      }

      for (int l = 0; l < b_frag.num_elements; l++) {
        float curr = b_frag.x[l];
        float tf32 = wmma::__float_to_tf32(curr);
        delta_b_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
        b_frag.x[l] = tf32;
      }

      wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
      wmma::mma_sync(ab_frag, a_frag, delta_b_frag, ab_frag);
      wmma::mma_sync(ab_frag, delta_a_frag, b_frag, ab_frag);

      wmma::store_matrix_sync(buffer_out + threadIdx.y * WMMA_N, ab_frag,
                              blockDim.y * WMMA_N, wmma::mem_row_major);
    }

    W += WMMA_K * N; // move the pointer to W along by BKxN
  }

  __syncthreads();

  // N_START = cCol * blockDim.y * WMMA_N

  for (uint n_block = 0;
       n_block <
       min(N - cCol * (blockDim.y * WMMA_N), blockDim.y * WMMA_N) / blockDim.x;
       n_block++) {
    for (int m = 0; m < nmiter; m++) {
      // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node for
      // lm channel
      // ((m * rowStrideB + threadRow) % nl -> index into the current lm index
      if (m * rowStrideB + threadRow < 16) {
        int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;

        int lidx = lstart + (gid / nl) * 16 + gid % nl;

        if (lidx < NNODES * 16 &&
            cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol < N)
          OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 +
              threadCol] =
              path_weight *
              buffer_out[(m * rowStrideB + threadRow) * (blockDim.y * WMMA_N) +
                         n_block * 32 + threadCol];
      }
    }
  }
#endif
}

torch::Tensor linear_wmma(torch::Tensor X, torch::Tensor W) {

  const int NNODES = X.size(0);
  const int M = X.size(1);
  const int N = W.size(2);
  const int K = W.size(1);

  torch::Tensor output =
      torch::zeros({NNODES, M, N},
                   torch::TensorOptions().dtype(X.dtype()).device(X.device()));

  dim3 blockDim;

  blockDim.x = WARP_SIZE;

  if (N >= 128) {
    blockDim.y = 8;
  } else if (N == 96) {
    blockDim.y = 6;
  } else if (N == 64) {
    blockDim.y = 4;
  } else if (N == 32) {
    blockDim.y = 2;
  }

  size_t shared_size = 0;
  void *sptr = nullptr;

  shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &shared_size);
  shared_array<float>(M * blockDim.y * WMMA_N, sptr, &shared_size);

  vector<cudaStream_t> streams;

  for (int l = 0; l < 4; l++) {
    dim3 griddim;
    griddim.x = find_integer_divisor(NNODES * (2 * l + 1), 16);
    griddim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

    cudaStream_t stream;

    cudaStreamCreate(&stream);

    if (N >= 128) {
      linear_wmma_kernel<8><<<griddim, blockDim, shared_size, stream>>>(
          X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
          NNODES, M, N, K, l);
    } else if (N == 96) {
      linear_wmma_kernel<6><<<griddim, blockDim, shared_size, stream>>>(
          X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
          NNODES, M, N, K, l);
    } else if (N == 64) {
      linear_wmma_kernel<4><<<griddim, blockDim, shared_size, stream>>>(
          X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
          NNODES, M, N, K, l);
    } else if (N == 32) {
      linear_wmma_kernel<2><<<griddim, blockDim, shared_size, stream>>>(
          X.data_ptr<float>(), W.data_ptr<float>(), output.data_ptr<float>(),
          NNODES, M, N, K, l);
    }

    streams.push_back(stream);
  }

  for (int l = 0; l < streams.size(); l++) {
    cudaStreamDestroy(streams[l]);
  }

  return output;
}

template <const int NWARPS>
__global__ void __launch_bounds__(NWARPS *WARP_SIZE)
    elemental_linear_wmma_kernel(
        const float *__restrict__ X, const float *__restrict__ W,
        const int64_t *__restrict__ node_idx, const int64_t nselected,
        const int64_t element_id, const int64_t nelements,
        float *__restrict__ OUT, const int NNODES, const uint M, const uint N,
        const uint K, const uint L) {

#if __CUDA_ARCH__ >= 800
  const uint cCol = blockIdx.y;

  extern __shared__ char buffer[];

  void *sptr = buffer;
  size_t space = 0;

  float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
  float *buffer_out =
      shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

  // sqrt((2L+1))/sqrt(n_channels * n_elements)
  // const float path_weight = sqrt((float)(2 * L + 1)) / sqrt((float)K *
  // (float)nelements);
  const float path_weight = 1.0f / sqrt((float)K);

  const uint lstart = L * L;
  const uint nl = 2 * L + 1;

  W += element_id * 4 * K * N +
       L * K * N; // move W to the correct weights sub-matrix

  const uint threadCol = threadIdx.x; // [0-32]
  const uint threadRow = threadIdx.y; //  128 / 32 = [0-4]
  const uint rowStrideB = blockDim.y;
  const uint nmiter = find_integer_divisor(M, rowStrideB);
  // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
  // wmma::precision::tf32, wmma::row_major> a_frag, delta_a_frag;
  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::col_major>
      a_frag, delta_a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::row_major>
      b_frag, delta_b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

  wmma::fill_fragment(ab_frag, 0.0f);

  // const uint aRow = 0;

  const uint bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

  for (uint bkIdx = 0; bkIdx < K; bkIdx += WMMA_K) // 0, 16, 32
  {
    __syncthreads();

    if (bkIdx % 32 == 0) {
      for (int m = 0; m < nmiter; m++) {
        int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;

        // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node
        // ((m * rowStrideB + threadRow) % nl -> index into the current L
        // channel
        if (m * rowStrideB + threadRow < 16 && gid / nl < nselected) {
          int lidx = lstart + node_idx[(gid / nl)] * 16 + gid % nl;
          // int lidx = lstart + gid / nl * 16 + gid % nl;
          if (lidx < NNODES * 16) // bounds checking
          {
            Xs[threadCol * 33 + m * rowStrideB + threadRow] =
                X[lidx * K + bkIdx + threadCol];
            // Xs[threadCol * 33 + m * rowStrideB + threadRow] = X[lidx * K +
            // bkIdx + threadCol];
          } else {
            Xs[threadCol * 33 + m * rowStrideB + threadRow] = 0.0f;
          }
        }
      }
    }

    __syncthreads();

    if (bCol < N) {
      wmma::load_matrix_sync(a_frag, Xs + (bkIdx % 32) * 33, 33);
      wmma::load_matrix_sync(b_frag, W + bCol, N);

      for (int l = 0; l < a_frag.num_elements; l++) {
        float curr = a_frag.x[l];
        float tf32 = wmma::__float_to_tf32(curr);
        delta_a_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
        a_frag.x[l] = tf32;
      }

      for (int l = 0; l < b_frag.num_elements; l++) {
        float curr = b_frag.x[l];
        float tf32 = wmma::__float_to_tf32(curr);
        delta_b_frag.x[l] = wmma::__float_to_tf32(curr - tf32);
        b_frag.x[l] = tf32;
      }

      wmma::mma_sync(ab_frag, a_frag, b_frag, ab_frag);
      wmma::mma_sync(ab_frag, a_frag, delta_b_frag, ab_frag);
      wmma::mma_sync(ab_frag, delta_a_frag, b_frag, ab_frag);

      wmma::store_matrix_sync(buffer_out + threadIdx.y * WMMA_N, ab_frag,
                              blockDim.y * WMMA_N, wmma::mem_row_major);
    }

    W += WMMA_K * N; // move the pointer to W along by BKxN
  }

  __syncthreads();

  // N_START = cCol * blockDim.y * WMMA_N

  for (uint n_block = 0;
       n_block <
       min(N - cCol * (blockDim.y * WMMA_N), blockDim.y * WMMA_N) / blockDim.x;
       n_block++) {
    for (int m = 0; m < nmiter; m++) {

      int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;
      // ((m * rowStrideB + threadRow)/nl) * 16 -> start index for each node for
      // lm channel
      // ((m * rowStrideB + threadRow) % nl -> index into the current lm index
      if (m * rowStrideB + threadRow < 16 && gid / nl < nselected) {

        /// int lidx = lstart + (gid / nl) * 16 + gid % nl;
        int lidx = lstart + node_idx[(gid / nl)] * 16 + gid % nl;

        if (lidx < NNODES * 16 &&
            cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol < N) {
          OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 +
              threadCol] =
              path_weight *
              buffer_out[(m * rowStrideB + threadRow) * (blockDim.y * WMMA_N) +
                         n_block * 32 + threadCol];
          // OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 +
          // threadCol] = path_weight * buffer_out[(m * rowStrideB + threadRow)
          // * (blockDim.y * WMMA_N) + n_block * 32 + threadCol];
        }
      }
    }
  }
#endif
}

torch::Tensor elemental_linear_wmma(torch::Tensor X, torch::Tensor W,
                                    torch::Tensor elemental_embedding) {

  const int NNODES = X.size(0);
  const int M = X.size(1);
  const int N = W.size(-1);
  const int K = W.size(2);

  torch::Tensor output =
      torch::zeros({NNODES, M, N},
                   torch::TensorOptions().dtype(X.dtype()).device(X.device()));

  dim3 blockDim;

  // blockDim.y = min(8, find_integer_divisor(N, WMMA_N));
  blockDim.x = WARP_SIZE;

  if (N >= 128) {
    blockDim.y = 8;
  } else if (N == 96) {
    blockDim.y = 6;
  } else if (N == 64) {
    blockDim.y = 4;
  } else if (N == 32) {
    blockDim.y = 2;
  }

  size_t shared_size = 0;
  void *sptr = nullptr;

  shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &shared_size);
  shared_array<float>(M * blockDim.y * WMMA_N, sptr, &shared_size);

  vector<cudaStream_t> streams;

  for (int element_id = 0; element_id < elemental_embedding.size(1);
       element_id++) {
    torch::Tensor elemental_embedding_ =
        elemental_embedding.index({Ellipsis, element_id});
    torch::Tensor node_idx = torch::where(elemental_embedding_ == 1)[0];

    int64_t nselected = node_idx.size(0);

    if (nselected > 0) {

      for (int l = 0; l < 4; l++) {
        dim3 griddim;
        griddim.x = find_integer_divisor(nselected * (2 * l + 1), 16);
        griddim.y = find_integer_divisor(N, blockDim.y * WMMA_N);

        cudaStream_t stream;

        cudaStreamCreate(&stream);

        if (N >= 128) {
          elemental_linear_wmma_kernel<8>
              <<<griddim, blockDim, shared_size, stream>>>(
                  X.data_ptr<float>(), W.data_ptr<float>(),
                  node_idx.data_ptr<int64_t>(), nselected, element_id,
                  elemental_embedding.size(-1), output.data_ptr<float>(),
                  NNODES, M, N, K, l);
        } else if (N == 96) {
          elemental_linear_wmma_kernel<6>
              <<<griddim, blockDim, shared_size, stream>>>(
                  X.data_ptr<float>(), W.data_ptr<float>(),
                  node_idx.data_ptr<int64_t>(), nselected, element_id,
                  elemental_embedding.size(-1), output.data_ptr<float>(),
                  NNODES, M, N, K, l);
        } else if (N == 64) {
          elemental_linear_wmma_kernel<4>
              <<<griddim, blockDim, shared_size, stream>>>(
                  X.data_ptr<float>(), W.data_ptr<float>(),
                  node_idx.data_ptr<int64_t>(), nselected, element_id,
                  elemental_embedding.size(-1), output.data_ptr<float>(),
                  NNODES, M, N, K, l);
        } else if (N == 32) {
          elemental_linear_wmma_kernel<2>
              <<<griddim, blockDim, shared_size, stream>>>(
                  X.data_ptr<float>(), W.data_ptr<float>(),
                  node_idx.data_ptr<int64_t>(), nselected, element_id,
                  elemental_embedding.size(-1), output.data_ptr<float>(),
                  NNODES, M, N, K, l);
        }

        streams.push_back(stream);
      }
    }
  }

  for (int l = 0; l < streams.size(); l++) {
    cudaStreamDestroy(streams[l]);
  }

  return output;
}