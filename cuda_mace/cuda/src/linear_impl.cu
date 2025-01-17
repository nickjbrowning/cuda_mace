
#include <mma.h>
using namespace nvcuda;

#define WARP_SIZE 32

// MMA matrix tile dimensions.
#define WMMA_M 16
#define WMMA_N 16
#define WMMA_K 8

#define M_BATCH 16
#define N_BATCH 32
#define K_BATCH 32

template <int NWARPS>
__global__ void
linear_kernel_ptr(const float *__restrict__ X, const float *__restrict__ W,
                  float *__restrict__ OUT, const int NNODES, const int M,
                  const int N, const int K, const int L) {

  const int cCol = blockIdx.y;

  extern __shared__ char buffer[];

  void *sptr = buffer;
  unsigned int space = 0;

  float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
  float *buffer_out =
      shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

  const float path_weight = 1.0f / sqrt((float)K);

  const int lstart = L * L;
  const int nl = 2 * L + 1;

  W += L * K * N; // move W to the correct weights sub-matrix

  const int threadCol = threadIdx.x;
  const int threadRow = threadIdx.y;
  const int rowStrideB = blockDim.y;
  const int nmiter = find_integer_divisor(M, rowStrideB);

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::col_major>
      a_frag, delta_a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::row_major>
      b_frag, delta_b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

  wmma::fill_fragment(ab_frag, 0.0f);

  const int bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

  for (int bkIdx = 0; bkIdx < K; bkIdx += WMMA_K) // 0, 16, 32
  {
    __syncthreads();

    if (bkIdx % 32 == 0) {
      for (int m = 0; m < nmiter; m++) {

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

  for (int n_block = 0;
       n_block <
       min(N - cCol * (blockDim.y * WMMA_N), blockDim.y * WMMA_N) / blockDim.x;
       n_block++) {
    for (int m = 0; m < nmiter; m++) {

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
}
template <int NWARPS>
__global__ void elemental_linear_kernel_ptr(
    const float *__restrict__ X, const float *__restrict__ W,
    const long *__restrict__ node_idx, const int nselected,
    const int element_id, const int nelements, float *__restrict__ OUT,
    const int NNODES, const int M, const int N, const int K, const int L) {

  const int cCol = blockIdx.y;

  extern __shared__ char buffer[];

  void *sptr = buffer;
  unsigned int space = 0;

  float *Xs = shared_array<float>(K_BATCH * (K_BATCH + 1), sptr, &space);
  float *buffer_out =
      shared_array<float>(M * blockDim.y * WMMA_N, sptr, &space);

  const float path_weight = 1.0f / sqrt((float)K);

  const int lstart = L * L;
  const int nl = 2 * L + 1;

  W += element_id * 4 * K * N +
       L * K * N; // move W to the correct weights sub-matrix

  const int threadCol = threadIdx.x; // [0-32]
  const int threadRow = threadIdx.y; //  128 / 32 = [0-4]
  const int rowStrideB = blockDim.y;
  const int nmiter = find_integer_divisor(M, rowStrideB);

  wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::col_major>
      a_frag, delta_a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, wmma::precision::tf32,
                 wmma::row_major>
      b_frag, delta_b_frag;
  wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> ab_frag;

  wmma::fill_fragment(ab_frag, 0.0f);

  const int bCol = (blockIdx.y * blockDim.y + threadIdx.y) * WMMA_N;

  for (int bkIdx = 0; bkIdx < K; bkIdx += WMMA_K) // 0, 16, 32
  {
    __syncthreads();

    if (bkIdx % 32 == 0) {
      for (int m = 0; m < nmiter; m++) {
        int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;

        if (m * rowStrideB + threadRow < 16 && gid / nl < nselected) {
          int lidx = lstart + node_idx[(gid / nl)] * 16 + gid % nl;

          if (lidx < NNODES * 16) {
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

  for (int n_block = 0;
       n_block <
       min(N - cCol * (blockDim.y * WMMA_N), blockDim.y * WMMA_N) / blockDim.x;
       n_block++) {
    for (int m = 0; m < nmiter; m++) {

      int gid = blockIdx.x * 16 + m * rowStrideB + threadRow;

      if (m * rowStrideB + threadRow < 16 && gid / nl < nselected) {

        int lidx = lstart + node_idx[(gid / nl)] * 16 + gid % nl;

        if (lidx < NNODES * 16 &&
            cCol * (blockDim.y * WMMA_N) + n_block * 32 + threadCol < N) {
          OUT[lidx * N + cCol * (blockDim.y * WMMA_N) + n_block * 32 +
              threadCol] =
              path_weight *
              buffer_out[(m * rowStrideB + threadRow) * (blockDim.y * WMMA_N) +
                         n_block * 32 + threadCol];
        }
      }
    }
  } 
}