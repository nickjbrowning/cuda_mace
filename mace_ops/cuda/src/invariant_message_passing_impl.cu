#include "cuda_utils.cuh"
#include "torch_utils.cuh"
#include "invariant_message_passing_impl.cuh"

#include <torch/script.h>
#include <iostream>

using namespace std;
using namespace torch::indexing;
using namespace torch::autograd;

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

#define FULL_MASK 0xffffffff

#define WARP_SIZE 32
#define NWARPS_PER_BLOCK 4

template <typename scalar_t, const int TM, const int TN>
__global__ void forward_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X, // [nnodes nchannels]
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y, // [nedges, (L+1)**2]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sender_list,      //
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,    // which index we need to sum a particular edge into -> monotonically increasing.
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences, // the indexes in reciever_list which deliniate the set of edges per node.
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> output)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    int32_t *buffer_sender = shared_array<int32_t>(512, sptr, &space);

    const uint threadCol = threadIdx.x % WARP_SIZE;
    const uint threadRow = threadIdx.x / WARP_SIZE;

    const uint N = X.size(1);
    const uint edge_start = first_occurences[blockIdx.x];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? receiver_list.size(0) : first_occurences[blockIdx.x + 1];
    const uint node_index = receiver_list[edge_start];

    scalar_t regY[TM] = {0.0};
    scalar_t regX[TN] = {0.0};
    scalar_t regOut[TN * TM] = {0.0};
    scalar_t regC[TN * TM] = {0.0};
    // check if this node has neighbours
    if (edge_end - edge_start == 0)
    {
        return;
    }

    if (edge_end - edge_start > 512)
    {
        printf("WARNING: Cuda Invariant Message Passing only supports at most 512 neighbours, node: %d has %d.\n", node_index, (edge_end - edge_start));
    }

    for (int tid = threadIdx.x; tid < edge_end - edge_start; tid += blockDim.x)
    {
        buffer_sender[tid] = sender_list[edge_start + tid];
    }

    __syncthreads();

    for (int feature = threadCol; feature < N; feature += WARP_SIZE * TN)
    {

        for (int m = threadRow; m < 16; m += NWARPS_PER_BLOCK * TM)
        {
            int32_t lm_index = sqrt(m);

            for (int i = 0; i < TN; i++)
            {
                regX[i] = 0.0;
            }

            for (int j = 0; j < TM; j++)
            {
                regY[j] = 0.0;
            }

            for (int i = 0; i < TN; i++)
            {
                for (int j = 0; j < TM; j++)
                {
                    regC[i * TM + j] = 0.0;
                    regOut[i * TM + j] = 0.0;
                }
            }

            for (uint edge = edge_start; edge < edge_end; edge++)
            {
                for (int i = 0; i < TN; i++)
                {
                    regX[i] = X[buffer_sender[edge - edge_start]][i * WARP_SIZE + feature] * radial[edge][lm_index][i * WARP_SIZE + feature];
                }

                for (int j = 0; j < TM; j++)
                {
                    regY[j] = Y[edge][j * NWARPS_PER_BLOCK + m];
                }

                for (int i = 0; i < TN; i++)
                {
                    for (int j = 0; j < TM; j++)
                    {
                        scalar_t val = regX[i] * regY[j];
                        scalar_t val_compensated = val - regC[i * TM + j];
                        scalar_t tmp_new = regOut[i * TM + j] + val_compensated;
                        regC[i * TM + j] = (tmp_new - regOut[i * TM + j]) - val_compensated;
                        regOut[i * TM + j] = tmp_new;
                    }
                }
            }

            for (int i = 0; i < TN; i++)
            {
                for (int j = 0; j < TM; j++)
                {
                    output[node_index][j * NWARPS_PER_BLOCK + m][i * WARP_SIZE + feature] = regOut[i * TM + j];
                }
            }
        }
    }
}

torch::Tensor forward_gpu(
    torch::Tensor X,
    torch::Tensor Y,
    torch::Tensor radial,
    torch::Tensor sender_list,
    torch::Tensor receiver_list,
    torch::Tensor first_occurences,
    const int64_t nnodes)
{

    const uint nedges = Y.size(0);
    const int nspherical_harm = Y.size(1);
    const int nfeatures = X.size(1);

    TORCH_CHECK(nfeatures % WARP_SIZE == 0, "feature dimension must be a multiple of 32");
    TORCH_CHECK(nspherical_harm == 16, "number of edge spherical harmonics must be 16");
    TORCH_CHECK(nfeatures <= 128, "feature dimension cannot be greater than 128");

    torch::Tensor output = torch::empty({nnodes, nspherical_harm, nfeatures},
                                        torch::TensorOptions()
                                            .dtype(X.dtype())
                                            .device(X.device()));

    dim3 gridDim(nnodes);

    dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "forward_gpu", ([&]
                                  {
            size_t space = 0;
            void *sptr = nullptr;

            //shared_array<double>(16 * NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space); // 64 * 128 * 8 = 65kb of shared memory...too large...
            
            //shared_array<double>(16 * NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space);
            //shared_array<scalar_t>(radial.size(1) * NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space);
            //shared_array<scalar_t>(16 * NWARPS_PER_BLOCK, sptr, &space);
            //shared_array<scalar_t>(NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space);


            shared_array<int32_t>(512, sptr, &space);

            if (nfeatures >= 128)
            {
                forward_kernel<scalar_t, 4, 4><<<gridDim, blockDim, space>>>(
                    X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
            }
            else if (nfeatures == 96)
            {
                forward_kernel<scalar_t, 4, 3><<<gridDim, blockDim, space>>>(
                    X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
            }else if (nfeatures == 64)
            {
                forward_kernel<scalar_t, 4, 2><<<gridDim, blockDim, space>>>(
                    X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
            }else if (nfeatures == 32)
            {
                forward_kernel<scalar_t, 4, 1><<<gridDim, blockDim, space>>>(
                    X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    output.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());
            } }

                                  ));

    cudaDeviceSynchronize();

    return output;
}

template <typename scalar_t, const int TM, const int TN>
__global__ void  backward_edge_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> X,       // [nedges, feat]
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,       // [nedges, m]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,  // [nedges, LMAX, feat]
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in, // [nnodes, m, feat]
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sender_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> gradY,
    torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_radial)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_grad_in = shared_array<scalar_t>(16 * X.size(1), sptr, &space);
    scalar_t *buffer_Y = shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);
    scalar_t *buffer_dY = shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);

    const uint threadCol = threadIdx.x % WARP_SIZE;
    const uint threadRow = threadIdx.x / WARP_SIZE;

    const uint edge_start = first_occurences[blockIdx.x];
    const uint node_index = receiver_list[edge_start];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? receiver_list.size(0) : first_occurences[blockIdx.x + 1];

    const uint N_start = blockIdx.y * TN * WARP_SIZE;

    scalar_t regX[TN] = {0.0};
    scalar_t regW[4 * TN] = {0.0};

    scalar_t regGradW[4 * TN] = {0.0};

    if (edge_end - edge_start == 0)
    {
        return;
    }

    for (int m = 0; m < 16 / NWARPS_PER_BLOCK; m++)
    {
        for (int n = 0; n < TN; n++)
        {
            // if (N_start + n * WARP_SIZE + threadCol < X.size(1))
            buffer_grad_in[(m * NWARPS_PER_BLOCK + threadRow) * X.size(1) + n * WARP_SIZE + threadCol] = grad_in[node_index][m * NWARPS_PER_BLOCK + threadRow][N_start + n * WARP_SIZE + threadCol];
        }
    }

    __syncthreads();

    int niter = find_integer_divisor(edge_end - edge_start, NWARPS_PER_BLOCK);

    for (uint ni = 0; ni < niter; ni++)
    {
        uint edge = edge_start + ni * NWARPS_PER_BLOCK + threadRow;

        if (edge < edge_end)
        {
            uint sender_id = sender_list[edge];

            if (threadCol < 16)
            {
                buffer_Y[threadCol * NWARPS_PER_BLOCK + threadRow] = Y[edge][threadCol];
                buffer_dY[threadCol * NWARPS_PER_BLOCK + threadRow] = 0.0;
            }

            __syncwarp();

            for (int n = 0; n < TN; n++)
            {

                regX[n] = X[sender_id][N_start + n * WARP_SIZE + threadCol];
            }

            for (int n = 0; n < TN; n++)
            {
                for (int L = 0; L < 4; L++)
                {
                    regGradW[L * TN + n] = 0.0;
                    regW[L * TN + n] = radial[edge][L][N_start + n * WARP_SIZE + threadCol];
                }
            }

            for (int L = 0; L < 4; L++)
            {
                uint mstart = L * L;
                uint mend = (L + 1) * (L + 1);

                for (int m = mstart; m < mend; m++)
                {
                    scalar_t sph = buffer_Y[m * NWARPS_PER_BLOCK + threadRow]; // Y[edge][m];

                    scalar_t dgradY = 0.0;

                    for (int n = 0; n < TN; n++)
                    {
                        //  scalar_t gradin = regGradIn[m * TN + n];
                        scalar_t gradin = buffer_grad_in[m * X.size(1) + n * WARP_SIZE + threadCol];
                        scalar_t w = regW[L * TN + n];

                        regGradW[L * TN + n] += sph * regX[n] * gradin;

                        dgradY += gradin * w * regX[n];
                    }

                    for (int offset = 16; offset > 0; offset /= 2)
                    {
                        dgradY += __shfl_down_sync(FULL_MASK, dgradY, offset, WARP_SIZE);
                    }

                    // threadIdx % WARP_SIZE = 0 dgradY contains the derivative of the output wrt. Y
                    if (threadCol == 0)
                        buffer_dY[m * NWARPS_PER_BLOCK + threadRow] = dgradY;
                }
            }
        }

        __syncthreads();

        if (edge < edge_end)
        {
            if (threadCol < 16)
            {
                gradY[edge][threadCol] = buffer_dY[threadCol * NWARPS_PER_BLOCK + threadRow];
            }

            for (int n = 0; n < TN; n++)
            {
                //  write grad_radial
                for (int L = 0; L < 4; L++)
                {
                    grad_radial[edge][L][N_start + n * WARP_SIZE + threadCol] = regGradW[L * TN + n];
                }
            }
        }
    }
}

template <typename scalar_t, const int TN>
__global__ void backward_node_kernel2(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sender_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sorted_sender_idx,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> gradX)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    scalar_t *buffer_out = shared_array<scalar_t>(NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space);
    // scalar_t *buffer_Y = shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);

    const uint threadCol = threadIdx.x % WARP_SIZE;
    const uint threadRow = threadIdx.x / WARP_SIZE;

    const uint edge_start = first_occurences[blockIdx.x];
    const uint node_index = sender_list[sorted_sender_idx[edge_start]];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? sender_list.size(0) : first_occurences[blockIdx.x + 1];

    if (edge_end - edge_start == 0)
    {
        return;
    }

    for (int feature = threadCol; feature < gradX.size(1); feature += WARP_SIZE)
    {
        __syncthreads();

        scalar_t tmp = 0.0;
        scalar_t c = 0.0; // Compensation for lost precision - TODO

        for (int m = threadRow; m < 16; m += NWARPS_PER_BLOCK)
        {
            int32_t lm_index = sqrt(m);

            for (uint edge = edge_start; edge < edge_end; edge++)
            {
                // TODO -> store in smem
                uint sorted_id = sorted_sender_idx[edge];
                uint receiver_id = receiver_list[sorted_id];

                scalar_t gin = grad_in[receiver_id][m][feature];

                scalar_t val = gin * radial[sorted_id][lm_index][feature] * Y[sorted_id][m];
                scalar_t val_compensated = val - c;
                scalar_t tmp_new = tmp + val_compensated;
                c = (tmp_new - tmp) - val_compensated;
                tmp = tmp_new;
            }
        }

        buffer_out[threadRow * WARP_SIZE + threadCol] = tmp;

        __syncthreads();

        /* need to reduce over m here*/
        if (threadRow == 0)
        {
            tmp = 0.0;
            for (int i = 0; i < NWARPS_PER_BLOCK; i++)
            {
                tmp += buffer_out[i * WARP_SIZE + threadCol];
            }

            gradX[node_index][feature] = tmp;
        }
    }
}

template <typename scalar_t, const int TN>
__global__ void  backward_node_kernel(
    const torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> Y,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> radial,
    const torch::PackedTensorAccessor64<scalar_t, 3, torch::RestrictPtrTraits> grad_in,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sender_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> receiver_list,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> sorted_sender_idx,
    const torch::PackedTensorAccessor64<int32_t, 1, torch::RestrictPtrTraits> first_occurences,
    torch::PackedTensorAccessor64<scalar_t, 2, torch::RestrictPtrTraits> gradX)
{

    extern __shared__ char buffer[];

    void *sptr = buffer;
    size_t space = 0;

    double *buffer_out = shared_array<double>(NWARPS_PER_BLOCK * WARP_SIZE, sptr, &space);
    scalar_t *buffer_Y = shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr, &space);

    const uint threadCol = threadIdx.x % WARP_SIZE;
    const uint threadRow = threadIdx.x / WARP_SIZE;

    const uint edge_start = first_occurences[blockIdx.x];
    const uint node_index = sender_list[sorted_sender_idx[edge_start]];
    const uint edge_end = (blockIdx.x == first_occurences.size(0) - 1) ? sender_list.size(0) : first_occurences[blockIdx.x + 1];

    const uint N_start = blockIdx.y * TN * WARP_SIZE;

    scalar_t regGradIn[16 * TN] = {0.0};
    scalar_t regW[4 * TN] = {0.0};
    double regGradX[TN] = {0.0};

    if (edge_end - edge_start == 0)
    {
        return;
    }

    int niter = find_integer_divisor(edge_end - edge_start, NWARPS_PER_BLOCK);

    for (uint ni = 0; ni < niter; ni++)
    {
        __syncthreads();

        uint edge = edge_start + ni * NWARPS_PER_BLOCK + threadRow;

        if (edge < edge_end)
        {
            uint sorted_id = sorted_sender_idx[edge];
            uint receiver_id = receiver_list[sorted_id];
            if (threadCol < 16)
            {
                buffer_Y[threadCol * NWARPS_PER_BLOCK + threadRow] = Y[sorted_id][threadCol];
            }
            __syncwarp();

            for (int m = 0; m < 16; m++)
            {
                for (int n = 0; n < TN; n++)
                {
                    regGradIn[m * TN + n] = grad_in[receiver_id][m][N_start + n * WARP_SIZE + threadCol];
                }
            }

            for (int n = 0; n < TN; n++)
            {
                for (int L = 0; L < 4; L++)
                {
                    regW[L * TN + n] = radial[sorted_id][L][N_start + n * WARP_SIZE + threadCol];
                }
            }

            for (int m = 0; m < 16; m++)
            {
                int L = sqrt(m);
                for (int n = 0; n < TN; n++)
                {
                    regGradX[n] += ((double)regW[L * TN + n]) * ((double)regGradIn[m * TN + n]) * ((double)buffer_Y[m * NWARPS_PER_BLOCK + threadRow]);
                }
            }
        }
    }

    // now need to accumulate partial results from each warp.
    for (int n = 0; n < TN; n++)
    {
        __syncthreads();

        buffer_out[threadRow * WARP_SIZE + threadCol] = regGradX[n];

        __syncthreads();

        if (threadRow == 0)
        {

            double tmp = 0.0;

            for (int j = 0; j < NWARPS_PER_BLOCK; j++)
            {
                tmp += buffer_out[j * WARP_SIZE + threadCol];
            }

            gradX[node_index][N_start + n * WARP_SIZE + threadCol] = tmp;
        }
    }
}

std::vector<torch::Tensor> backward_gpu(torch::Tensor X,
                                        torch::Tensor Y,
                                        torch::Tensor radial,
                                        torch::Tensor grad_in,
                                        torch::Tensor sender_list,
                                        torch::Tensor receiver_list,
                                        torch::Tensor first_occurences,
                                        const int64_t nnodes)
{
    uint nedges = Y.size(0);
    uint nfeatures = X.size(1);

    TORCH_CHECK(X.requires_grad(), "X must require grad for invariant message passing backwards_kernel to be called.");
    TORCH_CHECK(Y.requires_grad(), "Y must require grad for invariant message passing backwards_kernel to be called.");
    TORCH_CHECK(radial.requires_grad(), "radial must require grad for invariant message passing backwards_kernel to be called.");

    torch::Tensor gradRadial = torch::empty_like(radial,
                                                 torch::TensorOptions()
                                                     .dtype(radial.dtype())
                                                     .device(radial.device()));

    torch::Tensor gradX = torch::empty({X.size(0), X.size(1)},
                                       torch::TensorOptions()
                                           .dtype(X.dtype())
                                           .device(X.device()));

    torch::Tensor gradY = torch::empty_like(Y,
                                            torch::TensorOptions()
                                                .dtype(Y.dtype())
                                                .device(Y.device()));

    torch::Tensor sorted_sender_idx = torch::argsort(sender_list).to(torch::kInt32);

    // torch::Tensor sorted_sender_idx = torch::zeros_like(sender_list);
    torch::Tensor first_occurences_node = calculate_first_occurences_gpu_with_sort(sender_list, X.size(0), 64, sorted_sender_idx);

    AT_DISPATCH_FLOATING_TYPES(
        X.type(), "backward_gpu", ([&]
                                   {
        dim3 blockDim(NWARPS_PER_BLOCK * WARP_SIZE, 1, 1);
        dim3 gridDim(nnodes, 1);

        void *sptr = nullptr;
        size_t space = 0;

        shared_array<scalar_t>(16 * X.size(1), sptr, &space);
        shared_array<scalar_t>(2 * NWARPS_PER_BLOCK * 16, sptr, &space); // buffer_Y, buffer_dY

        void *sptr_node = nullptr;
        size_t space_node = 0;

        shared_array<double>(NWARPS_PER_BLOCK * WARP_SIZE, sptr_node, &space_node); 
        shared_array<scalar_t>(NWARPS_PER_BLOCK * 16, sptr_node, &space_node); // buffer_Y, buffer_dY


        if (nfeatures == 96)
        {

            backward_edge_kernel<scalar_t, 4, 3><<<gridDim, blockDim, space>>>(
                X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());


            backward_node_kernel<scalar_t, 3><<<gridDim, blockDim, space_node>>>(
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                sorted_sender_idx.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences_node.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());

        }
        else if (nfeatures == 64)
        {
            backward_edge_kernel<scalar_t, 4, 2><<<gridDim, blockDim, space>>>(
                X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());


            backward_node_kernel<scalar_t, 2><<<gridDim, blockDim, space_node>>>(
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                sorted_sender_idx.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences_node.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
        }
        else if (nfeatures == 32)
        {
            backward_edge_kernel<scalar_t, 4, 1><<<gridDim, blockDim, space>>>(
                X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());


           backward_node_kernel<scalar_t, 1><<<gridDim, blockDim, space_node>>>(
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                sorted_sender_idx.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences_node.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
        }else {
                backward_edge_kernel<scalar_t, 4, 4><<<gridDim, blockDim, space>>>(
                    X.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                    sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    first_occurences.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                    gradY.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                    gradRadial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>());

               backward_node_kernel<scalar_t, 4><<<gridDim, blockDim, space_node>>>(
                Y.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>(),
                radial.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                grad_in.packed_accessor64<scalar_t, 3, torch::RestrictPtrTraits>(),
                sender_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                receiver_list.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                sorted_sender_idx.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                first_occurences_node.packed_accessor64<int32_t, 1, torch::RestrictPtrTraits>(),
                gradX.packed_accessor64<scalar_t, 2, torch::RestrictPtrTraits>());
            } }));

    cudaDeviceSynchronize();

    return {gradX, gradY, gradRadial};
}