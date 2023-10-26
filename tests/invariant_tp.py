import torch
from time import time
from mace_ops import cuda
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps

def benchmark(dtype, device):

    nedges = 30000
    nnodes = 1000
    nfeatures = 64
    nl = 16

    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")
    a = torch.rand((nedges, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    b = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    indices_cuda = indices.cuda().int()

    start = time()
    for _ in range(1000):
        neighbour_cuda = torch.ops.mace_ops_equivariant_tp.calculate_neighbours(
            indices_cuda, nnodes, 64)
    finish = time()
    print(f"The indices for CUDA implementation took {finish-start:.3f} seconds")

    X = a.clone().detach().cuda().requires_grad_(True)
    Y = b.clone().detach().cuda().cuda().requires_grad_(True).contiguous()

    print (X.shape)
    print (Y.shape)

    print ("forwards_only")

    start = time()
    for i in range (1000):
        #out = torch.ops.invariant_tp.forward(X,Y,indices_cuda, neighbour_cuda, nnodes, 32, 8, 1)
        out = torch.ops.invariant_tp.invariant_outer_product_scattersum(X,Y, indices_cuda, nnodes)

    end = time()
    print (end - start)
    

    start = time()
    for i in range (1000):
        #out = torch.ops.invariant_tp.forward(X,Y,indices_cuda, neighbour_cuda, nnodes, 32, 8, 1)

        out = torch.ops.invariant_tp.invariant_outer_product_scattersum(X,Y, indices_cuda, nnodes)
        
        loss = out.sum()

        loss.backward()
    end = time()
    print (out[0])
    
    print (end - start)
    
    
if __name__ == "__main__":
    benchmark(torch.float32, "cpu")
    # benchmark(torch.float64, "cpu")
    # if torch.cuda.is_available():
    #    benchmark(torch.float32, "cuda")
    #    benchmark(torch.float64, "cuda")