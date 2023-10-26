import torch
from time import time
from mace_ops import cuda
from mace_ops.cuda.instruction import Instruction, _normalize_instruction_path_weights
from mace_ops.cuda.irreps import Irreps


def reference(X, Y,  radial, lm_to_L, receiver_list, nnodes ):

    output = torch.zeros(nnodes, Y.shape[1], X.shape[1], device=X.device, dtype=X.dtype)

    for i in range (Y.shape[1]):
        
        out = X * Y[:, i][:, None] * radial[:, lm_to_L[i], :]
        
        output[:, i, :].index_add_(0,receiver_list, out )

    return output


def benchmark(dtype, device):

    nedges = 30000
    nnodes = 1000
    nfeatures = 96
    L_MAX = 3
    nl = (L_MAX +1) ** 2


    print(f"--DTYPE: {dtype}")
    print(f"Benchmarking dtype {dtype} and device {device}")
    print(f"nodes: {nnodes} and edges: {nedges}")
    print(f"nfeatures: {nfeatures} and nsphericalharmonics: {nl}")

    X = torch.rand((nedges, nfeatures), dtype=dtype,
                   device=device, requires_grad=True)
    Y = torch.rand((nedges, nl), dtype=dtype,
                   device=device, requires_grad=True)
    radial = torch.randn((nedges, L_MAX+1, nfeatures), dtype=dtype,
                   device=device, requires_grad=True) 

    lm_to_L = torch.tensor([0, 1,1,1,2,2,2,2,2,3,3,3,3,3,3,3]).int().cuda()

    indices = torch.sort(torch.randint(nnodes, (nedges,), device=device))[0]

    indices_cuda = indices.cuda().int()
    
    X_ref = X.clone().detach().requires_grad_(True)
    Y_ref = Y.clone().detach().requires_grad_(True)
    radial_ref = radial.clone().detach().requires_grad_(True)

    torch.matmul(torch.rand(1024, 1024, device='cuda'),torch.rand(1024, 1024, device='cuda'))
    torch.cuda.synchronize()
    
    start = time()
    for i in range (1):
        out  = reference(X_ref, Y_ref, radial_ref, lm_to_L, indices_cuda, nnodes)

        t = out.sum()

        t.backward()
    torch.cuda.synchronize()

    end = time()
    
    print (end - start)

    print ("-- reference grad--")
    #print (out[0])
    print ("x_grad:", X_ref.grad)
    print ("radial_grad:", radial_ref.grad)
    print ("Y_grad:", Y_ref.grad)

    neighbour_cuda = torch.ops.invariant_tp.calculate_neighbours(indices_cuda, nnodes, 64)
    

    Y = Y.clone().detach().transpose(-1, -2).contiguous().requires_grad_(True)

    start = time()
    for i in range (1000):
        out =  torch.ops.invariant_tp.forward_test(
            X,
            Y,
            radial,
            lm_to_L,
            indices_cuda, 
            neighbour_cuda,
            nnodes,
            32,
            4,
            1)
        
    end = time()
    
    print (end - start)

    start = time()
    for i in range (1000):
        out = torch.ops.invariant_tp.forward(
            X,
            Y,
            radial,
            lm_to_L,
            indices_cuda, 
            nnodes)
        
        test = out.sum()

        test.backward()

        
    end = time()
    #print (out[0])

    print ("x_grad:", X.grad / 1000)
    print ("radial_grad:", radial.grad / 1000)
    print ("Y_grad:", Y.grad / 1000 )

    print (end - start)
    
    
if __name__ == "__main__":
    benchmark(torch.float32, "cuda")