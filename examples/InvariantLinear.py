# Implementation of the linear layer
from time import time
from typing import List
from math import prod
import torch
from e3nn import o3
from mace_ops.ops.linear import Linear

torch.backends.cuda.matmul.allow_tf32 = False

    
class LinearRef(torch.nn.Module):

    def __init__(self, irreps_in, irreps_out, e3nn_instructions, e3nn_weights):

        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        self.e3nn_instructions = e3nn_instructions
        self.e3nn_weights = e3nn_weights

        self.out_lmax = int(irreps_out.lmax)
        self.out_dim = int(irreps_out.dim / (self.out_lmax + 1) ** 2)

        self.instructions = []

        flat_weight_index = 0
        for ins in e3nn_instructions:
            path_nweight = prod(ins.path_shape)
            mul_ir_out = irreps_out[ins.i_out]
            # extract the weights for the current path
            w = e3nn_weights.narrow(-1, flat_weight_index, path_nweight)
            w = w.reshape(ins.path_shape)
            # 0 | 1 2 3 | 4 5 6
            start = ins.i_in ** 2
            end = start + (2 * ins.i_in + 1)

            self.instructions.append((start, end, w, ins.path_weight))

            flat_weight_index += path_nweight

    def forward(self, x):

        output = torch.zeros(x.shape[0], (self.out_lmax + 1) ** 2, self.out_dim,
                             device='cuda', dtype=torch.float32)

        for i, instruction in enumerate(self.instructions):

            start_l_idx, end_l_idx, weights, path_weight = instruction

            output[:, start_l_idx:end_l_idx, :] = path_weight * \
                torch.matmul(x[:, start_l_idx:end_l_idx, :], weights)

        return output

class reshape_irreps(torch.nn.Module):
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, ix : ix + mul * d]  # [batch, sample, mul * repr]
            ix += mul * d
            field = field.reshape(batch, mul, d)
            out.append(field)
        return torch.cat(out, dim=-1)

class unreshape_irreps(torch.nn.Module):
    # This is the inverse of reshape_irreps
    def __init__(self, irreps: o3.Irreps) -> None:
        super().__init__()
        self.irreps = o3.Irreps(irreps)
        self.dims = []
        self.muls = []
        for mul, ir in self.irreps:
            d = ir.dim
            self.dims.append(d)
            self.muls.append(mul)
    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        ix = 0
        out = []
        batch, _, _ = tensor.shape
        for mul, d in zip(self.muls, self.dims):
            field = tensor[:, :, ix : ix + d]
            ix += d
            field = field.reshape(batch, -1)
            print("field shape", field.shape)
            out.append(field)
        return torch.cat(out, dim=-1)

# INPUTS#
n_channels = 96
n_out_channels = 96
max_l = 3
nnodes = 5000

x = torch.randn(nnodes, (max_l+1)**2, n_channels,
                device='cuda', dtype=torch.float32, requires_grad=True)
x_ref = x.clone().detach().requires_grad_(True)

## E3NN LINEAR##
irreps_in = o3.Irreps(
    (n_channels * o3.Irreps.spherical_harmonics(max_l))
    .sort()
    .irreps.simplify()
)
irreps_out = o3.Irreps(
    f"{n_out_channels}x0e + {n_out_channels}x1o + {n_out_channels}x2e + {n_out_channels}x3o")

unrespahe_ = unreshape_irreps(irreps_in)
reshape_ = reshape_irreps(irreps_out)
#need o3 linear to pull weights asnd instructions
linear = o3.Linear(irreps_in=irreps_in, irreps_out=irreps_out).to('cuda')
print(linear)
print("irreps_in", irreps_in)
print("x shape", x.shape)
linear_e3nn_out = reshape_(linear(unrespahe_(x.permute(0,2,1))))
print("linear_e3nn out", linear_e3nn_out.shape)

instructions = linear.instructions
ws = linear.weight

#reference linear#
linear_ref = LinearRef(irreps_in, irreps_out, instructions, ws)

##CUDA LINEAR##
linear_cuda = Linear(irreps_in, irreps_out, instructions, ws)

torch.cuda.cudart().cudaProfilerStart()
torch.cuda.synchronize()
start = time()
for i in range(1):
    cuda_out = linear_cuda(x)
    t = cuda_out.sum() 
    t.backward()
end = time()

torch.cuda.cudart().cudaProfilerStop()
print("fwd CUDA linear:", end - start)

linear_ref = linear_ref(x_ref)
linear_ref.sum().backward()
torch.cuda.synchronize()

torch.set_printoptions(precision=5)

error_e3nn_ref = (linear_e3nn_out.permute(0,2,1) - linear_ref).abs().mean()
print("error e3nn ref", error_e3nn_ref)
idx = torch.where (linear_ref - cuda_out > 1e-5)
print("error e3nn cuda", (linear_e3nn_out.permute(0,2,1) - cuda_out).abs().mean())
if (len(idx[0]) > 0):
    print ("Possible issues with precision of output...")
    print (idx)
    print (linear_ref[idx])
    print (cuda_out[idx])

idx = torch.where (x_ref.grad - x.grad > 1e-5)

if (len(idx[0]) > 0):
    print ("Possible issues with precision of grad X...")
    print (idx)
    print (x.grad[idx])
    print (x_ref.grad[idx])

model = torch.compile(linear_cuda)
print (model)

assert torch.allclose(linear_ref, cuda_out, atol=1e-5)
assert torch.allclose(x_ref.grad, x.grad, atol=1e-5)
