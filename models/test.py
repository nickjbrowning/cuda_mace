from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple

import ase
import numpy as np
import torch
from mace.tools import torch_geometric
from mace import data, tools
from e3nn import o3, nn
from mace import modules
from time import time
import inspect
from math import sqrt

from mace.tools.scatter import scatter_sum

from mace.modules.utils import (
    get_edge_vectors_and_lengths,
)

from mace_ops.ops.invariant_message_passing import InvariantMessagePassingTP
from mace_ops.ops.linear import Linear, ElementalLinear
from mace_ops.ops.symmetric_contraction import SymmetricContraction as CUDAContraction

class SymmetricContractionWrapper(torch.nn.Module):
    def __init__(self, symmetric_contractions):
        super().__init__()
        self.symmetric_contractions = symmetric_contractions

    def forward(self, x, y):
        y = y.argmax(dim=-1).int()
        out = self.symmetric_contractions(x, y).squeeze()
        return out

class linear_matmul(torch.nn.Module):
    def __init__(self, linear_e3nn):
        super().__init__()
        num_channels_in = linear_e3nn.__dict__["irreps_in"].num_irreps
        num_channels_out = linear_e3nn.__dict__["irreps_out"].num_irreps
        self.weights = (
            linear_e3nn.weight.data.reshape(num_channels_in, num_channels_out)
            / num_channels_in**0.5
        )

    def forward(self, x):
        return torch.matmul(x, self.weights)

def linear_to_cuda(linear):
    return Linear(
        linear.__dict__["irreps_in"],
        linear.__dict__["irreps_out"],
        linear.instructions,
        linear.weight,
    )

def element_linear_to_cuda(skip_tp):
    #print("elementlinear", skip_tp)
    num_elements = skip_tp.__dict__["irreps_in2"].dim
    n_channels = skip_tp.__dict__["irreps_in1"][0].dim
    lmax = skip_tp.__dict__["irreps_in1"].lmax
    ws = skip_tp.weight.data.reshape(
        [lmax + 1, n_channels, num_elements, n_channels]
    ).permute(2, 0, 1, 3)
    ws = ws.flatten(1) / sqrt(num_elements)
    linear_instructions = o3.Linear(
        skip_tp.__dict__["irreps_in1"], skip_tp.__dict__["irreps_out"]
    )
    return ElementalLinear(
        skip_tp.__dict__["irreps_in1"],
        skip_tp.__dict__["irreps_out"],
        linear_instructions.instructions,
        ws,
        num_elements,
    )

class InvariantInteraction(torch.nn.Module):

    def __init__(self, mace_model, profile=False):
        super().__init__()
        self.linear_up = linear_matmul(mace_model.interactions[0].linear_up)
        self.linear = linear_to_cuda(mace_model.interactions[0].linear)
        self.tp = InvariantMessagePassingTP()
        self.skip_tp = element_linear_to_cuda(mace_model.interactions[0].skip_tp)
        self.avg_num_neighbors = mace_model.interactions[0].avg_num_neighbors
        self.profile = profile
        
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = torch.tensor(node_feats.shape[0])
        
        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::linear up")
        node_feats = self.linear_up(node_feats)

        if (self.profile):
            torch.cuda.nvtx.range_pop()
        
        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::inv tp")
            
        message = self.tp.forward(
            node_feats,
            edge_attrs,
            edge_feats.view(edge_feats.shape[0], -1, node_feats.shape[-1]),
            sender.int(),
            receiver.int(),
            num_nodes,
        )
        
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::message linear")
            
        message = self.linear(message) / self.avg_num_neighbors
        
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::skip tp")
            
        message = self.skip_tp(message, node_attrs)
    
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        return (
            message,
            None,
        )
        

class InvariantResidualInteraction(torch.nn.Module):

    def __init__(self, mace_model, profile=False):
        super().__init__()
        
        self.linear_up = linear_matmul(mace_model.interactions[1].linear_up)
        self.linear = linear_to_cuda(mace_model.interactions[1].linear)
        self.tp = InvariantMessagePassingTP()
        self.skip_tp = mace_model.interactions[1].skip_tp
        self.avg_num_neighbors = mace_model.interactions[1].avg_num_neighbors
        self.profile = profile
        
    def forward(
        self,
        node_attrs: torch.Tensor,
        node_feats: torch.Tensor,
        edge_attrs: torch.Tensor,
        edge_feats: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> Tuple[torch.Tensor, None]:
        
        sender = edge_index[0]
        receiver = edge_index[1]
        num_nodes = torch.tensor(node_feats.shape[0])
        
        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::skip tp")
        sc = self.skip_tp(node_feats, node_attrs)
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::linear up")
        node_feats = self.linear_up(node_feats)

        if (self.profile):
            torch.cuda.nvtx.range_pop()
        
        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::inv tp")
            
        message = self.tp.forward(
            node_feats,
            edge_attrs,
            edge_feats.view(edge_feats.shape[0], -1, node_feats.shape[-1]),
            sender.int(),
            receiver.int(),
            num_nodes,
        )
        
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("InvariantInteraction::message linear")
            
        message = self.linear(message) / self.avg_num_neighbors
        
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        return (
            message,
            sc,
        )

class OptimizedInvariantMACE(torch.nn.Module):
    def __init__(
        self,
        mace_model : torch.nn.Module,
        profile=False
    ):
        super().__init__()
        self.register_buffer(
            "atomic_numbers", mace_model.atomic_numbers
        )
        self.register_buffer(
            "r_max", mace_model.r_max
        )
        self.register_buffer(
            "num_interactions", mace_model.num_interactions
        )

        self.node_embedding = mace_model.node_embedding
        self.radial_embedding = mace_model.radial_embedding
        
        self.spherical_harmonics = torch.classes.spherical_harmonics.SphericalHarmonics()

        # Interactions and readout
        self.atomic_energies_fn = mace_model.atomic_energies_fn
    
        self.interactions = torch.nn.ModuleList([InvariantInteraction(mace_model, profile), InvariantResidualInteraction(mace_model, profile)])
        self.products = mace_model.products
        self.readouts = mace_model.readouts
        
        for i in range(mace_model.num_interactions):
            symm_contract = mace_model.products[i].symmetric_contractions
            all_weights = {}
            for j in range(len(symm_contract.contractions)):
                all_weights[str(j)] = {}
                all_weights[str(j)][3] = (
                    symm_contract.contractions[j].weights_max.detach().clone().type(torch.float32)
                )
                all_weights[str(j)][2] = (
                    symm_contract.contractions[j].weights[0].detach().clone().type(torch.float32)
                )
                all_weights[str(j)][1] = (
                    symm_contract.contractions[j].weights[1].detach().clone().type(torch.float32)
                )
            
            irreps_in = o3.Irreps(mace_model.products[i].symmetric_contractions.irreps_in)
            coupling_irreps = o3.Irreps([irrep.ir for irrep in irreps_in])
            irreps_out = o3.Irreps(mace_model.products[i].symmetric_contractions.irreps_out)
            
            symmetric_contractions = CUDAContraction(
                coupling_irreps,
                irreps_out,
                all_weights,
                nthreadX=32,
                nthreadY=4,
                nthreadZ=1,
                dtype=torch.float32,
            )
            self.products[i].symmetric_contractions = SymmetricContractionWrapper(
                symmetric_contractions
            )
            self.products[i].linear = linear_matmul(mace_model.products[i].linear)

        r,h = np.linspace(1e-12, self.r_max.item() + 1.0, 128, retstep=True)
        r = torch.tensor(r, dtype=torch.float32).to("cuda")
        print (r)
        bessel_j = self.radial_embedding(r.unsqueeze(-1)) 
        
        self.edge_splines  = []
        for i, interaction in enumerate(mace_model.interactions):
            R = interaction.conv_tp_weights(bessel_j)
            print ("R", R)
            spline = torch.classes.cubic_spline.CubicSpline(r.cuda(), R.cuda())
            print ("spline coeffs:", spline.get_coefficients())
            self.edge_splines.append(spline)

        self.profile = profile
        self.orig_model = mace_model

    def forward(
        self,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, Optional[torch.Tensor]]:
        # Setup
        num_graphs = data["ptr"].numel() - 1

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::atomic_energies")
        # Atomic energies
        node_e0 = self.atomic_energies_fn(data["node_attrs"])
        # node_e0 = self.atomic_energies_fn(data["node_attrs"])
        e0 = scatter_sum( src=node_e0, index=data["batch"], dim=-1, dim_size=num_graphs) # [n_graphs,]
            
        if (self.profile):
            torch.cuda.nvtx.range_pop()
        # Embeddings

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::embeddings")
        #print ("node attrs:", data["node_attrs"].shape)
        node_feats = self.node_embedding (data["node_attrs"])
        
        if (self.profile):
            torch.cuda.nvtx.range_pop()
            
        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::edge vectors")
        vectors, lengths = get_edge_vectors_and_lengths(
            positions=data["positions"],
            edge_index=data["edge_index"],
            shifts=data["shifts"],
        )
        if (self.profile):
            torch.cuda.nvtx.range_pop()

        if (self.profile):
            torch.cuda.nvtx.range_push("MACE::sph")

        edge_attrs = self.spherical_harmonics.forward(vectors)
        
        print (edge_attrs)
        
        if (self.profile):
            torch.cuda.nvtx.range_pop()
            
        # Interactions
        energies = [e0]
        node_energies_list = [node_e0]
        node_feats_list = []
        for j, (interaction, product, readout, edge_spline) in enumerate(zip(
            self.interactions, self.products, self.readouts, self.edge_splines)
        ):
            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::edge spline")
            
            print (torch.sort(lengths.squeeze(-1)))
            
            edge_feats = edge_spline.forward(lengths.squeeze(-1))
            
            print (j, "edge feats", edge_feats)
            
            if (self.profile):
                torch.cuda.nvtx.range_pop()
            
            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::interaction: {}".format(j))

            node_feats, sc = interaction(
                node_attrs=data["node_attrs"],
                node_feats=node_feats,
                edge_attrs=edge_attrs,
                edge_feats=edge_feats,
                edge_index=data["edge_index"],
            )

            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::product: {}".format(j))
            
            print (node_feats.shape, data["node_attrs"].shape)
            if (sc is not None):
                print (sc.shape)
            node_feats = product(node_feats=node_feats, sc=sc, node_attrs=data["node_attrs"]
            )
            
            if (self.profile):
                torch.cuda.nvtx.range_pop()

            if (self.profile):
                torch.cuda.nvtx.range_push("MACE::readout: {}".format(j))
                
            node_feats_list.append(node_feats)
            node_energies = readout(node_feats).squeeze(-1)  # [n_nodes, ]
            energy = scatter_sum(
                src=node_energies,
                index=data["batch"],
                dim=-1,
                dim_size=num_graphs,
            )  # [n_graphs,]
            energies.append(energy)
            node_energies_list.append(node_energies)
            if (self.profile):
                torch.cuda.nvtx.range_pop()

        if (self.profile):
                torch.cuda.nvtx.range_push("MACE:: node sum")

        # Sum over energy contributions
        contributions = torch.stack(energies, dim=-1)
        total_energy = torch.sum(contributions, dim=-1)  # [n_graphs, ]

        if (self.profile):
            torch.cuda.nvtx.range_pop()
        # Outputs

        return total_energy


def build_parser():
    """
    Create a parser for the command line tool.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Optimize a MACE model for CUDA inference."
    )
    parser.add_argument("--model", type=str, help="Path to the MACE model.")
    parser.add_argument(
        "--output",
        type=str,
        default="optimized_model.model",
        help="Path to the output file.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="float32",
        help="Default dtype of the model.",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )
    parser.add_argument(
        "--accuracy",
        action="store_true",
        help="Benchmark the optimized model.",
        default=False,
    )
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="",
        help="Path to the benchmark file.",
    )
    return parser


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = False
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient

def accuracy(
    model,
    model_opt,
    size=2,
) -> None:
    from ase import build
    # build very large diamond structure
    atoms = build.bulk("C", "diamond", a=3.567, cubic=True)
    atoms_list = [atoms.repeat((size, size, size))]
    print("Number of atoms", len(atoms_list[0]))

    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = tools.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=model.r_max.item()
            )
            for config in configs
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader)).to("cuda")
    print("num edges", batch.edge_index.shape)

    for batch in data_loader:
        batch = batch.to("cuda")
        batch['positions'] = batch['positions'] + 0.05*torch.randn(batch['positions'].shape, dtype=batch['positions'].dtype, device=batch['positions'].device)
        
        print("num nodes", batch.num_nodes)
        batch_2 = batch.clone()
        batch_3 = batch.clone()
        # make a copy of the model
        batch['positions'].requires_grad_(True)
        batch["node_attrs"].requires_grad_(True)
        output_opt = model_opt(batch.to_dict())
        forces_opt = compute_forces(output_opt, batch['positions'], training=False)
        
        model = model.float()
        output_org_float32 = model(
            batch_3.to_dict(), training=False, compute_force=True)
        model = model.double()
        model.atomic_energies_fn.atomic_energies = (
            model.atomic_energies_fn.atomic_energies.double()
        )
        batch_2.positions = batch_2.positions.double()
        batch_2.node_attrs = batch_2.node_attrs.double()
        batch_2.shifts = batch_2.shifts.double()
        output_org = model(batch_2.to_dict(),
                           training=False, compute_force=True)
        
        batch_2.positions.requires_grad=True
        forces_org =  compute_forces(output_org, batch_2.positions, training=False)
        
        print("energy org", output_org)
        print("energy opt", output_opt)
        # error_energy = (output_org["energy"] -
        #                 output_opt["energy"]).abs().mean()
        # error_energy_float32 = (
        #     (output_org_float32["energy"] - output_org["energy"]).abs().mean()
        # )
        # print("error energy float32", error_energy_float32)
        # print("error energy", error_energy)
        # # assert torch.allclose(output_org["energy"], output_opt["energy"], atol=1e-5)
        # error_forces = (forces_org -
        #                 output_opt["forces"]).abs().mean()
        # print("error forces", error_forces)
        # error_forces_float32 = (
        #     (output_org_float32["forces"] - output_org["forces"]).abs().mean()
        # )
        # print (output_org_float32["forces"])
        # print (output_org["forces"])
        # print (output_opt["forces"])
        
        # print("error forces float32", error_forces_float32)
        # # compute relative forces error
        # error_relative_forces = (
        #     output_org["forces"] - output_opt["forces"]
        # ).abs().mean() / output_org["forces"].abs().mean()
        # # error mean each component relative
        # error_relative_components = (
        #     (
        #         (output_org["forces"] - output_opt["forces"])
        #         / (output_org["forces"] + 10e-9)
        #     )
        #     .abs()
        #     .mean()
        # )
        # conservative_error = output_opt["forces"].sum()
        # print("conservative error", conservative_error)
        # print("error relative forces", error_relative_forces)
        # print("error relative components", error_relative_components)
        # # # print("forces opt", output_opt["forces"])
        # # # print("forces org", output_org["forces"])
        # # assert torch.allclose(output_org["forces"], output_opt["forces"], atol=1e-5)

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
        
    model = torch.load(args.model).to("cuda").float()
    
    r,h = np.linspace(1e-12, model.r_max.item() + 1.0, 128, retstep=True)
    r = torch.tensor(r, dtype=torch.float32).to("cuda")
    bessel_j = model.radial_embedding(r.unsqueeze(-1)) 
    
    edge_splines  = []
    for i, interaction in enumerate(model.interactions):
        R = interaction.conv_tp_weights(bessel_j)
        spline = torch.classes.cubic_spline.CubicSpline(r.cuda(), R.cuda(), h)
        out = spline.forward(r)
        
        idx =  (torch.where(out - R > 1e-5))
        print (out[idx] - R[idx])
        
    
    
    