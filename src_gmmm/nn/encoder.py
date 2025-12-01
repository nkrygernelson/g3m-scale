from typing import Optional

import torch
import torch.nn as nn
from torch_scatter import scatter_sum

from ..nn.layers import EdgeEmbedding, EquivLayerNorm, FourierEmbedding


class InteractionLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
    ):
        super(InteractionLayer, self).__init__()
        self.node_dim = node_dim
        self.W = nn.Linear(edge_dim, 3 * node_dim)
        self.msg_nn = nn.Sequential(
            nn.Linear(node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )
        self.edge_inference_nn = nn.Sequential(
            nn.Linear(node_dim, 1),
            nn.Sigmoid(),
        )

        self.ln = EquivLayerNorm(dims=(node_dim, node_dim))

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ):
        src_idx, dst_idx = edge_node_index

        node_states_s, node_states_v = self.ln.forward(
            node_states_s, node_states_v, node_index
        )

        W = self.W(edge_states)
        phi = self.msg_nn(node_states_s)
        Wphi = W * phi[src_idx]  # num_edges, 3*node_size
        phi_s, phi_vv, phi_vs = torch.split(Wphi, self.node_dim, dim=1)
        edge = self.edge_inference_nn(phi_s)
        messages_s = phi_s * edge
        messages_v = (
            node_states_v[src_idx] * phi_vv[:, None, :]
            + phi_vs[:, None, :] * unit_vectors[..., None]
        ) * edge[..., None]

        reduced_messages_s = scatter_sum(
            messages_s, dst_idx, dim=0, out=torch.zeros_like(node_states_s)
        )
        reduced_messages_v = scatter_sum(
            messages_v, dst_idx, dim=0, out=torch.zeros_like(node_states_v)
        )

        return (
            node_states_s + reduced_messages_s,
            node_states_v + reduced_messages_v,
        )


class UpdateLayer(nn.Module):
    def __init__(
        self,
        node_dim: int,
    ):
        super(UpdateLayer, self).__init__()
        self.node_dim = node_dim
        self.UV = nn.Linear(node_dim, 2 * node_dim, bias=False)
        self.UV_nn = nn.Sequential(
            nn.Linear(2 * node_dim, node_dim),
            nn.SiLU(),
            nn.Linear(node_dim, 3 * node_dim),
        )

    def forward(self, node_states_s: torch.Tensor, node_states_v: torch.Tensor):
        UVv = self.UV(node_states_v)  # (n_nodes, 3, 2 * F)
        Uv, Vv = torch.split(UVv, self.node_dim, -1)  # (n_nodes, 3, F)
        Vv_norm = torch.sqrt(
            torch.sum(Vv**2, dim=1) + 1e-6
        )  # norm over spatial components

        a = self.UV_nn(torch.cat((Vv_norm, node_states_s), dim=1))
        a_vv, a_sv, a_ss = torch.split(a, self.node_dim, dim=1)

        inner_prod = torch.sum(Uv * Vv, dim=1)
        delta_s = a_ss + a_sv * inner_prod
        delta_v = a_vv[:, None, :] * Uv  # a_vv.shape = (n_nodes, F)

        return node_states_s + delta_s, node_states_v + delta_v


class EdgeLayer(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, residual: bool = False):
        super().__init__()
        self.node_dim = node_dim
        self.edge_nn = nn.Sequential(
            nn.Linear(edge_dim + 2 * node_dim, 2 * node_dim),
            nn.SiLU(),
            nn.Linear(2 * node_dim, edge_dim),
        )
        self.residual = residual
        self.mask = nn.Parameter(
            torch.as_tensor([1.0 for _ in range(edge_dim)]), requires_grad=True
        )

    def forward(
        self,
        node_states: torch.Tensor,
        edge_states: torch.Tensor,
        edges: torch.LongTensor,
    ):
        concat_states = torch.cat(
            (node_states[edges].view(-1, 2 * self.node_dim), edge_states), axis=1
        )
        if self.residual:
            return self.mask[None, :] * edge_states + self.edge_nn(concat_states)
        else:
            return self.edge_nn(concat_states)


class EquivEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_embedding: Optional[FourierEmbedding] = None,
        edge_embedding: Optional[EdgeEmbedding] = None,
        num_layers: int = 4,
        h_input_dim: int = 100,
        smooth_h: bool = True,
    ):
        super(EquivEncoder, self).__init__()

        # Embedding layers
        self.hidden_dim = hidden_dim
        if smooth_h:
            self.node_embedding = nn.Linear(h_input_dim, hidden_dim, bias=False)
        else:
            # we just need to embed the given discrete h
            self.node_embedding = nn.Embedding(h_input_dim + 1, hidden_dim)

        if time_embedding is None:
            time_embedding = FourierEmbedding(1, hidden_dim, trainable=True)

        self.time_embedding = time_embedding
        self.node_time_projection = nn.Linear(
            hidden_dim + time_embedding.out_features, hidden_dim
        )

        if edge_embedding is None:
            edge_embedding = EdgeEmbedding(num_rbf_features=hidden_dim // 2)

        self.edge_embedding = edge_embedding

        # Interaction layers
        self.interactions = nn.ModuleList(
            [
                InteractionLayer(hidden_dim, edge_embedding.out_features)
                for _ in range(num_layers)
            ]
        )

        # Update layers
        self.updates = nn.ModuleList(
            [UpdateLayer(hidden_dim) for _ in range(num_layers)]
        )

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        t = self.time_embedding(t)
        t_per_atom = t[node_index]

        node_states_v = pos.new_zeros((*pos.shape, self.hidden_dim))
        node_states_s = self.node_embedding(h)
        node_states_s = torch.cat([node_states_s, t_per_atom], dim=1)
        node_states_s = self.node_time_projection(node_states_s)

        edge_states, unit_vectors = self.edge_embedding.forward(
            positions=pos, edge_index=edge_node_index
        )

        for (
            interaction,
            update,
        ) in zip(self.interactions, self.updates):
            node_states_s, node_states_v = interaction.forward(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=edge_states,
                unit_vectors=unit_vectors,
                node_index=node_index,
                edge_node_index=edge_node_index,
            )
            node_states_s, node_states_v = update(node_states_s, node_states_v)

        states = {"s": node_states_s, "v": node_states_v}

        return states