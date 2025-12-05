from typing import Optional

import torch
from torch_geometric.nn import knn_graph
import torch.nn as nn
from torch_scatter import scatter_sum, scatter_mean
from ..nn_coarsen.clustering import voxel_clustering
from ..nn_coarsen.schedule import linear_schedule, exp_schedule
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


class FeatureAggregationLayer(InteractionLayer):
    """
    Fine nodes to coarse nodes
    """
    def __init__(self, node_dim: int, edge_dim: int):
        # Initialize the parent class (InteractionLayer)
        # This automatically sets up self.W, self.msg_nn, self.ln, etc.
        super().__init__(node_dim, edge_dim)

    def forward(
        self,
        node_states_s: torch.Tensor,
        node_states_v: torch.Tensor,
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
        n_super_nodes: int
    ):
        src_idx, dst_idx = edge_node_index

        node_states_s, node_states_v = self.ln(
            node_states_s, node_states_v, node_index
        )

        
        W = self.W(edge_states)
        phi = self.msg_nn(node_states_s)
        Wphi = W * phi[src_idx] 
        
        phi_s, phi_vv, phi_vs = torch.split(Wphi, self.node_dim, dim=1)
        edge = self.edge_inference_nn(phi_s)
        
        messages_s = phi_s * edge
        messages_v = (
            node_states_v[src_idx] * phi_vv[:, None, :]
            + phi_vs[:, None, :] * unit_vectors[..., None]
        ) * edge[..., None]

        s_coarse = scatter_mean(messages_s, dst_idx, dim=0, dim_size=n_super_nodes)
        v_coarse = scatter_mean(messages_v, dst_idx, dim=0, dim_size=n_super_nodes)


        return s_coarse, v_coarse


class FeatureAggregator(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, layers: int = 3):
        super().__init__()
        self.layers = layers
        self.update_layers = nn.ModuleList(
            [UpdateLayer(node_dim) for _ in range(layers)]
        )
        self.aggregation_layers = nn.ModuleList(
            [FeatureAggregationLayer(node_dim, edge_dim) for _ in range(layers)]
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(node_dim) for _ in range(layers)]
        )

    def forward(
        self,
        node_states_s: torch.Tensor, 
        node_states_v: torch.Tensor, 
        edge_states: torch.Tensor,
        unit_vectors: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
        n_super_nodes: int
    ):
    
        s_coarse = torch.zeros(
            (n_super_nodes, node_states_s.shape[-1]), 
            device=node_states_s.device, 
            dtype=node_states_s.dtype
        )

      
        v_coarse = torch.zeros(
            (n_super_nodes, *node_states_v.shape[1:]), 
            device=node_states_v.device, 
            dtype=node_states_v.dtype
        )

        for i,(agg_layer, update_layer) in enumerate(zip(self.aggregation_layers, self.update_layers)):
            delta_s, delta_v = agg_layer(
                node_states_s, 
                node_states_v, 
                edge_states,
                unit_vectors,
                node_index,
                edge_node_index,
                n_super_nodes
            )
            scale = 1.0 / (self.layers ** 0.5)
            
            s_coarse = s_coarse + (delta_s * scale)
            v_coarse = v_coarse + (delta_v * scale)
            
            s_updated, v_updated = update_layer(s_coarse, v_coarse)
            
            s_coarse = self.layer_norms[i](s_updated)
            v_coarse = v_updated

        return s_coarse, v_coarse   
    

class SpreadingLayer(InteractionLayer):
  
    def __init__(self, hid_dim: int, edge_dim: int, ):
        super().__init__(hid_dim, edge_dim)
     

    def forward(
        self,
        s_coarse: torch.Tensor,
        v_coarse: torch.Tensor,       
        edge_states: torch.Tensor,    
        unit_vectors: torch.Tensor,   
        bipartite_edge_index: torch.Tensor, 
        n_fine: int                  
    ):
        src_idx, dst_idx = bipartite_edge_index 

        W = self.W(edge_states)
        phi = self.msg_nn(s_coarse)
        Wphi = W * phi[src_idx] 
        
        phi_s, phi_vv, phi_vs = torch.split(Wphi, self.node_dim, dim=1)
        edge_gate = self.edge_inference_nn(phi_s)

     
        msg_s = phi_s * edge_gate
        msg_v = (
            v_coarse[src_idx] * phi_vv[:, None, :] 
            + phi_vs[:, None, :] * unit_vectors[..., None]
        ) * edge_gate[..., None]

        # Output size is n_fine (N)
        s_fine = scatter_mean(msg_s, dst_idx, dim=0, dim_size=n_fine)
        v_fine = scatter_mean(msg_v, dst_idx, dim=0, dim_size=n_fine)

        return s_fine, v_fine


class FeatureSpreader(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int, layers: int = 3):
        super().__init__()
        # Note: 'layers' argument is kept for compatibility but ignored 
        # to strictly enforce the single-update logic.
        
        # Single Spreading Layer (Coarse -> Fine)
        self.spreading_layer = SpreadingLayer(node_dim, edge_dim)
        
        # Single Update Layer (Fine -> Fine)
        self.update_layer = UpdateLayer(node_dim)
        
        # Single Layer Norm
        self.layer_norm = nn.LayerNorm(node_dim)

    def forward(
        self,
        s_coarse: torch.Tensor,       
        v_coarse: torch.Tensor,       
        edge_states: torch.Tensor,    
        unit_vectors: torch.Tensor,   
        bipartite_edge_index: torch.Tensor, 
        n_fine: int                   
    ):
        
    
        delta_s, delta_v = self.spreading_layer(
            s_coarse,
            v_coarse,
            edge_states,
            unit_vectors,
            bipartite_edge_index,
            n_fine
        )

        delta_s, delta_v = self.update_layer(delta_s, delta_v)
        delta_s = self.layer_norm(delta_s)

        return delta_s, delta_v
class EquivEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 128,
        time_embedding: Optional[FourierEmbedding] = None,
        edge_embedding: Optional[EdgeEmbedding] = None,
        num_layers: int = 4,
        h_input_dim: int = 100,
        smooth_h: bool = True,
        k:int=15,
        num_coarsen_layers:int=3,):
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

        self.aggregators = nn.ModuleList(
            [
                FeatureAggregator(
                    hidden_dim,
                    edge_embedding.out_features,
                    layers=num_coarsen_layers,
                )
                for _ in range(num_coarsen_layers)
            ]
        )

        self.spreaders = nn.ModuleList(
            [
                FeatureSpreader(
                    hidden_dim,
                    edge_embedding.out_features,
                    layers=num_coarsen_layers,
                )
                for _ in range(num_coarsen_layers)
            ]
        )

        self.k = k

    def graph_cluster(self, pos, node_index, t):
        list_super_pos = []
        list_super_graphs = []
        list_cluster_idx = [] 
        list_super_edge_index = []
        global_super_node_offset = 0
        graphs = node_index.unique()

        for g_id in graphs:
            mask = (node_index==g_id)
            pos_i = pos[mask]
            node_index_i = node_index[mask]
            t_i = t[node_index_i].mean() 
            n_i = pos_i.shape[0]
            n_cg, k_i = exp_schedule(t=1-t_i, k=self.k, N=n_i)
            #boundary conditions
            # If the schedule asks for N nodes (or more), identity mapping
            if n_cg >= n_i:  
                cluster_idx_i = torch.arange(n_i, device=pos.device)
                n_new = n_i
                super_pos_i = pos_i
                edge_index_i = knn_graph(super_pos_i, k=k_i)
            else:
                cluster_idx_i, n_new = voxel_clustering(pos_i, n_cg)
                super_pos_i = scatter_mean(pos_i, cluster_idx_i, dim=0)
                
                # Use k_i. Because n_new is small, k_i will be large (>= n_new),
                # automatically resulting in a fully connected graph.
                edge_index_i = knn_graph(super_pos_i, k=k_i)


            # Reindex for batching
            global_cluster_idx = cluster_idx_i + global_super_node_offset
            global_edge_index = edge_index_i + global_super_node_offset
            list_super_pos.append(super_pos_i)
            list_cluster_idx.append(global_cluster_idx)
            list_super_edge_index.append(global_edge_index)
            list_super_graphs.append(torch.full((n_new,), g_id, device=pos.device))

            global_super_node_offset += n_new

        super_pos = torch.cat(list_super_pos, dim=0)
        super_edge_index = torch.cat(list_super_edge_index, dim=1)
        # keeps track of whch batch super_nodes are in
        super_graph = torch.cat(list_super_graphs, dim=0)
        # keeps track of which cluster each nodes are in
        super_cluster_idx = torch.cat(list_cluster_idx, dim=0)
        # adds the super node position to the list of positions
        all_pos = torch.cat([pos, super_pos], dim=0)
        # total amount of nodes and super nodes
        N_total = pos.shape[0]
        # we go through every node and based on our construction the destination (the supernode is n_total
        # elements away from the node's cluster index)
        # src are all the nodes indices from 0 to N-1
        src = torch.arange(N_total, device = pos.device)
        # for every one of these nodes it has the the index of the super_node
        # so at dst[individual ndoe index] -> super_node index
        dst = super_cluster_idx+N_total
        # creates two rows src, dst as compatible with edge_node_index
        # these edges have no features yet so we  must ccompute them
        edge_index_shifted = torch.stack([src, dst], dim=0)
        # we use these indices to look for distances in all_pos so we muse the shifted index
        up_edge_states, up_unit_vec = self.edge_embedding(
            positions=all_pos, 
            edge_index=edge_index_shifted
        )
        # create edges where the node points to the super_node
        up_edge_index = torch.stack([src, super_cluster_idx])
        # edge_index for calcualtion of the super_node to fine_node edge feature
        down_edge_index_shifted = torch.stack([dst, src], dim=0)

        down_edge_states, down_unit_vec = self.edge_embedding(
            positions=all_pos, 
            edge_index=down_edge_index_shifted
        )
        down_edge_index = torch.stack([super_cluster_idx, src], dim=0)
        super_edge_states, super_unit_vec = self.edge_embedding(
            positions=super_pos, edge_index=super_edge_index
        )

        return {
            "down_edge_states": down_edge_states,
            "down_unit_vec": down_unit_vec,
            "up_unit_vec":up_unit_vec,
            "up_edge_states": up_edge_states,
            "up_edge_index": up_edge_index,
            "down_edge_index": down_edge_index,
            "super_node_index": super_graph,
            "super_pos": super_pos,
            "super_edge_index": super_edge_index,
            "super_edge_states": super_edge_states,
            "super_unit_vec": super_unit_vec,
        }

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: Optional[torch.Tensor],
    ) -> dict[str, torch.Tensor]:

        t_emb = self.time_embedding(t)
        t_per_atom = t_emb[node_index]
        N_total = pos.shape[0]
        node_states_v = pos.new_zeros((*pos.shape, self.hidden_dim))
        node_states_s = self.node_embedding(h)
        node_states_s = torch.cat([node_states_s, t_per_atom], dim=1)
        node_states_s = self.node_time_projection(node_states_s)
        clustered_batch  =self.graph_cluster(pos=pos, node_index=node_index, t=t)

        super_node_index = clustered_batch["super_node_index"]
        super_edge_index = clustered_batch["super_edge_index"]
        down_edge_states = clustered_batch["down_edge_states"]
        down_unit_vec = clustered_batch["down_unit_vec"]
        down_edge_index = clustered_batch["down_edge_index"]
        up_edge_states = clustered_batch["up_edge_states"]
        up_unit_vec = clustered_batch["up_unit_vec"]
        up_edge_index = clustered_batch["up_edge_index"]
        super_pos = clustered_batch["super_pos"]
        super_edge_states = clustered_batch["super_edge_states"]
        super_unit_vec = clustered_batch["super_unit_vec"]


        for (
            interaction,
            update,
            aggregator,
            spreader
        ) in zip(self.interactions, self.updates, self.aggregators, self.spreaders):

            # this is essentially the coarsening
            # up
            super_s, super_v = aggregator(
                node_states_s=node_states_s,
                node_states_v=node_states_v,
                edge_states=up_edge_states,
                unit_vectors=up_unit_vec,
                node_index = node_index,
                edge_node_index=up_edge_index,
                n_super_nodes = super_pos.shape[0]
            )

            old_super_s, old_super_v = super_s, super_v
            super_s, super_v = interaction(node_states_s=super_s,
                                                 node_states_v=super_v,
                                                 edge_states= super_edge_states,
                                                 unit_vectors=super_unit_vec,
                                                 node_index=super_node_index,
                                                 edge_node_index= super_edge_index,  
            )
            delta_super_s = super_s-old_super_s
            delta_super_v = super_v-old_super_v
            # down
            delta_node_states_s, delta_node_states_v = spreader(
                s_coarse=delta_super_s,
                v_coarse=delta_super_v,
                edge_states=down_edge_states,
                unit_vectors=down_unit_vec,
                bipartite_edge_index=down_edge_index,
                n_fine=N_total,
            )
            node_states_s=node_states_s+delta_node_states_s
            node_states_v=node_states_v+delta_node_states_v
            node_states_s, node_states_v = update(node_states_s, node_states_v)           
        states = {"s": node_states_s, "v": node_states_v}

        return states
