import torch

import torch_geometric.transforms as T
import torch_geometric

from torch_geometric.nn.pool import avg_pool_x
from torch_geometric.nn.norm import BatchNorm
from torch_geometric.nn.conv import MessagePassing, GATConv

class STAdaptiveGNN(torch.nn.Module):
    def __init__(self, model, in_dim, hidden_dim, out_dim):
        super(STAdaptiveGNN, self).__init__()
        self.model = model

        self.og_proj = torch_geometric.nn.MLP([in_dim, hidden_dim, hidden_dim, hidden_dim], plain_last=False, norm=BatchNorm(hidden_dim))
        self.cg_proj = torch_geometric.nn.MLP([in_dim, hidden_dim, hidden_dim, hidden_dim], plain_last=False, norm=BatchNorm(hidden_dim))

        self.og_lin = torch.nn.ModuleList(
           [torch_geometric.nn.MLP([hidden_dim, hidden_dim, hidden_dim], plain_last=False, norm=BatchNorm(hidden_dim)) for i in range(self.model.num_layers)]
        )
        self.og_lin.append(torch_geometric.nn.Linear(hidden_dim, out_dim))

        self.gate_proj = torch.nn.Linear(hidden_dim * 2, hidden_dim)
        self.coarsening_ops = torch.nn.ModuleList([FeatureAggregation(hidden_dim, hidden_dim) for _ in range(self.model.num_layers)])
        self.spread_ops = torch.nn.ModuleList([FeatureSpread(hidden_dim, hidden_dim) for _ in range(self.model.num_layers)])

        self.act = torch.nn.GELU()

        self.out = torch_geometric.nn.MLP([hidden_dim, hidden_dim, hidden_dim, out_dim], norm=BatchNorm(hidden_dim))
    
    def forward(self, batch):
        # Embed Inputs
        x_og = self.og_proj(batch.x_og, batch.x_og_batch)
        x_cg = self.cg_proj(batch.x_cg, batch.x_cg_batch)
        
        for i, (conv, norm) in enumerate(zip(self.model.convs, self.model.norms)):
            prev_x_og = x_og
            prev_x_cg = x_cg

            # Coarsening Operation
            x_cg = self.coarsening_ops[i](
              x_og, x_cg, batch.pos_og, batch.pos_cg, 
              batch.og_to_cg_edge_index, batch.og_to_cg_edge_attr
            )

            # Message Passing Operation
            if self.model.supports_edge_attr:
                x_cg = conv(x_cg, batch.edge_index_cg, batch.edge_attr_cg)
            else:
                x_cg = conv(x_cg, batch.edge_index_cg)
            
            # Spreading Operation
            spread = self.spread_ops[i](
              x_cg + prev_x_cg, x_og, batch.pos_cg, batch.pos_og, 
              batch.og_to_cg_edge_index[[1,0],:], batch.og_to_cg_edge_attr
            )

            # Gating Operation
            gate_input = torch.cat([x_og, spread], dim=-1)
            gate = torch.sigmoid(self.gate_proj(gate_input))
            x_og = gate * x_og + (1 - gate) * spread

            # Norm, Act, Lin, and Skip
            if self.model.supports_norm_batch and i < self.model.num_layers - 1:
                x_og = norm(x_og, batch.x_og_batch)
            else:
                x_og = norm(x_og)
            x_og = self.act(x_og)
            x_og = self.og_lin[i](x_og, batch.x_og_batch)
            x_og = x_og + prev_x_og

        # Project back to ambient dimension
        x_og = self.out(x_og, batch.x_og_batch)

        return x_og
    
    def parameters(self):
        return list(super(STAdaptiveGNN, self).parameters())


