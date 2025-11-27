import math
import torch

from torch import Tensor
from torch.nn import Parameter

from typing import Any, Callable, Dict, Final, List, Optional, Tuple, Union
from torch_geometric.utils import add_self_loops, scatter
from torch_geometric.nn.pool import voxel_grid
from torch_geometric.nn.pool.pool import pool_batch, pool_edge, pool_pos
from torch_geometric.nn.pool.consecutive import consecutive_cluster
def voxel_clustering(pos, n_nodes_cg):
    # Get the dimensionality of the space
    n_dim = pos.shape[1]
    
    # Calculate the number of voxels per dimension
    voxels_per_dim = int(n_nodes_cg ** (1 / n_dim))
    
    # Calculate the range of the data in each dimension
    min_vals, _ = torch.min(pos, dim=0)
    max_vals, _ = torch.max(pos, dim=0)
    data_range = max_vals - min_vals
    
    # Calculate voxel size for each dimension
    voxel_size = data_range / voxels_per_dim
    
    # Calculate voxel indices for each point
    voxel_indices = ((pos - min_vals) / voxel_size).long()
    
    # Clip the indices to ensure they're within the valid range
    voxel_indices = torch.clamp(voxel_indices, min=0, max=voxels_per_dim-1)
    
    # Calculate the stride for each dimension
    stride = torch.cumprod(torch.tensor([1] + [voxels_per_dim] * (n_dim - 1)), dim=0)
    
    # Calculate flat indices
    flat_indices = (voxel_indices * stride.to(pos.device)).sum(dim=1)
    
    # Assign unique cluster IDs to each unique voxel
    n_new_nodes, cluster_assignment = torch.unique(flat_indices, return_inverse=True)

    return cluster_assignment, len(n_new_nodes)