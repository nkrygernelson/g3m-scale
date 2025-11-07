import torch
from torch_scatter import scatter_mean


def scatter_center(pos: torch.Tensor, index: torch.Tensor = None):
    return pos - scatter_mean(pos, index=index, dim=0)[index]


def center(pos: torch.Tensor):
    return pos - torch.mean(pos, dim=0)


def is_centered(
    pos: torch.Tensor, index: torch.Tensor, tol: float = 1e-3, debug: bool = True
):
    com = scatter_mean(pos, index=index, dim=0)
    if debug:
        print("Debug is_centered:", torch.amax(torch.abs(com)))
    return torch.all(com < tol)
