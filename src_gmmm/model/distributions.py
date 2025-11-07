import torch
import torch.nn as nn

from ..utils.ops import scatter_center


class DistributionGaussian(nn.Module):
    def __init__(self, dim: int = 3, scale: float = 1.0, zero_cog: bool = True):
        super(DistributionGaussian, self).__init__()
        self.dim = dim
        self.scale = scale
        self.zero_cog = zero_cog

    def sample(self, index: torch.Tensor):
        sample = torch.randn((len(index), self.dim), device=index.device) * self.scale
        if self.zero_cog:
            sample = scatter_center(sample, index=index)

        return sample
