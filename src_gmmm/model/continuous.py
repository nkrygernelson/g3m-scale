from abc import ABC
from typing import Any, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.distributions import DistributionGaussian


class SDE(ABC, nn.Module):

    def diffusion(self, t: torch.Tensor):
        raise NotImplementedError

    def forward_drift(self, t: torch.Tensor, zt: torch.Tensor):
        raise NotImplementedError

    def loc_scale(self, t: torch.Tensor):
        raise NotImplementedError

    def reverse_drift(
        self,
        t: torch.Tensor,
        zt: torch.Tensor,
        score: torch.Tensor,
        eta: Optional[torch.Tensor | float] = None,
    ):
        f = self.forward_drift(t, zt)
        g = self.diffusion(t)

        if eta is None:
            eta = g

        return f - 0.5 * (g**2 + eta**2) * score


class LinearLogSNRVPSDE(SDE):
    def __init__(self, log_snr_min: float = -10.0, log_snr_max: float = 10.0) -> None:
        super().__init__()
        self.register_buffer("_log_snr_min", torch.as_tensor(log_snr_min))
        self.register_buffer("_log_snr_max", torch.as_tensor(log_snr_max))

    def gamma(self, t: torch.Tensor):
        return self._log_snr_min + (self._log_snr_max - self._log_snr_min) * t

    def beta(self, t: torch.Tensor):
        dg_dt = self._log_snr_max - self._log_snr_min
        s2 = torch.sigmoid(self._log_snr_min + dg_dt * t)
        beta = s2 * dg_dt
        return beta

    def diffusion(self, t: torch.Tensor):
        return torch.sqrt(self.beta(t))

    def forward_drift(
        self,
        t: torch.Tensor,
        zt: torch.Tensor,
    ):
        return -0.5 * self.beta(t) * zt

    def loc_scale(self, t: torch.Tensor):
        gamma_t = self.gamma(t)
        loc = torch.sigmoid(-gamma_t) ** 0.5
        scale = torch.sigmoid(gamma_t) ** 0.5

        return loc, scale


class ContinuousDiffusion(nn.Module):
    def __init__(
        self,
        sde: SDE,
        parameterization: Literal["eps", "x0"],
        distribution: DistributionGaussian,
        clamp_pred_in_reverse: Optional[Tuple[float, float]] = None,
    ):
        super(ContinuousDiffusion, self).__init__()
        self.sde = sde
        self.parameterization = parameterization
        self.distribution = distribution
        self.clamp_pred_in_reverse = clamp_pred_in_reverse

    def loss_diffusion(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        t: torch.Tensor,
        *args: Optional[Any],
    ):
        assert pred.shape == target.shape
        return F.mse_loss(pred, target)

    def training_targets(self, t: torch.Tensor, x: torch.Tensor, index: torch.Tensor):
        a, b = self.sde.loc_scale(t)
        eps = self.distribution.sample(index)

        x_t = a * x + b * eps

        if self.parameterization == "eps":
            target = eps
        elif self.parameterization == "x0":
            target = x
        else:
            raise NotImplementedError

        return x_t, target

    @torch.inference_mode()
    def reverse_step(
        self,
        t: torch.Tensor,
        x_t: torch.Tensor,
        pred: torch.Tensor,
        dt: torch.Tensor,
        index: Optional[torch.Tensor] = None,
        **_,
    ):

        if self.clamp_pred_in_reverse:
            assert self.parameterization == "x0"
            pred = torch.clamp(pred, *self.clamp_pred_in_reverse)

        score = self.construct_score(t=t, x_t=x_t, pred=pred)

        drift_dt = self.sde.reverse_drift(t=t, zt=x_t, score=score) * dt
        diff_dt = (
            self.sde.diffusion(t) * torch.randn_like(x_t) * torch.sqrt(torch.abs(dt))
        )

        return x_t + drift_dt + diff_dt

    def construct_score(self, t: torch.Tensor, x_t: torch.Tensor, pred: torch.Tensor):

        loc, scale = self.sde.loc_scale(t)

        if self.parameterization == "eps":
            score = -pred / scale
        elif self.parameterization == "x0":
            score = (loc * pred - x_t) / scale**2
        else:
            raise NotImplementedError

        return score

    @torch.inference_mode()
    def sample_prior(self, index: torch.Tensor):
        return self.distribution.sample(index)
