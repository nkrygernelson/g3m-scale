import abc
from typing import Literal, Optional, Sequence

import torch
import torch.nn as nn

from ..utils.ops import scatter_center


class Readout(nn.Module, abc.ABC):

    def forward(
        self,
        t: torch.Tensor,
        states: dict[str, torch.Tensor],
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ) -> Sequence[torch.Tensor]:
        raise NotImplementedError


class DataPointReadout(Readout):
    def __init__(
        self,
        hidden_dim: int,
        h_output_dim: Optional[int] = 0,
        pred_h: bool = True,
        pred_pos: bool = True,
        zero_cog: bool = True,
        parameterization: Literal["residual-pos"] = "residual-pos",
    ) -> None:
        super(DataPointReadout, self).__init__()

        if pred_h:
            assert h_output_dim > 0
            self.net_h = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, h_output_dim),
            )

        if pred_pos:
            self.net_pos = nn.Linear(in_features=hidden_dim, out_features=1, bias=False)
            self.zero_cog = zero_cog

        self.pred_h = pred_h
        self.pred_pos = pred_pos
        self.parameterization = parameterization

    def forward(
        self,
        t: torch.Tensor,
        states: dict[str, torch.Tensor],
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: Optional[torch.Tensor],
        edge_node_index: torch.Tensor,
    ) -> dict[str, torch.Tensor]:

        out = dict()

        if self.pred_h:
            out["h"] = self.net_h(states["s"])

        if self.pred_pos:
            out_pos = self.net_pos(states["v"]).squeeze()

            match self.parameterization:
                case "residual-pos":
                    out_pos += pos
                # otherwise raw output is 'out_pos'

            if self.zero_cog:
                out_pos = scatter_center(out_pos, index=node_index)

            out["pos"] = out_pos

        return out
