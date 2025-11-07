import torch
import torch.nn as nn

from ..nn.encoder import EquivEncoder
from ..nn.readout import Readout


class EquivariantParameterization(nn.Module):
    def __init__(
        self,
        encoder: EquivEncoder,
        readout: Readout,
    ):
        super(EquivariantParameterization, self).__init__()
        self.encoder = encoder
        self.readout = readout

    def forward(
        self,
        t: torch.Tensor,
        h: torch.Tensor,
        pos: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ):
        states = self.encoder.forward(
            t=t,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )
        return self.readout.forward(
            t,
            states,
            h=h,
            pos=pos,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )
