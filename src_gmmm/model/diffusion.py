from typing import Literal, Optional, Union

import torch
import torch.nn as nn
from torch_geometric.data import Batch, Data

from ..model.continuous import ContinuousDiffusion
from ..model.score import EquivariantParameterization


class EquivariantDiffusion(nn.Module):
    def __init__(
        self,
        parameterization: EquivariantParameterization,
        diffusion_pos: Optional[ContinuousDiffusion],
        diffusion_h: Optional[ContinuousDiffusion],
    ):
        super().__init__()

        self.parameterization = parameterization
        self.diffusions = nn.ModuleDict({"pos": diffusion_pos, "h": diffusion_h})

    def loss_diffusion(self, t: torch.Tensor, batch: Batch | Data):
        latents, targets = self.training_targets(t=t, batch=batch)

        preds = self.parameterization.forward(
            t=t,
            **latents,
            node_index=batch.batch,
            edge_node_index=batch.edge_node_index,
        )

        losses = {}

        for key in targets:
            loss = self.diffusions[key].loss_diffusion(
                preds[key],
                targets[key],
                t[batch.batch],  # cast time, when computing node-level property
                latents[key],
            )
            losses[key] = loss

        return losses

    def training_targets(
        self, t: torch.Tensor, batch: Batch | Data
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        targets = {}

        index = batch.batch

        # position part
        if self.diffusion_pos is None:
            pos_t = batch.pos
        else:
            pos_t, target_pos_t = self.diffusion_pos.training_targets(
                t[index], batch.pos, index=batch.batch
            )
            targets["pos"] = target_pos_t

        # atomic species part
        if self.diffusion_h is None:
            h_t = batch.h
        else:
            h_t, target_h_t = self.diffusion_h.training_targets(
                t=t[index], x=batch.h, index=batch.batch
            )
            targets["h"] = target_h_t

        latents = {"pos": pos_t, "h": h_t}

        return latents, targets

    @torch.inference_mode()
    def sample_prior(
        self,
        batch: Batch | Data,
    ):
        index = batch.batch

        if self.diffusion_pos is None:
            assert batch.pos is not None
            pos = batch.pos
        else:
            pos = self.diffusion_pos.sample_prior(index)

        if self.diffusion_h is None:
            assert batch.h is not None
            h = batch.h
        else:
            num_nodes = len(index)
            h = self.diffusions["h"].sample_prior(n=num_nodes)

        return pos, h

    @torch.no_grad()
    def sample(
        self,
        batch: Batch | Data,
        method: Literal["em"] = "em",
        return_traj: bool = False,
        n_steps: int = 1000,
        ts: float = 1.0,
        tf: float = 1e-3,
        **kwargs,
    ) -> Union[
        dict[str, torch.Tensor],
        tuple[dict[str, torch.Tensor], dict[str, list[torch.Tensor]]],
    ]:

        node_index, edge_node_index = batch.batch, batch.edge_node_index
        num_graphs = batch.num_graphs
        device = node_index.device

        ts = torch.linspace(ts, tf, n_steps + 1, device=device)
        pos_t, h_t = self.sample_prior(batch=batch)

        if return_traj:
            traj = {
                "pos": [pos_t],
                "h": [h_t],
            }

        for i in range(n_steps):
            t = ts[i]
            dt = ts[i + 1] - t

            t = torch.full((num_graphs, 1), t, device=device)

            if method == "em":
                pos_t, h_t = self.reverse_step_em(
                    t=t,
                    dt=dt,
                    pos_t=pos_t,
                    h_t=h_t,
                    node_index=node_index,
                    edge_node_index=edge_node_index,
                )

            if return_traj:
                traj["pos"].append(pos_t)
                traj["h"].append(h_t)

        samples = {
            "pos": pos_t,
            "h": h_t,
        }

        if return_traj:
            return samples, traj

        else:
            return samples

    def reverse_step_em(
        self,
        t: torch.Tensor,
        dt: torch.Tensor,
        pos_t: torch.Tensor,
        h_t: torch.Tensor,
        node_index: torch.Tensor,
        edge_node_index: torch.Tensor,
    ):

        # get NN predictions
        preds = self.parameterization.forward(
            t=t,
            pos=pos_t,
            h=h_t,
            node_index=node_index,
            edge_node_index=edge_node_index,
        )
        # reverse step on each modality
        if self.diffusion_pos:
            pos_t = self.diffusion_pos.reverse_step(
                t=t[node_index], x_t=pos_t, pred=preds["pos"], dt=dt, index=node_index
            )

        if self.diffusion_h:
            h_t = self.diffusion_h.reverse_step(
                t=t[node_index], x_t=h_t, pred=preds["h"], dt=dt, index=node_index
            )

        return pos_t, h_t

    @property
    def diffusion_pos(self) -> Optional[ContinuousDiffusion]:
        return self.diffusions["pos"]

    @property
    def diffusion_h(self) -> Optional[ContinuousDiffusion]:
        return self.diffusions["h"]
