from typing import Optional

import ase
import pytorch_lightning as pl
import torch
from ase.data import chemical_symbols

from ..data.utils import atoms_from_tensors
from ..metrics.base import Metrics
from ..model.diffusion import EquivariantDiffusion


def sample_uniform(
    size: int,
    device: torch.device,
    lb: float = 0.0,
    ub: float = 1.0,
):
    """
    NOTE: invert bounds to get sample in (lb, ub], instead of [lb, ub)
    """
    return (lb - ub) * torch.rand(size, device=device) + ub


def sample_antithetic(
    size: int, device: torch.device, lb: float = 0.0, ub: float = 1.0
):
    """
    NOTE: similar trick as above to ensure samples in [lb, ub)
    """
    t0 = sample_uniform(1, device=device)
    t = 1.0 - ((t0 + torch.linspace(0.0, 1.0, size + 1, device=device)[:-1]) % 1.0)
    return (ub - lb) * t + lb


class LitModule(pl.LightningModule):
    def __init__(
        self,
        model: EquivariantDiffusion,
        decoder: list[str] = chemical_symbols,
        n_integration_steps: int = 250,
        lr: float = 1e-4,
        warm_up_steps: int = 0,
        with_ema: bool = True,
        ema_decay: float = 0.999,
        ema_start_step: int = 100,
        antithetic_time_sampling: bool = False,
        loss_during_val: bool = True,
        loss_weights: Optional[dict] = None,
        metrics: Optional[Metrics] = None,
    ):
        super().__init__()
        self.model = model
        if with_ema:
            self.model_ema = torch.optim.swa_utils.AveragedModel(
                model,
                multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(ema_decay),
            )
        else:
            self.model_ema = None

        self.decoder = decoder

        self.metrics = metrics

        self.loss_weights = (
            {"pos": 1.0, "h": 1.0} if loss_weights is None else loss_weights
        )
        self.save_hyperparameters(ignore=["model", "metrics"])

    def basic_step(self, batch) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

        if self.hparams.antithetic_time_sampling:
            t = sample_antithetic(lb=1e-3, size=batch.num_graphs, device=self.device)[
                :, None
            ]
        else:
            t = sample_uniform(lb=1e-3, size=batch.num_graphs, device=self.device)[
                :, None
            ]
        losses = self.model.loss_diffusion(t=t, batch=batch)
        loss = torch.sum(
            torch.stack(
                [self.loss_weights[key] * losses[key] for key in losses], dim=-1
            ),
            dim=-1,
        )

        losses["weighted"] = loss

        return loss, losses

    def sample_step(self, batch):
        # sample
        atoms = self.sample(batch)

        if self.metrics:
            self.metrics.update(atoms)

        return atoms

    def training_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ) -> torch.Tensor:

        loss, losses = self.basic_step(batch)
        self.log_dict({f"train/loss_{key}": torch.mean(losses[key]) for key in losses})

        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        if self.model_ema is not None:
            if self.global_step >= self.hparams.ema_start_step:
                self.model_ema.update_parameters(self.model)

    def validation_step(
        self, batch, batch_idx: int, dataloader_idx: Optional[int] = None
    ):

        if self.hparams.loss_during_val:
            # log loss first
            loss, losses = self.basic_step(batch)
            self.log_dict(
                {f"val/loss_{key}": torch.mean(losses[key]) for key in losses},
                on_epoch=True,
            )

            del loss
            del losses

        # sample
        atoms = self.sample_step(batch)

        return atoms

    def test_step(self, batch, batch_idx):
        return self.sample_step(batch)

    def predict_step(self, batch, batch_idx: int, dataloader_idx: int = 0):
        return self.sample(batch)

    def compute_and_log_metrics(self, stage: str):
        for metrics in [
            self.metrics,
        ]:
            if metrics:
                summary = metrics.summarize()
                self.log_dict(
                    {f"{stage}/{key}": summary[key] for key in summary}, on_epoch=True
                )

    def on_validation_epoch_start(self) -> None:
        for metrics in [
            self.metrics,
        ]:
            if metrics:
                metrics.reset()

    def on_validation_epoch_end(self):
        self.compute_and_log_metrics(stage="val")

    def on_test_epoch_start(self) -> None:
        return self.on_validation_epoch_start()

    def on_test_epoch_end(self):
        self.compute_and_log_metrics(stage="test")

    @torch.no_grad()
    def sample(self, batch, ema: bool = True) -> list[ase.Atoms]:
        model = self.get_model(ema=ema)
        ptr = batch.ptr

        samples = model.sample(batch, n_steps=self.hparams.n_integration_steps)
        atoms = self.atoms_from_tensors(**samples, ptr=ptr)
        return atoms

    def configure_optimizers(self):
        opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.hparams.lr,
            amsgrad=True,
            weight_decay=1e-12,
        )

        warm_up_steps = getattr(self.hparams, "warm_up_steps", 0)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            opt, start_factor=1e-2, total_iters=warm_up_steps
        )

        opt_config = {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
        return opt_config

    def get_model(self, ema: bool = True) -> EquivariantDiffusion:
        if self.model_ema and (ema or self.global_step >= self.hparams.ema_start_step):
            model: EquivariantDiffusion = self.model_ema.module
            print("Using EMA Model.")
        else:
            model: EquivariantDiffusion = self.model
            print("Using current model.")
        return model

    def atoms_from_tensors(self, h: torch.Tensor, pos: torch.Tensor, ptr: torch.Tensor):
        return atoms_from_tensors(h, pos, ptr=ptr, decoder=self.decoder)
