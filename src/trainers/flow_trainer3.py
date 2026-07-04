import lightning as L
import torch
import torch.nn.functional as F

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:
    linear_sum_assignment = None

from src.model.flow import LatentFlow


def greedy_linear_sum_assignment(cost: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = cost.shape
    if n_rows != n_cols:
        raise ValueError("OT coupling expects a square cost matrix")

    row_used = torch.zeros(n_rows, dtype=torch.bool)
    col_used = torch.zeros(n_cols, dtype=torch.bool)
    assignment = torch.empty(n_rows, dtype=torch.long)

    for flat_idx in torch.argsort(cost.flatten()):
        row = int(flat_idx // n_cols)
        col = int(flat_idx % n_cols)
        if row_used[row] or col_used[col]:
            continue
        row_used[row] = True
        col_used[col] = True
        assignment[row] = col
        if bool(row_used.all()):
            break

    return assignment


class LatentFlowMatchingTrainer(L.LightningModule):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        time_dim: int = 128,
        dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        sample_steps: int = 100,
        sample_solver: str = "heun",
        ot_coupling: bool = False,
        fixed_noise: bool = False,
        time_weighting: str = "none",
        cosine_loss_weight: float = 0.0,
        norm_loss_weight: float = 0.0,
        t_eps: float = 0.0,
        scheduler: str = "none",
        onecycle_pct_start: float = 0.05,
        onecycle_final_div_factor: float = 20.0,
        z_mean: torch.Tensor | None = None,
        z_std: torch.Tensor | None = None,
        pca_mean: torch.Tensor | None = None,
        pca_components: torch.Tensor | None = None,
        pca_std: torch.Tensor | None = None,
        pca_residual_components: torch.Tensor | None = None,
        pca_residual_std: torch.Tensor | None = None,
        sample_residual: bool = False,
        noise_codes: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=[
            "z_mean",
            "z_std",
            "pca_mean",
            "pca_components",
            "pca_std",
            "pca_residual_components",
            "pca_residual_std",
            "noise_codes",
        ])
        self.flow = LatentFlow(
            z_dim=z_dim,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_dim=time_dim,
            dropout=dropout,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.sample_steps = sample_steps
        self.sample_solver = sample_solver
        self.ot_coupling = ot_coupling
        self.fixed_noise = fixed_noise
        self.time_weighting = time_weighting
        self.cosine_loss_weight = cosine_loss_weight
        self.norm_loss_weight = norm_loss_weight
        self.t_eps = t_eps
        self.sample_residual = sample_residual

        self.register_buffer("z_mean", z_mean)
        self.register_buffer("z_std", z_std)
        self.register_buffer("pca_mean", pca_mean)
        self.register_buffer("pca_components", pca_components)
        self.register_buffer("pca_std", pca_std)
        self.register_buffer("pca_residual_components", pca_residual_components)
        self.register_buffer("pca_residual_std", pca_residual_std)
        self.register_buffer("noise_codes", noise_codes, persistent=False)

    @property
    def uses_pca(self) -> bool:
        return (
            self.pca_mean is not None
            and self.pca_components is not None
            and self.pca_std is not None
        )

    def encode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        z = (codes - self.z_mean) / self.z_std
        if self.uses_pca:
            z = (z - self.pca_mean) @ self.pca_components.T
            z = z / self.pca_std
        return z

    def has_residual_pca(self) -> bool:
        return (
            self.pca_residual_components is not None
            and self.pca_residual_std is not None
            and self.pca_residual_components.numel() > 0
        )

    def decode_to_base_norm(self, z: torch.Tensor, sample_residual: bool | None = None) -> torch.Tensor:
        if self.uses_pca:
            z = (z * self.pca_std) @ self.pca_components
            if sample_residual is None:
                sample_residual = self.sample_residual
            if sample_residual and self.has_residual_pca():
                residual = torch.randn(
                    z.size(0),
                    self.pca_residual_std.size(1),
                    device=z.device,
                    dtype=z.dtype,
                )
                residual = residual * self.pca_residual_std
                z = z + residual @ self.pca_residual_components
            z = z + self.pca_mean
        return z

    def decode_to_code(self, z: torch.Tensor, sample_residual: bool | None = None) -> torch.Tensor:
        z_base = self.decode_to_base_norm(z, sample_residual=sample_residual)
        return z_base * self.z_std + self.z_mean

    @staticmethod
    def _match_with_ot(z_0: torch.Tensor, z_1: torch.Tensor) -> torch.Tensor:
        cost = torch.cdist(z_0.detach().float(), z_1.detach().float()).pow(2).cpu()
        if linear_sum_assignment is not None:
            _, col = linear_sum_assignment(cost.numpy())
            return torch.as_tensor(col, device=z_1.device, dtype=torch.long)
        return greedy_linear_sum_assignment(cost).to(z_1.device)

    def _time_weights(self, t: torch.Tensor) -> torch.Tensor:
        if self.time_weighting == "none":
            return torch.ones_like(t)
        if self.time_weighting == "middle":
            return 1.0 / (0.25 + (t - 0.5).abs())
        if self.time_weighting == "endpoints":
            return 1.0 / (t * (1.0 - t)).clamp_min(0.05)
        raise ValueError(f"Unknown time weighting: {self.time_weighting}")

    def _shared_step(self, batch: torch.Tensor, mode: str):
        z_1 = self.encode_codes(batch.code)
        
        if self.fixed_noise and self.noise_codes is not None and hasattr(batch, "data_idx"):
            data_idx = batch.data_idx.view(-1).to(device=z_1.device, dtype=torch.long)
            z_0 = self.noise_codes[data_idx].to(dtype=z_1.dtype)
        else:
            z_0 = torch.randn_like(z_1)

        if self.ot_coupling and not self.fixed_noise and z_1.size(0) > 1:
            perm = self._match_with_ot(z_0, z_1)
            z_1 = z_1[perm]

        t = torch.rand(z_1.size(0), 1, device=z_1.device, dtype=z_1.dtype)
        if self.t_eps > 0.0:
            t = t.mul(1.0 - 2.0 * self.t_eps).add(self.t_eps)

        z_t = (1.0 - t)*z_0 + t*z_1
        velocity_target = z_1 - z_0

        velocity_pred = self.flow(z_t, t)

        mse_per_sample = (velocity_pred - velocity_target).pow(2).mean(dim=-1)
        mse_loss = mse_per_sample.mean()

        weights = self._time_weights(t.squeeze(-1))
        weighted_mse = (mse_per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

        pred_norm_per_sample = velocity_pred.norm(dim=-1)
        target_norm_per_sample = velocity_target.norm(dim=-1)
        norm_loss = (
            (pred_norm_per_sample - target_norm_per_sample)
            / target_norm_per_sample.clamp_min(1e-6)
        ).pow(2).mean()

        cos_per_sample = F.cosine_similarity(velocity_pred, velocity_target, dim=-1)
        cos = cos_per_sample.mean()
        cosine_loss = 1.0 - cos

        loss = (
            weighted_mse
            + self.cosine_loss_weight * cosine_loss
            + self.norm_loss_weight * norm_loss
        )
        # loss = F.mse_loss(velocity_pred, velocity_target)
        with torch.no_grad():
            pred_norm = pred_norm_per_sample.mean()
            target_norm = target_norm_per_sample.mean()
            norm_ratio = pred_norm / target_norm.clamp_min(1e-6)

        self.log(f"{mode}/velocity_cos", cos, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/pred_velocity_norm", pred_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/target_velocity_norm", target_norm, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/velocity_norm_ratio", norm_ratio, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/loss", mse_loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/weighted_mse", weighted_mse, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/norm_loss", norm_loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        self.log(f"{mode}/optim_loss", loss, on_step=False, on_epoch=True, sync_dist=True, batch_size=z_1.size(0))
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="val")

    @torch.no_grad()
    def sample(
        self,
        n_samples: int,
        steps: int | None = None,
        unnormalize: bool = True,
        solver: str | None = None,
        sample_residual: bool | None = None,
    ):
        steps = int(steps or self.sample_steps)
        solver = solver or self.sample_solver
        device = self.device
        z = torch.randn(n_samples, self.hparams.z_dim, device=device)
        dt = 1.0 / steps

        for step in range(steps):
            t0 = torch.full((n_samples, 1), step / steps, device=device, dtype=z.dtype)
            if solver == "euler":
                velocity = self.flow(z, t0)
                z = z + dt * velocity
            elif solver == "heun":
                t1 = torch.full((n_samples, 1), (step + 1) / steps, device=device, dtype=z.dtype)
                v0 = self.flow(z, t0)
                z_euler = z + dt * v0
                v1 = self.flow(z_euler, t1)
                z = z + 0.5 * dt * (v0 + v1)
            else:
                raise ValueError(f"Unknown sampler: {solver}")

        if unnormalize:
            z = self.decode_to_code(z, sample_residual=sample_residual)
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.95),
        )

        if self.hparams.scheduler == "none":
            return optimizer

        if self.hparams.scheduler == "onecycle":
            total_steps = int(self.trainer.estimated_stepping_batches)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.lr,
                total_steps=max(total_steps, 1),
                pct_start=self.hparams.onecycle_pct_start,
                anneal_strategy="cos",
                final_div_factor=self.hparams.onecycle_final_div_factor,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                },
            }

        if self.hparams.scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(int(self.trainer.max_epochs), 1),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                },
            }

        raise ValueError(f"Unknown scheduler: {self.hparams.scheduler}")
    
