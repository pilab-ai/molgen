import lightning as L
import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

from src.model.flow import LatentFlow

class LatentFlowMatchingTrainer(L.LightningModule):
    def __init__(
        self,
        z_dim: int,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        time_dim: int = 128,
        condition_hidden_dim: int | None = None,
        dropout: float = 0.0,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        sample_steps: int = 100,
        z_mean: torch.Tensor | None = None,
        z_std: torch.Tensor | None = None,
        degree_mean: torch.Tensor | None = None,
        degree_std: torch.Tensor | None = None,
        temperature_mean: torch.Tensor | None = None,
        temperature_std: torch.Tensor | None = None,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "z_mean",
                "z_std",
                "degree_mean",
                "degree_std",
                "temperature_mean",
                "temperature_std",
            ]
        )
        self.flow = LatentFlow(
            z_dim=z_dim,
            # unit_embedding_dim=,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            time_dim=time_dim,
            condition_hidden_dim=condition_hidden_dim,
            dropout=dropout,
        )
        self.lr = lr
        self.weight_decay = weight_decay
        self.sample_steps = sample_steps

        self.register_buffer("z_mean", z_mean)
        self.register_buffer("z_std", z_std)
        self.register_buffer("degree_mean", degree_mean)
        self.register_buffer("degree_std", degree_std)
        self.register_buffer("temperature_mean", temperature_mean)
        self.register_buffer("temperature_std", temperature_std)

    def _shared_step(self, batch: torch.Tensor, mode: str):
        z_1 = (batch.code - self.z_mean) / self.z_std
        z_0 = torch.randn_like(z_1)

        t_randn = torch.randn(z_1.size(0), 1, device=z_1.device, dtype=z_1.dtype)
        t = torch.sigmoid(t_randn) # 使用 Logit-Normal 分布
        
        z_t = (1.0 - t)*z_0 + t*z_1
        velocity_target = z_1 - z_0

        velocity_pred = self.flow(z_t, t)

        
        cos = F.cosine_similarity(velocity_pred, velocity_target, dim=-1).mean()
        
        loss = F.mse_loss(velocity_pred, velocity_target)
        # weight = 1.0 / (0.25 + (t.squeeze(-1) - 0.5).abs())
        # mse_loss = F.mse_loss(velocity_pred, velocity_target)
        # weight_mse_loss = (mse_loss * weight).mean()
        # loss = weight_mse_loss

        with torch.no_grad():
            pred_norm = velocity_pred.norm(dim=-1).mean()
            target_norm = velocity_target.norm(dim=-1).mean()

        
        self.log(
            f"{mode}/velocity_cos", 
            cos, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=z_1.size(0)
            )
        
        self.log(
            f"{mode}/pred_velocity_norm", 
            pred_norm, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=z_1.size(0)
            )
        
        self.log(
            f"{mode}/target_velocity_norm", 
            target_norm, 
            on_step=False, 
            on_epoch=True, 
            sync_dist=True, 
            batch_size=z_1.size(0)
            )
        
        self.log(
            f"{mode}/loss", 
            loss, 
            on_step=True, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True, 
            batch_size=z_1.size(0)
            )
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, mode="val")

    @torch.no_grad()
    def sample(
        self,
        unit_embedding: torch.Tensor,
        degree: torch.Tensor,
        temperature: torch.Tensor,
        n_samples: int | None = None,
        steps: int | None = None,
        unnormalize: bool = True,
    ):
        steps = int(steps or self.sample_steps)
        device = self.device

        # unit_embedding = self._as_matrix(unit_embedding.to(device=device)).float()
        unit_embedding = unit_embedding.to(device=device)
        n_samples = int(n_samples or unit_embedding.size(0))
        if unit_embedding.size(0) == 1 and n_samples > 1:
            unit_embedding = unit_embedding.expand(n_samples, -1)

        degree = degree.to(device=device)
        temperature = temperature.to(device=device)
        if degree.numel() == 1 and n_samples > 1:
            degree = degree.expand(n_samples)
        if temperature.numel() == 1 and n_samples > 1:
            temperature = temperature.expand(n_samples)

        unit_embedding, degree, temperature = self._normalize_conditions(
            unit_embedding,
            degree,
            temperature,
        )

        z = torch.randn(n_samples, self.hparams.z_dim, device=device)
        dt = 1.0 / steps

        for step in range(steps):
            t = torch.full((n_samples, 1), step / steps, device=device, dtype=z.dtype)
            velocity = self.flow(z, t, unit_embedding, degree, temperature)
            z = z + dt * velocity

        if unnormalize:
            z = z * self.z_std + self.z_mean
        return z

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
            },
        }
    

