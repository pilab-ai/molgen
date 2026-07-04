import math
import torch 
from torch import nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("time embedding dim must be even")
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]

        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=device, dtype=t.dtype)
            / max(half - 1, 1)
        )
        args = t * 1000.0 * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)

class ConditionEncoder(nn.Module):
    def __init__(
        self,
        unit_embedding_dim: int,
        cond_dim: int,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dim = hidden_dim or cond_dim * 2

        self.unit_encoder = nn.Sequential(
            nn.LayerNorm(unit_embedding_dim),
            nn.Linear(unit_embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.scalar_encoder = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cond_dim),
        )
        self.out = nn.Sequential(
            nn.Linear(cond_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, cond_dim),
        )

    def forward(
        self,
        unit_embedding: torch.Tensor,
        degree: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if unit_embedding.ndim > 2:
            unit_embedding = unit_embedding.view(unit_embedding.size(0), -1)
        if degree.ndim == 1:
            degree = degree[:, None]
        if temperature.ndim == 1:
            temperature = temperature[:, None]

        unit_cond = self.unit_encoder(unit_embedding.float())
        scalar_cond = self.scalar_encoder(
            torch.cat([degree.float(), temperature.float()], dim=-1)
        )
        return self.out(torch.cat([unit_cond, scalar_cond], dim=-1))
    
class ResidualMLPBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        time_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        # num_heads: int = 4
    ):
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio

        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(time_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:

        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        x = self.norm(h)
        x = x * (1.0 + scale) + shift
        x = self.fc2(self.dropout(F.silu(self.fc1(x))))
        return h + self.dropout(x)


class LatentFlow(nn.Module):
    def __init__(
        self,
        z_dim: int,
        unit_embedding_dim: int,
        hidden_dim: int = 1024,
        n_layers: int = 4,
        time_dim: int = 128,
        condition_hidden_dim: int | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.condition_embed = ConditionEncoder(
            unit_embedding_dim=unit_embedding_dim,
            cond_dim=time_dim,
            hidden_dim=condition_hidden_dim,
            dropout=dropout,
        )

        self.input_proj = nn.Linear(z_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            ResidualMLPBlock(
                hidden_dim=hidden_dim,
                time_dim=time_dim,
                dropout=dropout,
            )
            for _ in range(n_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, z_dim)

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        unit_embedding: torch.Tensor,
        degree: torch.Tensor,
        temperature: torch.Tensor,
    ) -> torch.Tensor:
        if t.ndim == 1:
            t = t[:, None]
        t_emb = self.time_embed(t)
        cond_emb = self.condition_embed(unit_embedding, degree, temperature)
        context = t_emb + cond_emb
        h = self.input_proj(z_t) + self.time_proj(context)
        for block in self.blocks:
            h = block(h, context)
        h = self.output_norm(h)
        return self.output_proj(h)

# class LatentFlow(nn.Module):
#     def __init__(
#         self,
#         z_dim: int,
#         hidden_dim: int = 1024,
#         n_layers: int = 4,
#         time_dim: int = 128,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.z_dim = z_dim
#         self.time_embed = nn.Sequential(
#             SinusoidalTimeEmbedding(time_dim),
#             nn.Linear(time_dim, time_dim),
#             nn.SiLU(),
#             nn.Linear(time_dim, time_dim),
#         )

#         self.input_proj = nn.Linear(z_dim, hidden_dim)
#         self.time_proj = nn.Linear(time_dim, hidden_dim)
#         self.blocks = nn.ModuleList(
#             ResidualMLPBlock(
#                 hidden_dim=hidden_dim,
#                 time_dim=time_dim,
#                 dropout=dropout,
#             )
#             for _ in range(n_layers)
#         )
#         self.output_norm = nn.LayerNorm(hidden_dim)
#         self.output_proj = nn.Linear(hidden_dim, z_dim)

#     def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         if t.ndim == 1:
#             t = t[:, None]
#         t_emb = self.time_embed(t)
#         h = self.input_proj(z_t) + self.time_proj(t_emb)
#         for block in self.blocks:
#             h = block(h, t_emb)
#         h = self.output_norm(h)
#         return self.output_proj(h)
    

# class LatentFlow(nn.Module):
#     def __init__(
#         self,
#         z_dim: int,
#         hidden_dim: int = 1024,
#         n_layers: int = 4,
#         time_dim: int = 128,
#         dropout: float = 0.0,
#     ):
#         super().__init__()
#         self.z_dim = z_dim
#         self.time_embed = nn.Sequential(
#             SinusoidalTimeEmbedding(time_dim),
#             nn.Linear(time_dim, time_dim),
#             nn.SiLU(),
#             nn.Linear(time_dim, time_dim),
#         )

#         layers = []
#         in_dim = z_dim + time_dim
#         for _ in range(n_layers):
#             layers.extend(
#                 [
#                     nn.Linear(in_dim, hidden_dim),
#                     nn.LayerNorm(hidden_dim),
#                     nn.SiLU(),
#                     nn.Dropout(dropout),
#                 ]
#             )
#             in_dim = hidden_dim
#         layers.append(nn.Linear(hidden_dim, z_dim))
#         self.net = nn.Sequential(*layers)

#     def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
#         if t.ndim == 1:
#             t = t[:, None]
#         t_emb = self.time_embed(t)
#         return self.net(torch.cat([z_t, t_emb], dim=-1))
    

    