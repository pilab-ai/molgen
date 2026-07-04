import torch
from torch import nn
import torch.nn.functional as F

class SlotFiLMBlock(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        z_dim: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(z_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        scale, shift = self.film(z).chunk(2, dim=-1)
        x = self.norm(h)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return h + self.dropout(x)


class TMolDecoder(nn.Module):
    def __init__(
        self,
        z_dim: int = 512,
        max_atoms: int = 10000,
        n_atom_types: int = 5,
        n_bond_types: int = 4,
        hidden_dim: int = 512,
        n_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.max_atoms = max_atoms
        self.n_atom_types = n_atom_types
        self.n_bond_types = n_bond_types

        # 原子槽位的初始 embedding。乘以0.02 是小随机数初始化，避免一开始特征幅度过大
        self.slot = nn.Parameter(torch.randn(max_atoms, hidden_dim) * 0.02)
        # 全局条件的 embedding。这个全局条件会加到每个原子槽位上，使每个 slot 都知道当前要生成的是哪个分子
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.blocks = nn.ModuleList(
            SlotFiLMBlock(hidden_dim=hidden_dim, z_dim=z_dim, dropout=dropout)
            for _ in range(n_layers)
        )
        self.final_norm = nn.LayerNorm(hidden_dim)

        # 原子是否存在的预测头 [B, max_atoms, 1]
        self.exist_head = nn.Linear(hidden_dim, 1)
        # 原子类型预测头 [B, max_atoms, n_atom_types]
        self.atom_head = nn.Linear(hidden_dim, n_atom_types)
        # 坐标预测头 [B, max_atoms, 3]
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

        # self.edge_head = nn.Sequential(
        #     nn.Linear(hidden_dim * 2 + 1, hidden_dim),
        #     nn.GELU(),
        #     nn.Linear(hidden_dim, n_bond_types + 1),  # 0 = no bond
        # )

    def forward(self, z, edge_candidates=None):
        # z: [B, z_dim]
        B = z.size(0)

        # 每个 batch 样本都共享同一套初始 slot embedding
        # [max_atoms, hidden_dim] -> [1, max_atoms, hidden_dim] -> [B, max_atoms, hidden_dim]
        h = self.slot.unsqueeze(0).expand(B, -1, -1)
        # 每个分子的全局 latent vector 被加到了该分子的所有原子槽位上
        h = h + self.z_proj(z).unsqueeze(1)
        for block in self.blocks:
            h = block(h, z)
        h = self.final_norm(h)

        exist_logits = self.exist_head(h).squeeze(-1)      # [B, N]
        atom_logits = self.atom_head(h)                    # [B, N, atom_types]
        coords = self.coord_head(h)                        # [B, N, 3]

        out = {
            "node_h": h,
            "exist_logits": exist_logits,
            "atom_logits": atom_logits,
            "coords": coords,
        }

        # if edge_candidates is not None:
        #     # edge_candidates: [M, 3], columns are batch_idx, src, dst
        #     b, i, j = edge_candidates.t()
        #     hi = h[b, i]
        #     hj = h[b, j]
        #     dist = torch.norm(coords[b, i] - coords[b, j], dim=-1, keepdim=True)
        #     edge_feat = torch.cat([hi, hj, dist], dim=-1)
        #     out["bond_logits"] = self.edge_head(edge_feat)  # [M, n_bond_types + 1]

        return out

    @torch.no_grad()
    def generate(self, z, exist_threshold=0.5, bond_threshold=0.5, cutoff=0.08):
        out = self.forward(z)
        exist_prob = torch.sigmoid(out["exist_logits"])
        atom_type = out["atom_logits"].argmax(dim=-1)
        coords = out["coords"]

        results = []
        for b in range(z.size(0)):
            active = exist_prob[b] > exist_threshold
            node_idx = active.nonzero(as_tuple=False).view(-1)

            pos = coords[b, node_idx]
            atom = atom_type[b, node_idx]

            if pos.size(0) <= 1:
                results.append((atom, pos, torch.empty(2, 0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            dist = torch.cdist(pos, pos)
            cand = (dist < cutoff) & (dist > 0)
            src, dst = cand.nonzero(as_tuple=True)

            edge_candidates = torch.stack([
                torch.zeros_like(src),
                node_idx[src],
                node_idx[dst],
            ], dim=-1).to(z.device)

            bond_logits = self.forward(z[b:b + 1], edge_candidates)["bond_logits"]
            bond_prob = F.softmax(bond_logits, dim=-1)
            bond_type = bond_prob.argmax(dim=-1)
            keep = (bond_type > 0) & (bond_prob[:, 1:].max(dim=-1).values > bond_threshold)

            edge_index = torch.stack([src[keep], dst[keep]], dim=0)
            edge_type = bond_type[keep] - 1

            results.append((atom, pos, edge_index, edge_type))

        return results



