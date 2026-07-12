#!/usr/bin/env python3
"""End-to-end pipeline test for molgen on the synthetic QM9 dataset.

Reimplements minimal PyG/Lightning utilities so the entire pipeline runs
with only PyTorch installed.  Produces an HTML report at the end.

Pipeline:
  1. Stage 1 — Auto-Decoder training (per-molecule latent codes)
  2. Export latent codes
  3. Stage 2 — Latent Flow Matching (unconditional, QM9 has no conditions)
  4. Sampling & reconstruction
  5. HTML report generation
"""

import os
import sys
import json
import math
import time
import html as html_mod
from datetime import datetime

import torch
from torch import nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Minimal PyG-like utilities
# ---------------------------------------------------------------------------

class Data:
    """Minimal stand-in for torch_geometric.data.Data."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def keys(self):
        return [k for k in self.__dict__ if not k.startswith("_")]

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __getitem__(self, key):
        return getattr(self, key)

    def __repr__(self):
        info = ", ".join(f"{k}={list(v.shape)}" for k, v in self.__dict__.items() if isinstance(v, torch.Tensor))
        return f"Data({info})"


def load_dataset(root):
    """Load InMemoryDataset from processed .pt file."""
    path = os.path.join(root, "processed/data_v3.pt")
    data_dict, slices = torch.load(path, weights_only=False)
    n_graphs = slices["x"].shape[0] - 1

    data_list = []
    for i in range(n_graphs):
        d = Data()
        for key in data_dict:
            if key == "edge_index":
                start, end = slices[key][i].item(), slices[key][i + 1].item()
                d[key] = data_dict[key][:, start:end]
            else:
                start, end = slices[key][i].item(), slices[key][i + 1].item()
                d[key] = data_dict[key][start:end]
        d.data_idx = i
        d.num_nodes = d.x.size(0) if hasattr(d, "x") else d.z.size(0)
        data_list.append(d)
    return data_list


def to_dense_batch(x, batch, max_num_nodes):
    """Convert sparse batch to dense [B, N, F] with node mask."""
    # Handle 1D input (scalar features per node)
    squeeze_output = False
    if x.dim() == 1:
        x = x.unsqueeze(-1)
        squeeze_output = True

    if batch is None:
        if squeeze_output:
            return x.unsqueeze(0).squeeze(-1), torch.ones(1, x.size(0), dtype=torch.bool)
        return x.unsqueeze(0), torch.ones(1, x.size(0), dtype=torch.bool)

    batch_size = batch.max().item() + 1
    feat_dim = x.size(1)

    dense = torch.zeros(batch_size, max_num_nodes, feat_dim, dtype=x.dtype, device=x.device)
    mask = torch.zeros(batch_size, max_num_nodes, dtype=torch.bool, device=x.device)

    for b in range(batch_size):
        idx = (batch == b).nonzero(as_tuple=False).view(-1)
        n = min(idx.size(0), max_num_nodes)
        dense[b, :n] = x[idx[:n]]
        mask[b, :n] = True

    if squeeze_output:
        dense = dense.squeeze(-1)

    return dense, mask


def scatter_mean(src, index, dim=0, dim_size=None):
    """Simple scatter mean."""
    if dim_size is None:
        dim_size = index.max().item() + 1
    out = torch.zeros(dim_size, *src.shape[1:], dtype=src.dtype, device=src.device)
    count = torch.zeros(dim_size, dtype=src.dtype, device=src.device)
    index_expand = index.unsqueeze(-1).expand_as(src) if src.dim() > 1 else index
    out.scatter_add_(dim, index_expand, src)
    count.scatter_add_(0, index, torch.ones(index.shape[0], dtype=src.dtype, device=src.device))
    count = count.unsqueeze(-1) if src.dim() > 1 else count
    return out / count.clamp(min=1)


def collate_data_list(data_list):
    """Collate a list of Data objects into a batch."""
    batch = torch.cat([
        torch.full((d.num_nodes,), i, dtype=torch.long)
        for i, d in enumerate(data_list)
    ])

    keys = list(dict.fromkeys(k for d in data_list for k in d.keys))
    batch_dict = {}
    for key in keys:
        vals = [getattr(d, key, None) for d in data_list]
        if any(v is None for v in vals):
            continue
        v0 = vals[0]
        if isinstance(v0, torch.Tensor):
            if key == "edge_index":
                # Offset edge indices
                offset = 0
                shifted = []
                for d in data_list:
                    shifted.append(getattr(d, key) + offset)
                    offset += d.num_nodes
                batch_dict[key] = torch.cat(shifted, dim=1)
            elif v0.dim() == 0:
                batch_dict[key] = torch.stack(vals)
            elif v0.dim() == 1:
                batch_dict[key] = torch.cat(vals, dim=0)
            else:
                batch_dict[key] = torch.cat(vals, dim=0)
        elif isinstance(v0, int):
            batch_dict[key] = torch.tensor(vals, dtype=torch.long)
        elif isinstance(v0, float):
            batch_dict[key] = torch.tensor(vals, dtype=torch.float32)
        else:
            batch_dict[key] = vals

    batch_dict["batch"] = batch
    batch_dict["num_graphs"] = len(data_list)
    for i, d in enumerate(data_list):
        if hasattr(d, "data_idx"):
            if "data_idx" not in batch_dict:
                batch_dict["data_idx"] = torch.zeros(len(data_list), dtype=torch.long)
            batch_dict["data_idx"][i] = d.data_idx

    return batch_dict


class SimpleDataLoader:
    """Minimal DataLoader for Data lists."""
    def __init__(self, data_list, batch_size=32, shuffle=True):
        self.data_list = data_list
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        indices = list(range(len(self.data_list)))
        if self.shuffle:
            import random
            random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            batch_indices = indices[i:i + self.batch_size]
            yield collate_data_list([self.data_list[j] for j in batch_indices])

    def __len__(self):
        return (len(self.data_list) + self.batch_size - 1) // self.batch_size


# ---------------------------------------------------------------------------
# Models (copied from src/model/)
# ---------------------------------------------------------------------------

class SlotFiLMBlock(nn.Module):
    def __init__(self, hidden_dim, z_dim, mlp_ratio=4, dropout=0.0):
        super().__init__()
        inner_dim = hidden_dim * mlp_ratio
        self.norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(z_dim, hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim, inner_dim)
        self.fc2 = nn.Linear(inner_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, h, z):
        scale, shift = self.film(z).chunk(2, dim=-1)
        x = self.norm(h)
        x = x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        x = self.fc2(F.gelu(self.fc1(x)))
        return h + self.dropout(x)


class TMolDecoder(nn.Module):
    def __init__(self, z_dim=256, max_atoms=29, n_atom_types=5, hidden_dim=256, n_layers=3, dropout=0.0):
        super().__init__()
        self.max_atoms = max_atoms
        self.n_atom_types = n_atom_types
        self.slot = nn.Parameter(torch.randn(max_atoms, hidden_dim) * 0.02)
        self.z_proj = nn.Linear(z_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            SlotFiLMBlock(hidden_dim=hidden_dim, z_dim=z_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.exist_head = nn.Linear(hidden_dim, 1)
        self.atom_head = nn.Linear(hidden_dim, n_atom_types)
        self.coord_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 3),
        )

    def forward(self, z):
        B = z.size(0)
        h = self.slot.unsqueeze(0).expand(B, -1, -1)
        h = h + self.z_proj(z).unsqueeze(1)
        for block in self.blocks:
            h = block(h, z)
        h = self.final_norm(h)
        exist_logits = self.exist_head(h).squeeze(-1)
        atom_logits = self.atom_head(h)
        coords = self.coord_head(h)
        return {
            "exist_logits": exist_logits,
            "atom_logits": atom_logits,
            "coords": coords,
        }


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        if t.ndim == 1:
            t = t[:, None]
        half = self.dim // 2
        device = t.device
        freq = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device, dtype=t.dtype) / max(half - 1, 1)
        )
        args = t * 1000.0 * freq[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualMLPBlock(nn.Module):
    def __init__(self, hidden_dim, time_dim, mlp_ratio=4, dropout=0.0):
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

    def forward(self, h, t_emb):
        scale, shift = self.film(t_emb).chunk(2, dim=-1)
        x = self.norm(h)
        x = x * (1.0 + scale) + shift
        x = self.fc2(self.dropout(F.silu(self.fc1(x))))
        return h + self.dropout(x)


class LatentFlowUnconditional(nn.Module):
    """Unconditional flow matching (for QM9, no PI-specific conditions)."""
    def __init__(self, z_dim=256, hidden_dim=256, n_layers=3, time_dim=64, dropout=0.0):
        super().__init__()
        self.z_dim = z_dim
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.input_proj = nn.Linear(z_dim, hidden_dim)
        self.time_proj = nn.Linear(time_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            ResidualMLPBlock(hidden_dim=hidden_dim, time_dim=time_dim, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, z_dim)

    def forward(self, z_t, t):
        if t.ndim == 1:
            t = t[:, None]
        t_emb = self.time_embed(t)
        h = self.input_proj(z_t) + self.time_proj(t_emb)
        for block in self.blocks:
            h = block(h, t_emb)
        h = self.output_norm(h)
        return self.output_proj(h)


# ---------------------------------------------------------------------------
# Training loops
# ---------------------------------------------------------------------------

def train_autodecoder(dataset, cfg, device="cpu"):
    """Stage 1: Auto-Decoder training."""
    print("\n" + "=" * 60)
    print("STAGE 1: Auto-Decoder Training")
    print("=" * 60)

    # Split train/val
    n = len(dataset)
    val_ratio = cfg.get("val_ratio", 0.1)
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val

    torch.manual_seed(cfg.get("seed", 42))
    perm = torch.randperm(n)
    train_indices = perm[:n_train].tolist()
    val_indices = perm[n_train:].tolist()

    train_data = [dataset[i] for i in train_indices]
    val_data = [dataset[i] for i in val_indices]

    # Determine n_atom_types from x features (QM9: one-hot 5 + 6 other = 11 features)
    # We'll treat x[:, :5] as atom-type one-hot, argmax gives class 0-4
    n_atom_types = 5  # QM9: H, C, N, O, F

    scale = cfg["max_size"]
    max_atoms = cfg.get("max_atoms", 29)
    code_dim = cfg.get("code_dim", 256)
    hidden_dim = cfg.get("hidden_dim", 256)
    n_layers = cfg.get("n_layers", 3)
    lr_dec = cfg.get("lr_dec", 5e-4)
    lr_code = cfg.get("lr_code", 1e-3)
    batch_size = cfg.get("batch_size", 16)
    n_epochs = cfg.get("max_epochs", 20)  # small for test

    # Model
    codes = nn.Embedding(n, code_dim)
    nn.init.normal_(codes.weight, mean=0.0, std=0.01)
    decoder = TMolDecoder(
        z_dim=code_dim,
        max_atoms=max_atoms,
        n_atom_types=n_atom_types,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
    ).to(device)
    codes = codes.to(device)

    optimizer = torch.optim.AdamW([
        {"params": decoder.parameters(), "lr": lr_dec},
        {"params": codes.parameters(), "lr": lr_code},
    ])

    train_loader = SimpleDataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = SimpleDataLoader(val_data, batch_size=batch_size, shuffle=False)

    history = {
        "train_loss": [], "val_loss": [],
        "train_exist": [], "train_atom": [], "train_coord": [],
        "val_exist": [], "val_atom": [], "val_coord": [],
    }

    best_val_loss = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(n_epochs):
        # --- Train ---
        decoder.train()
        epoch_losses = {"total": 0, "exist": 0, "atom": 0, "coord": 0}
        n_batches = 0

        for batch in train_loader:
            x = batch["x"].to(device)
            pos = batch["pos"].to(device)
            batch_idx = batch["batch"].to(device)
            data_idx = batch["data_idx"].to(device)
            num_graphs = batch["num_graphs"]

            # Center positions
            center = scatter_mean(pos, batch_idx, dim=0, dim_size=num_graphs)
            pos_centered = pos - center[batch_idx]
            pos_norm = pos_centered / scale

            # Dense batch
            atom_type_dense, node_mask = to_dense_batch(
                x[:, :5].argmax(dim=-1), batch_idx, max_num_nodes=max_atoms
            )
            coords_dense, _ = to_dense_batch(pos_norm, batch_idx, max_num_nodes=max_atoms)

            # Forward
            z = codes(data_idx)
            out = decoder(z)

            loss_exist = F.binary_cross_entropy_with_logits(
                out["exist_logits"], node_mask.float()
            )
            loss_atom = F.cross_entropy(
                out["atom_logits"][node_mask], atom_type_dense[node_mask]
            )
            loss_coord = F.smooth_l1_loss(
                out["coords"][node_mask] * scale,
                coords_dense[node_mask] * scale,
            )
            loss = loss_exist + loss_atom + loss_coord

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_losses["total"] += loss.item()
            epoch_losses["exist"] += loss_exist.item()
            epoch_losses["atom"] += loss_atom.item()
            epoch_losses["coord"] += loss_coord.item()
            n_batches += 1

        train_loss = epoch_losses["total"] / max(n_batches, 1)
        train_exist = epoch_losses["exist"] / max(n_batches, 1)
        train_atom = epoch_losses["atom"] / max(n_batches, 1)
        train_coord = epoch_losses["coord"] / max(n_batches, 1)

        # --- Validate ---
        decoder.eval()
        val_losses = {"total": 0, "exist": 0, "atom": 0, "coord": 0}
        val_batches = 0

        with torch.no_grad():
            for batch in val_loader:
                x = batch["x"].to(device)
                pos = batch["pos"].to(device)
                batch_idx = batch["batch"].to(device)
                data_idx = batch["data_idx"].to(device)
                num_graphs = batch["num_graphs"]

                center = scatter_mean(pos, batch_idx, dim=0, dim_size=num_graphs)
                pos_centered = pos - center[batch_idx]
                pos_norm = pos_centered / scale

                atom_type_dense, node_mask = to_dense_batch(
                    x[:, :5].argmax(dim=-1), batch_idx, max_num_nodes=max_atoms
                )
                coords_dense, _ = to_dense_batch(pos_norm, batch_idx, max_num_nodes=max_atoms)

                z = codes(data_idx)
                out = decoder(z)

                loss_exist = F.binary_cross_entropy_with_logits(
                    out["exist_logits"], node_mask.float()
                )
                loss_atom = F.cross_entropy(
                    out["atom_logits"][node_mask], atom_type_dense[node_mask]
                )
                loss_coord = F.smooth_l1_loss(
                    out["coords"][node_mask] * scale,
                    coords_dense[node_mask] * scale,
                )
                loss = loss_exist + loss_atom + loss_coord

                val_losses["total"] += loss.item()
                val_losses["exist"] += loss_exist.item()
                val_losses["atom"] += loss_atom.item()
                val_losses["coord"] += loss_coord.item()
                val_batches += 1

        val_loss = val_losses["total"] / max(val_batches, 1)
        val_exist = val_losses["exist"] / max(val_batches, 1)
        val_atom = val_losses["atom"] / max(val_batches, 1)
        val_coord = val_losses["coord"] / max(val_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_exist"].append(train_exist)
        history["train_atom"].append(train_atom)
        history["train_coord"].append(train_coord)
        history["val_exist"].append(val_exist)
        history["val_atom"].append(val_atom)
        history["val_coord"].append(val_coord)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                "decoder": {k: v.clone() for k, v in decoder.state_dict().items()},
                "codes": {k: v.clone() for k, v in codes.state_dict().items()},
            }

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"exist={train_exist:.4f}  atom={train_atom:.4f}  coord={train_coord:.4f}")

    elapsed = time.time() - start_time
    print(f"  Stage 1 complete in {elapsed:.1f}s  best_val_loss={best_val_loss:.4f}")

    # Restore best
    decoder.load_state_dict(best_state["decoder"])
    codes.load_state_dict(best_state["codes"])

    return decoder, codes, history, elapsed


def export_latents(codes, n_molecules, device="cpu"):
    """Export latent codes and compute statistics."""
    print("\n" + "=" * 60)
    print("EXPORT: Latent Codes")
    print("=" * 60)

    all_codes = codes.weight.detach()  # [N, code_dim]
    z_mean = all_codes.mean(dim=0, keepdim=True)
    z_std = all_codes.std(dim=0, keepdim=True).clamp_min(1e-6)

    print(f"  Code shape: {all_codes.shape}")
    print(f"  Code mean: {all_codes.mean().item():.6f}")
    print(f"  Code std:  {all_codes.std().item():.6f}")
    print(f"  Code min:  {all_codes.min().item():.6f}")
    print(f"  Code max:  {all_codes.max().item():.6f}")

    return all_codes, z_mean, z_std


def train_flow(codes, z_mean, z_std, cfg, device="cpu"):
    """Stage 2: Latent Flow Matching training (unconditional for QM9)."""
    print("\n" + "=" * 60)
    print("STAGE 2: Latent Flow Matching Training")
    print("=" * 60)

    code_dim = codes.size(1)
    hidden_dim = cfg.get("flow_hidden_dim", 256)
    n_layers = cfg.get("flow_n_layers", 3)
    time_dim = cfg.get("flow_time_dim", 64)
    lr = cfg.get("flow_lr", 1e-4)
    batch_size = cfg.get("flow_batch_size", 32)
    n_epochs = cfg.get("flow_max_epochs", 20)
    sample_steps = cfg.get("flow_sample_steps", 50)

    # Normalize codes
    codes_norm = (codes - z_mean) / z_std
    n_codes = codes_norm.size(0)

    flow = LatentFlowUnconditional(
        z_dim=code_dim,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        time_dim=time_dim,
    ).to(device)

    optimizer = torch.optim.AdamW(flow.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = {
        "train_loss": [], "val_loss": [],
        "train_cos": [], "val_cos": [],
        "train_pred_norm": [], "train_target_norm": [],
    }

    # Split codes into train/val
    n_val = max(1, int(n_codes * 0.1))
    n_train = n_codes - n_val
    torch.manual_seed(42)
    perm = torch.randperm(n_codes)
    train_codes = codes_norm[perm[:n_train]]
    val_codes = codes_norm[perm[n_train:]]

    best_val_loss = float("inf")
    best_state = None
    start_time = time.time()

    for epoch in range(n_epochs):
        # --- Train ---
        flow.train()
        epoch_loss = 0
        epoch_cos = 0
        epoch_pred_norm = 0
        epoch_target_norm = 0
        n_batches = 0

        perm_idx = torch.randperm(n_train)
        for i in range(0, n_train, batch_size):
            idx = perm_idx[i:i + batch_size]
            z_1 = train_codes[idx].to(device)
            bs = z_1.size(0)

            z_0 = torch.randn_like(z_1)
            t_randn = torch.randn(bs, 1, device=device)
            t = torch.sigmoid(t_randn)  # logit-normal

            z_t = (1.0 - t) * z_0 + t * z_1
            velocity_target = z_1 - z_0

            velocity_pred = flow(z_t, t)

            loss = F.mse_loss(velocity_pred, velocity_target)
            cos = F.cosine_similarity(velocity_pred, velocity_target, dim=-1).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_cos += cos.item()
            with torch.no_grad():
                epoch_pred_norm += velocity_pred.norm(dim=-1).mean().item()
                epoch_target_norm += velocity_target.norm(dim=-1).mean().item()
            n_batches += 1

        train_loss = epoch_loss / max(n_batches, 1)
        train_cos = epoch_cos / max(n_batches, 1)
        train_pred_norm = epoch_pred_norm / max(n_batches, 1)
        train_target_norm = epoch_target_norm / max(n_batches, 1)

        # --- Validate ---
        flow.eval()
        val_loss_sum = 0
        val_cos_sum = 0
        val_batches = 0

        with torch.no_grad():
            for i in range(0, n_val, batch_size):
                z_1 = val_codes[i:i + batch_size].to(device)
                bs = z_1.size(0)
                z_0 = torch.randn_like(z_1)
                t_randn = torch.randn(bs, 1, device=device)
                t = torch.sigmoid(t_randn)
                z_t = (1.0 - t) * z_0 + t * z_1
                velocity_target = z_1 - z_0
                velocity_pred = flow(z_t, t)
                loss = F.mse_loss(velocity_pred, velocity_target)
                cos = F.cosine_similarity(velocity_pred, velocity_target, dim=-1).mean()
                val_loss_sum += loss.item()
                val_cos_sum += cos.item()
                val_batches += 1

        val_loss = val_loss_sum / max(val_batches, 1)
        val_cos = val_cos_sum / max(val_batches, 1)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_cos"].append(train_cos)
        history["val_cos"].append(val_cos)
        history["train_pred_norm"].append(train_pred_norm)
        history["train_target_norm"].append(train_target_norm)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in flow.state_dict().items()}

        scheduler.step()

        if (epoch + 1) % max(1, n_epochs // 10) == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1:3d}/{n_epochs}  "
                  f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
                  f"cos_sim={train_cos:.4f}")

    elapsed = time.time() - start_time
    print(f"  Stage 2 complete in {elapsed:.1f}s  best_val_loss={best_val_loss:.4f}")

    flow.load_state_dict(best_state)
    return flow, history, elapsed


def sample_and_reconstruct(flow, decoder, codes, z_mean, z_std, cfg, device="cpu"):
    """Sample from flow and reconstruct molecules."""
    print("\n" + "=" * 60)
    print("SAMPLING & RECONSTRUCTION")
    print("=" * 60)

    code_dim = codes.size(1)
    n_samples = cfg.get("n_flow_samples", 10)
    sample_steps = cfg.get("flow_sample_steps", 50)

    # 1. Flow sampling (Euler ODE)
    z = torch.randn(n_samples, code_dim, device=device)
    dt = 1.0 / sample_steps
    for step in range(sample_steps):
        t = torch.full((n_samples, 1), step / sample_steps, device=device)
        velocity = flow(z, t)
        z = z + dt * velocity

    # Unnormalize
    z_sampled = z * z_std.to(device) + z_mean.to(device)

    # 2. Decoder reconstruction
    with torch.no_grad():
        out = decoder(z_sampled)

    # Collect results
    results = []
    for i in range(n_samples):
        exist_prob = torch.sigmoid(out["exist_logits"][i])
        active = exist_prob > 0.5
        n_active = active.sum().item()
        atom_types = out["atom_logits"][i][active].argmax(dim=-1)
        coords = out["coords"][i][active]

        results.append({
            "n_atoms": n_active,
            "atom_types": atom_types.tolist(),
            "coords_mean": coords.mean().item() if n_active > 0 else 0,
            "coords_std": coords.std().item() if n_active > 1 else 0,
        })

    # 3. Also reconstruct from original codes (ground-truth comparison)
    gt_results = []
    n_gt = min(n_samples, codes.size(0))
    with torch.no_grad():
        gt_out = decoder(codes[:n_gt].to(device))

    for i in range(n_gt):
        exist_prob = torch.sigmoid(gt_out["exist_logits"][i])
        active = exist_prob > 0.5
        n_active = active.sum().item()
        gt_results.append({"n_atoms": n_active})

    print(f"  Sampled {n_samples} molecules from flow")
    for i, r in enumerate(results[:5]):
        print(f"    Sample {i}: {r['n_atoms']} atoms, atom_types={r['atom_types'][:8]}")
    print(f"  Ground-truth reconstruction ({n_gt} molecules)")
    for i, r in enumerate(gt_results[:5]):
        print(f"    GT {i}: {r['n_atoms']} atoms predicted")

    return results, gt_results


# ---------------------------------------------------------------------------
# HTML Report Generation
# ---------------------------------------------------------------------------

def generate_html_report(stage1_hist, stage2_hist, sample_results, gt_results,
                         codes_stats, cfg, stage1_time, stage2_time, output_path):
    """Generate a comprehensive HTML report."""

    # SVG chart generation (inline, no external deps)
    def line_chart(data_series, labels, title, width=700, height=300):
        """Generate an SVG line chart."""
        all_vals = [v for series in data_series for v in series]
        if not all_vals:
            return "<p>No data</p>"

        y_min = min(all_vals)
        y_max = max(all_vals)
        if y_max == y_min:
            y_max = y_min + 1

        margin = {"top": 30, "right": 20, "bottom": 40, "left": 70}
        plot_w = width - margin["left"] - margin["right"]
        plot_h = height - margin["top"] - margin["bottom"]

        colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]

        paths = []
        for si, series in enumerate(data_series):
            n = len(series)
            if n == 0:
                continue
            points = []
            for i, v in enumerate(series):
                x = margin["left"] + (i / max(n - 1, 1)) * plot_w
                y = margin["top"] + plot_h - ((v - y_min) / (y_max - y_min)) * plot_h
                points.append(f"{x:.1f},{y:.1f}")
            color = colors[si % len(colors)]
            paths.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2"/>')

        # Y-axis ticks
        n_ticks = 5
        y_ticks = []
        for i in range(n_ticks + 1):
            val = y_min + (y_max - y_min) * i / n_ticks
            y = margin["top"] + plot_h - (i / n_ticks) * plot_h
            y_ticks.append(f'<line x1="{margin["left"]}" y1="{y:.1f}" x2="{margin["left"] + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="0.5"/>')
            y_ticks.append(f'<text x="{margin["left"] - 5}" y="{y:.1f}" text-anchor="end" font-size="10" fill="#6b7280">{val:.4f}</text>')

        # X-axis label
        x_label_y = margin["top"] + plot_h + 25

        svg = f'''<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg" style="background:#fff;border-radius:8px">
  <text x="{width // 2}" y="18" text-anchor="middle" font-size="13" font-weight="bold" fill="#1f2937">{html_mod.escape(title)}</text>
  {"".join(y_ticks)}
  <line x1="{margin["left"]}" y1="{margin["top"]}" x2="{margin["left"]}" y2="{margin["top"] + plot_h}" stroke="#9ca3af" stroke-width="1"/>
  <line x1="{margin["left"]}" y1="{margin["top"] + plot_h}" x2="{margin["left"] + plot_w}" y2="{margin["top"] + plot_h}" stroke="#9ca3af" stroke-width="1"/>
  <text x="{width // 2}" y="{x_label_y}" text-anchor="middle" font-size="11" fill="#6b7280">Epoch</text>
  {"".join(paths)}
</svg>'''
        return svg

    def legend(labels, colors_list=None):
        """Generate an SVG legend."""
        if colors_list is None:
            colors_list = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
        items = []
        for i, label in enumerate(labels):
            c = colors_list[i % len(colors_list)]
            items.append(f'<span style="display:inline-block;margin-right:16px"><span style="display:inline-block;width:14px;height:14px;background:{c};border-radius:2px;vertical-align:middle;margin-right:4px"></span>{html_mod.escape(label)}</span>')
        return '<div style="margin:8px 0;font-size:12px;color:#374151">' + "".join(items) + "</div>"

    # Build charts
    chart1 = line_chart(
        [stage1_hist["train_loss"], stage1_hist["val_loss"]],
        ["Train Loss", "Val Loss"],
        "Stage 1: Auto-Decoder Total Loss"
    )
    chart1_legend = legend(["Train Loss", "Val Loss"])

    chart1_detail = line_chart(
        [stage1_hist["train_exist"], stage1_hist["train_atom"], stage1_hist["train_coord"]],
        ["Exist BCE", "Atom CE", "Coord L1"],
        "Stage 1: Training Loss Breakdown"
    )
    chart1_detail_legend = legend(["Exist BCE", "Atom CE", "Coord L1"])

    chart2 = line_chart(
        [stage2_hist["train_loss"], stage2_hist["val_loss"]],
        ["Train Loss", "Val Loss"],
        "Stage 2: Flow Matching MSE Loss"
    )
    chart2_legend = legend(["Train Loss", "Val Loss"])

    chart2_cos = line_chart(
        [stage2_hist["train_cos"], stage2_hist["val_cos"]],
        ["Train Cosine Sim", "Val Cosine Sim"],
        "Stage 2: Velocity Cosine Similarity"
    )
    chart2_cos_legend = legend(["Train Cosine Sim", "Val Cosine Sim"])

    chart2_norms = line_chart(
        [stage2_hist["train_pred_norm"], stage2_hist["train_target_norm"]],
        ["Predicted Velocity Norm", "Target Velocity Norm"],
        "Stage 2: Velocity Norms"
    )
    chart2_norms_legend = legend(["Predicted Norm", "Target Norm"])

    # Sample table
    sample_rows = ""
    for i, r in enumerate(sample_results):
        atom_str = ", ".join(str(a) for a in r["atom_types"][:12])
        if len(r["atom_types"]) > 12:
            atom_str += "..."
        sample_rows += f"""<tr>
            <td>{i}</td>
            <td>{r['n_atoms']}</td>
            <td>{atom_str}</td>
            <td>{r['coords_mean']:.4f}</td>
            <td>{r['coords_std']:.4f}</td>
        </tr>"""

    # GT table
    gt_rows = ""
    for i, r in enumerate(gt_results):
        gt_rows += f"""<tr>
            <td>{i}</td>
            <td>{r['n_atoms']}</td>
        </tr>"""

    # Summary stats
    final_train_loss_s1 = stage1_hist["train_loss"][-1] if stage1_hist["train_loss"] else 0
    final_val_loss_s1 = stage1_hist["val_loss"][-1] if stage1_hist["val_loss"] else 0
    best_val_loss_s1 = min(stage1_hist["val_loss"]) if stage1_hist["val_loss"] else 0
    final_train_loss_s2 = stage2_hist["train_loss"][-1] if stage2_hist["train_loss"] else 0
    final_val_loss_s2 = stage2_hist["val_loss"][-1] if stage2_hist["val_loss"] else 0
    best_val_loss_s2 = min(stage2_hist["val_loss"]) if stage2_hist["val_loss"] else 0
    final_cos_sim = stage2_hist["train_cos"][-1] if stage2_hist["train_cos"] else 0

    n_atoms_list = [r["n_atoms"] for r in sample_results]
    avg_n_atoms = sum(n_atoms_list) / len(n_atoms_list) if n_atoms_list else 0

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>molgen Pipeline Test Report — QM9</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #f8fafc; color: #1e293b; padding: 2rem; }}
  .container {{ max-width: 960px; margin: 0 auto; }}
  h1 {{ font-size: 1.75rem; font-weight: 700; margin-bottom: 0.25rem; color: #0f172a; }}
  h2 {{ font-size: 1.35rem; font-weight: 600; margin: 2rem 0 0.75rem; color: #1e40af; border-bottom: 2px solid #dbeafe; padding-bottom: 0.35rem; }}
  h3 {{ font-size: 1.1rem; font-weight: 600; margin: 1.25rem 0 0.5rem; color: #334155; }}
  .subtitle {{ color: #64748b; font-size: 0.95rem; margin-bottom: 1.5rem; }}
  .card {{ background: #fff; border-radius: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); padding: 1.5rem; margin-bottom: 1.25rem; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 1rem; margin-bottom: 1.5rem; }}
  .stat {{ background: #fff; border-radius: 10px; padding: 1rem 1.25rem; box-shadow: 0 1px 3px rgba(0,0,0,0.06); }}
  .stat .label {{ font-size: 0.78rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.03em; }}
  .stat .value {{ font-size: 1.45rem; font-weight: 700; color: #0f172a; margin-top: 0.15rem; }}
  .stat .value.good {{ color: #16a34a; }}
  .stat .value.warn {{ color: #ea580c; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #f1f5f9; padding: 0.6rem 0.75rem; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid #e2e8f0; }}
  td {{ padding: 0.5rem 0.75rem; border-bottom: 1px solid #f1f5f9; }}
  tr:hover {{ background: #f8fafc; }}
  .chart {{ text-align: center; margin: 1rem 0; }}
  .tag {{ display: inline-block; padding: 0.15rem 0.5rem; border-radius: 9999px; font-size: 0.72rem; font-weight: 600; }}
  .tag-green {{ background: #dcfce7; color: #166534; }}
  .tag-blue {{ background: #dbeafe; color: #1e40af; }}
  .tag-orange {{ background: #ffedd5; color: #9a3412; }}
  code {{ background: #f1f5f9; padding: 0.15rem 0.35rem; border-radius: 4px; font-size: 0.85em; }}
  .config-table td:first-child {{ font-weight: 600; color: #475569; white-space: nowrap; }}
</style>
</head>
<body>
<div class="container">

<h1>🧪 molgen Pipeline Test Report</h1>
<p class="subtitle">QM9 synthetic dataset &middot; {now} &middot; device={cfg.get('device', 'cpu')}</p>

<!-- ============================================================ -->
<h2>📊 Summary</h2>
<div class="grid">
  <div class="stat">
    <div class="label">Dataset</div>
    <div class="value" style="font-size:1.1rem">QM9 (synthetic)</div>
  </div>
  <div class="stat">
    <div class="label">Molecules</div>
    <div class="value">{codes_stats['n_molecules']}</div>
  </div>
  <div class="stat">
    <div class="label">Stage 1 Time</div>
    <div class="value">{stage1_time:.1f}s</div>
  </div>
  <div class="stat">
    <div class="label">Stage 2 Time</div>
    <div class="value">{stage2_time:.1f}s</div>
  </div>
  <div class="stat">
    <div class="label">Best Val Loss (S1)</div>
    <div class="value {'good' if best_val_loss_s1 < 1 else ''}">{best_val_loss_s1:.4f}</div>
  </div>
  <div class="stat">
    <div class="label">Best Val Loss (S2)</div>
    <div class="value {'good' if best_val_loss_s2 < 0.5 else ''}">{best_val_loss_s2:.4f}</div>
  </div>
  <div class="stat">
    <div class="label">Flow Cosine Sim</div>
    <div class="value {'good' if final_cos_sim > 0.5 else 'warn'}">{final_cos_sim:.4f}</div>
  </div>
  <div class="stat">
    <div class="label">Avg Sampled Atoms</div>
    <div class="value">{avg_n_atoms:.1f}</div>
  </div>
</div>

<!-- ============================================================ -->
<h2>⚙️ Configuration</h2>
<div class="card">
<table class="config-table">
  <tr><td>Dataset</td><td>QM9 synthetic (100 molecules)</td></tr>
  <tr><td>Max atoms (padded)</td><td>{cfg.get('max_atoms', 29)}</td></tr>
  <tr><td>Atom types</td><td>{cfg.get('n_atom_types', 5)} (H, C, N, O, F)</td></tr>
  <tr><td>Position scale</td><td>{cfg.get('max_size', 8)}</td></tr>
  <tr><td>Code dimension</td><td>{cfg.get('code_dim', 256)}</td></tr>
  <tr><td>Decoder hidden dim</td><td>{cfg.get('hidden_dim', 256)}</td></tr>
  <tr><td>Decoder layers</td><td>{cfg.get('n_layers', 3)}</td></tr>
  <tr><td>Stage 1 epochs</td><td>{cfg.get('max_epochs', 20)}</td></tr>
  <tr><td>Stage 1 lr (decoder)</td><td>{cfg.get('lr_dec', 5e-4)}</td></tr>
  <tr><td>Stage 1 lr (codes)</td><td>{cfg.get('lr_code', 1e-3)}</td></tr>
  <tr><td>Flow hidden dim</td><td>{cfg.get('flow_hidden_dim', 256)}</td></tr>
  <tr><td>Flow layers</td><td>{cfg.get('flow_n_layers', 3)}</td></tr>
  <tr><td>Stage 2 epochs</td><td>{cfg.get('flow_max_epochs', 20)}</td></tr>
  <tr><td>Flow sample steps</td><td>{cfg.get('flow_sample_steps', 50)}</td></tr>
  <tr><td>Device</td><td>{cfg.get('device', 'cpu')}</td></tr>
</table>
</div>

<!-- ============================================================ -->
<h2>🔬 Stage 1: Auto-Decoder</h2>

<div class="card">
<h3>Training Curves</h3>
<div class="chart">{chart1}</div>
{chart1_legend}
</div>

<div class="card">
<h3>Loss Breakdown (Train)</h3>
<div class="chart">{chart1_detail}</div>
{chart1_detail_legend}
</div>

<div class="card">
<h3>Latent Code Statistics</h3>
<table>
  <tr><th>Metric</th><th>Value</th></tr>
  <tr><td>Mean</td><td>{codes_stats['mean']:.6f}</td></tr>
  <tr><td>Std</td><td>{codes_stats['std']:.6f}</td></tr>
  <tr><td>Min</td><td>{codes_stats['min']:.6f}</td></tr>
  <tr><td>Max</td><td>{codes_stats['max']:.6f}</td></tr>
  <tr><td>Norm (mean per-code)</td><td>{codes_stats['norm_mean']:.6f}</td></tr>
</table>
</div>

<!-- ============================================================ -->
<h2>🌊 Stage 2: Latent Flow Matching</h2>

<div class="card">
<h3>MSE Loss</h3>
<div class="chart">{chart2}</div>
{chart2_legend}
</div>

<div class="card">
<h3>Velocity Cosine Similarity</h3>
<div class="chart">{chart2_cos}</div>
{chart2_cos_legend}
</div>

<div class="card">
<h3>Velocity Norms (Train)</h3>
<div class="chart">{chart2_norms}</div>
{chart2_norms_legend}
</div>

<!-- ============================================================ -->
<h2>🧬 Sampling & Reconstruction</h2>

<div class="card">
<h3>Flow-Sampled Molecules</h3>
<p style="font-size:0.85rem;color:#64748b;margin-bottom:0.75rem">
  Molecules sampled by: noise → Euler ODE solver ({cfg.get('flow_sample_steps', 50)} steps) → unnormalize → decoder → atom existence/type/coords.
  Atom type indices: 0=H, 1=C, 2=N, 3=O, 4=F.
</p>
<table>
  <tr><th>#</th><th>Predicted Atoms</th><th>Atom Types</th><th>Coord Mean</th><th>Coord Std</th></tr>
  {sample_rows}
</table>
</div>

<div class="card">
<h3>Ground-Truth Code Reconstruction</h3>
<p style="font-size:0.85rem;color:#64748b;margin-bottom:0.75rem">
  Decoder applied to the learned latent codes from Stage 1 (no flow sampling).
</p>
<table>
  <tr><th>#</th><th>Predicted Atoms</th></tr>
  {gt_rows}
</table>
</div>

<!-- ============================================================ -->
<h2>📝 Notes</h2>
<div class="card" style="font-size:0.88rem;color:#475569;line-height:1.65">
  <ul style="padding-left:1.25rem">
    <li>This report uses <strong>synthetic QM9 data</strong> (100 random graphs) generated by <code>create_test_datasets.py</code>.</li>
    <li>Stage 1 auto-decoder learns per-molecule latent codes via <code>nn.Embedding</code> + slot-FiLM transformer.</li>
    <li>Stage 2 flow matching trains an <strong>unconditional</strong> velocity field (QM9 has no PI-specific conditions like degree/temperature).</li>
    <li>Training uses reduced epochs and model sizes for testing purposes — production configs use 1000+ epochs and larger hidden dims.</li>
    <li>All runs on <strong>{cfg.get('device', 'cpu')}</strong> — GPU training would be significantly faster.</li>
  </ul>
</div>

<div style="text-align:center;color:#94a3b8;font-size:0.78rem;margin-top:2rem;padding-top:1rem;border-top:1px solid #e2e8f0">
  Generated by <code>test_pipeline_qm9.py</code> &middot; molgen pipeline test &middot; {now}
</div>

</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"\n  HTML report saved → {output_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA GPU")
    else:
        print("Using CPU")

    # Load QM9 dataset
    print("Loading QM9 synthetic dataset...")
    dataset = load_dataset("dataset/qm9/data")
    print(f"  Loaded {len(dataset)} molecules")

    # Inspect a sample
    sample = dataset[0]
    print(f"  Sample: x={sample.x.shape}, z={sample.z.shape}, pos={sample.pos.shape}")

    # Configuration
    cfg = {
        # Data
        "max_size": 8,          # scale factor from qm9.yaml
        "max_atoms": 29,        # max nodes per graph in QM9 (padded to this)
        "n_atom_types": 5,      # H, C, N, O, F
        "val_ratio": 0.1,
        "seed": 42,

        # Stage 1 — Auto-Decoder
        "code_dim": 256,
        "hidden_dim": 256,
        "n_layers": 3,
        "lr_dec": 5e-4,
        "lr_code": 1e-3,
        "batch_size": 16,
        "max_epochs": 30,       # small for testing

        # Stage 2 — Flow Matching
        "flow_hidden_dim": 256,
        "flow_n_layers": 3,
        "flow_time_dim": 64,
        "flow_lr": 1e-4,
        "flow_batch_size": 32,
        "flow_max_epochs": 30,
        "flow_sample_steps": 50,

        # Sampling
        "n_flow_samples": 10,

        # Device
        "device": device,
    }

    # ---- Stage 1: Auto-Decoder ----
    decoder, codes, stage1_hist, stage1_time = train_autodecoder(dataset, cfg, device)

    # ---- Export Latents ----
    all_codes, z_mean, z_std = export_latents(codes, len(dataset), device)
    codes_stats = {
        "n_molecules": len(dataset),
        "mean": all_codes.mean().item(),
        "std": all_codes.std().item(),
        "min": all_codes.min().item(),
        "max": all_codes.max().item(),
        "norm_mean": all_codes.norm(dim=-1).mean().item(),
    }

    # ---- Stage 2: Flow Matching ----
    flow, stage2_hist, stage2_time = train_flow(all_codes, z_mean, z_std, cfg, device)

    # ---- Sampling & Reconstruction ----
    sample_results, gt_results = sample_and_reconstruct(
        flow, decoder, all_codes, z_mean, z_std, cfg, device
    )

    # ---- Generate HTML Report ----
    output_path = os.path.join(os.path.dirname(__file__), "pipeline_test_report_qm9.html")
    generate_html_report(
        stage1_hist, stage2_hist, sample_results, gt_results,
        codes_stats, cfg, stage1_time, stage2_time, output_path
    )

    print("\n" + "=" * 60)
    print("✅ Pipeline test complete!")
    print(f"   Report: {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
