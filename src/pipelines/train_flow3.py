import json
import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import lightning as L
from pathlib import Path
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.utils.create_data_pi_latent import PILatent
from src.trainers.flow_trainer3 import LatentFlowMatchingTrainer


def fit_pca_whitening(
    codes: torch.Tensor,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    pca_dim: int,
    eps: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    dict[str, float],
]:
    codes = codes.detach().float().cpu()
    z_mean = z_mean.detach().float().cpu()
    z_std = z_std.detach().float().cpu()

    z = (codes - z_mean) / z_std
    pca_mean = z.mean(dim=0, keepdim=True)
    z = z - pca_mean

    cov = z.T @ z
    cov = cov / max(z.size(0) - 1, 1)
    eigvals, eigvecs = torch.linalg.eigh(cov)
    order = eigvals.argsort(descending=True)
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]

    pca_dim = min(pca_dim, eigvecs.size(1))
    kept_vals = eigvals[:pca_dim].clamp_min(eps)
    components = eigvecs[:, :pca_dim].T.contiguous()
    pca_std = kept_vals.sqrt().view(1, pca_dim)
    residual_vals = eigvals[pca_dim:].clamp_min(eps)
    residual_components = eigvecs[:, pca_dim:].T.contiguous()
    residual_std = residual_vals.sqrt().view(1, -1)

    total_var = eigvals.clamp_min(0).sum().clamp_min(eps)
    explained = kept_vals.sum() / total_var
    info = {
        "pca_dim": float(pca_dim),
        "explained_variance": float(explained),
        "top_eigenvalue": float(eigvals[0]),
        "kept_min_eigenvalue": float(kept_vals[-1]),
        "residual_dim": float(residual_components.size(0)),
        "residual_variance": float(residual_vals.sum() / total_var),
    }
    return pca_mean, components, pca_std, residual_components, residual_std, info


def tensor_stats(x: torch.Tensor) -> dict[str, float]:
    x = x.detach().float().cpu()
    norms = x.norm(dim=-1)
    return {
        "global_mean": float(x.mean()),
        "global_std": float(x.std()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_min": float(norms.min()),
        "norm_max": float(norms.max()),
    }


def nearest_neighbor_stats(query: torch.Tensor, bank: torch.Tensor) -> dict[str, float]:
    query = query.detach().float().cpu()
    bank = bank.detach().float().cpu()
    dist = torch.cdist(query, bank).min(dim=1).values
    return {
        "mean": float(dist.mean()),
        "std": float(dist.std()),
        "p10": float(dist.quantile(0.10)),
        "p50": float(dist.quantile(0.50)),
        "p90": float(dist.quantile(0.90)),
    }


def evaluate_sample_distribution(
    samples_base_norm: torch.Tensor,
    dataset: PILatent,
    z_mean: torch.Tensor,
    z_std: torch.Tensor,
    seed: int,
    bank_size: int = 8192,
    query_size: int = 512,
) -> dict[str, dict[str, float]]:
    codes = dataset.data.code.detach().float().cpu()
    real_norm = (codes - z_mean.detach().float().cpu()) / z_std.detach().float().cpu()

    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(real_norm.size(0), generator=generator)
    bank_size = min(bank_size, max(real_norm.size(0) - 1, 1))
    query_size = min(query_size, samples_base_norm.size(0), real_norm.size(0) - bank_size)

    real_bank = real_norm[perm[:bank_size]].contiguous()
    real_query = real_norm[perm[bank_size:bank_size + query_size]].contiguous()
    sample_query = samples_base_norm.detach().float().cpu()[:query_size].contiguous()

    return {
        "generated_norm_stats": tensor_stats(samples_base_norm),
        "real_norm_stats": tensor_stats(real_bank),
        "generated_to_real_nn": nearest_neighbor_stats(sample_query, real_bank),
        "real_to_real_nn": nearest_neighbor_stats(real_query, real_bank),
    }


def split_dataset(dataset, val_ratio: float, seed: int):
    val_size = int(len(dataset) * val_ratio)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError("validation split leaves no training samples")
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)

def run(cfg):
    seed=cfg["trainer"]["seed"]
    L.seed_everything(seed=seed, workers=True)
    dataset = PILatent("dataset/pi/data")
    loss_cfg = cfg.get("loss", {})
    sample_cfg = cfg.get("sample", {})
    optim_cfg = cfg.get("optim", {})
    latent_transform_cfg = cfg.get("latent_transform", {"type": "none"})

    train_dataset, val_dataset = split_dataset(
        dataset,
        val_ratio=cfg["data"]["val_ratio"],
        seed=seed,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"]["num_workers"],
        pin_memory=cfg["trainer"]["pin_memory"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        num_workers=cfg["trainer"]["num_workers"],
        pin_memory=cfg["trainer"]["pin_memory"],
    )

    z_mean, z_std = dataset.get_z_stats()
    pca_mean = None
    pca_components = None
    pca_std = None
    pca_residual_components = None
    pca_residual_std = None
    pca_info = None
    noise_codes = None

    if latent_transform_cfg.get("type", "none") == "pca_whiten":
        pca_dim = int(latent_transform_cfg["dim"])
        if cfg["model"]["code_dim"] != pca_dim:
            raise ValueError(
                f"model.code_dim must equal latent_transform.dim when using PCA whitening "
                f"({cfg['model']['code_dim']} != {pca_dim})"
            )
        (
            pca_mean,
            pca_components,
            pca_std,
            pca_residual_components,
            pca_residual_std,
            pca_info,
        ) = fit_pca_whitening(
            codes=dataset.data.code,
            z_mean=z_mean,
            z_std=z_std,
            pca_dim=pca_dim,
            eps=latent_transform_cfg.get("eps", 1e-5),
        )
        print("PCA whitening:", pca_info)
    elif latent_transform_cfg.get("type", "none") != "none":
        raise ValueError(f"Unknown latent transform: {latent_transform_cfg['type']}")

    if loss_cfg.get("fixed_noise", False):
        generator = torch.Generator().manual_seed(seed + 1009)
        noise_codes = torch.randn(len(dataset), cfg["model"]["code_dim"], generator=generator)

    model = LatentFlowMatchingTrainer(
        z_dim=cfg["model"]["code_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["n_layers"],
        time_dim=cfg["model"]["time_dim"],
        dropout=cfg["model"]["dropout"],
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
        sample_steps=sample_cfg["steps"],
        sample_solver=sample_cfg.get("solver", "heun"),
        ot_coupling=loss_cfg.get("ot_coupling", False),
        fixed_noise=loss_cfg.get("fixed_noise", False),
        time_weighting=loss_cfg.get("time_weighting", "none"),
        cosine_loss_weight=loss_cfg.get("cosine_loss_weight", 0.0),
        norm_loss_weight=loss_cfg.get("norm_loss_weight", 0.0),
        t_eps=loss_cfg.get("t_eps", 0.0),
        scheduler=optim_cfg.get("scheduler", "none"),
        onecycle_pct_start=optim_cfg.get("onecycle_pct_start", 0.05),
        onecycle_final_div_factor=optim_cfg.get("onecycle_final_div_factor", 20.0),
        z_mean=z_mean,
        z_std=z_std,
        pca_mean=pca_mean,
        pca_components=pca_components,
        pca_std=pca_std,
        pca_residual_components=pca_residual_components,
        pca_residual_std=pca_residual_std,
        sample_residual=latent_transform_cfg.get("sample_residual", False),
        noise_codes=noise_codes,
    )


    tb_logger = TensorBoardLogger(
        save_dir=cfg["trainer"]["log_dir"], 
        name="flow_matching"
        )
    csv_logger = CSVLogger(
        save_dir=cfg["trainer"]["log_dir"], 
        name="flow_matching_csv"
        )
    checkpoint = ModelCheckpoint(
        monitor="val/loss",
        mode="min",
        save_top_k=3,
        save_last=True,
    )
    callbacks = [checkpoint]
    early_stop_cfg = cfg["trainer"].get("early_stopping", {})
    if early_stop_cfg.get("enabled", False):
        callbacks.append(
            EarlyStopping(
                monitor=early_stop_cfg.get("monitor", "val/loss"),
                mode=early_stop_cfg.get("mode", "min"),
                patience=early_stop_cfg.get("patience", 30),
                min_delta=early_stop_cfg.get("min_delta", 0.0),
                check_finite=True,
            )
        )

    trainer = L.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=cfg["trainer"]["strategy"],
        precision=cfg["trainer"]["precision"],
        max_epochs=cfg["trainer"]["max_epochs"],
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        check_val_every_n_epoch=cfg["trainer"]["check_val_every_n_epoch"],
        gradient_clip_val=cfg["trainer"].get("gradient_clip_val", 0.0),
        fast_dev_run=cfg["trainer"].get("fast_dev_run", False),
    )
    trainer.fit(model, train_loader, val_loader)

    if sample_cfg.get("save_after_fit", False) and trainer.is_global_zero:
        best_path = checkpoint.best_model_path
        sample_model = model
        if best_path:
            sample_model = LatentFlowMatchingTrainer.load_from_checkpoint(
                best_path,
                map_location=model.device,
                z_mean=z_mean,
                z_std=z_std,
                pca_mean=pca_mean,
                pca_components=pca_components,
                pca_std=pca_std,
                pca_residual_components=pca_residual_components,
                pca_residual_std=pca_residual_std,
                sample_residual=latent_transform_cfg.get("sample_residual", False),
                noise_codes=None,
            )
            sample_model.to(model.device)

        sample_model.eval()
        with torch.no_grad():
            samples_latent = sample_model.sample(
                n_samples=sample_cfg.get("n_samples", 16),
                steps=sample_cfg.get("steps", sample_model.sample_steps),
                solver=sample_cfg.get("solver", "heun"),
                unnormalize=False,
            )
            samples_base_norm = sample_model.decode_to_base_norm(
                samples_latent,
                sample_residual=latent_transform_cfg.get("sample_residual", False),
            )
            samples = samples_base_norm * sample_model.z_std + sample_model.z_mean

        output_path = Path(sample_cfg.get("output_path", "logs/flow/samples.pt"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "samples": samples.detach().cpu(),
                "samples_norm": samples_base_norm.detach().cpu(),
                "samples_latent": samples_latent.detach().cpu(),
                "best_model_path": best_path,
                "solver": sample_cfg.get("solver", "heun"),
                "steps": sample_cfg.get("steps", sample_model.sample_steps),
                "latent_transform": latent_transform_cfg,
                "pca_info": pca_info,
            },
            output_path,
        )

        if sample_cfg.get("evaluate_after_fit", True):
            summary = evaluate_sample_distribution(
                samples_base_norm=samples_base_norm,
                dataset=dataset,
                z_mean=z_mean,
                z_std=z_std,
                seed=seed,
                bank_size=sample_cfg.get("eval_bank_size", 8192),
                query_size=sample_cfg.get("eval_query_size", 512),
            )
            summary["best_model_path"] = best_path
            summary["solver"] = sample_cfg.get("solver", "heun")
            summary["steps"] = sample_cfg.get("steps", sample_model.sample_steps)
            summary["latent_transform"] = latent_transform_cfg
            summary["pca_info"] = pca_info

            summary_path = Path(sample_cfg.get("summary_path", "logs/flow/sample_summary.json"))
            summary_path.parent.mkdir(parents=True, exist_ok=True)
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
