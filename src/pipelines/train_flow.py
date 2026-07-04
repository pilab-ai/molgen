import torch
from torch_geometric.loader import DataLoader
from torch.utils.data import random_split
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger

from src.utils.create_data_pi_latent import PILatent
from src.trainers.flow_trainer import LatentFlowMatchingTrainer

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

    train_dataset, val_dataset = split_dataset(
        dataset,
        val_ratio=cfg["trainer"]["val_ratio"],
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
    condition_stats = dataset.get_condition_stats()
    unit_embedding_dim = dataset.get_unit_embedding_dim()

    model = LatentFlowMatchingTrainer(
        z_dim=cfg["model"]["code_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["n_layers"],
        time_dim=cfg["model"]["time_dim"],
        dropout=cfg["model"]["dropout"],
        lr=cfg["optim"]["lr"],
        weight_decay=cfg["optim"]["weight_decay"],
        sample_steps=cfg["sample"]["steps"],
        z_mean=z_mean,
        z_std=z_std,
        degree_mean=condition_stats["degree_mean"],
        degree_std=condition_stats["degree_std"],
        temperature_mean=condition_stats["temperature_mean"],
        temperature_std=condition_stats["temperature_std"],
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

    trainer = L.Trainer(
        accelerator=cfg["trainer"]["accelerator"],
        devices=cfg["trainer"]["devices"],
        strategy=cfg["trainer"]["strategy"],
        precision=cfg["trainer"]["precision"],
        max_epochs=cfg["trainer"]["max_epochs"],
        logger=[tb_logger, csv_logger],
        callbacks=[checkpoint],
        log_every_n_steps=cfg["trainer"]["log_every_n_steps"],
        check_val_every_n_epoch=cfg["trainer"]["check_val_every_n_epoch"],
        fast_dev_run=1,
    )
    trainer.fit(model, train_loader, val_loader)


