import random
from typing import Optional

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, TensorBoardLogger
from torch.utils.data import Dataset, Subset
from torch_geometric.loader import DataLoader

from src.trainers.autodecoder_trainer import TMolAutoDecoderTrainer

def make_val_subset(dataset, val_ratio: Optional[float], seed: int):
    val_size = int(val_ratio*len(dataset))
    if val_size is None or val_size <= 0 or val_size >= len(dataset):
        return dataset
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    return Subset(dataset, indices[:val_size])

def run(dataset, cfg):
    seed=cfg["trainer"]["seed"]
    L.seed_everything(seed=seed, workers=True)

    val_dataset = make_val_subset(dataset, cfg["trainer"]["val_ratio"], cfg["trainer"]["seed"])
    train_loader = DataLoader(
        dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=True,
        num_workers=cfg["trainer"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["trainer"]["num_workers"] > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["trainer"]["batch_size"],
        shuffle=False,
        num_workers=cfg["trainer"]["num_workers"],
        pin_memory=True,
        persistent_workers=cfg["trainer"]["num_workers"] > 0,
    )

    model = TMolAutoDecoderTrainer(
        n_codes=len(dataset),
        n_atom_types=len(cfg["data"]["elements"]),
        scale=cfg["data"]["max_size"],
        code_dim=cfg["model"]["code_dim"],
        max_atoms=cfg["model"]["max_atoms"], 
        hidden_dim=cfg["model"]["hidden_dim"],
        n_layers=cfg["model"]["n_layers"],
        lr_dec=cfg["optim"]["lr_dec"],
        lr_code=cfg["optim"]["lr_code"],
        weight_decay=cfg["optim"]["weight_decay"],
        code_reg_weight=cfg["optim"]["code_reg_weight"]
    )

    tb_logger = TensorBoardLogger(
        save_dir=cfg["trainer"]["log_dir"],
        name="autodecoder_model",
    )
    csv_logger = CSVLogger(
        save_dir=cfg["trainer"]["log_dir"],
        name="autodecoder_model_csv",
    )
    checkpoint = ModelCheckpoint(
        monitor="val/loss/total_loss",
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
        # fast_dev_run=1,
    )
    trainer.fit(
        model, 
        train_loader, 
        val_loader, 
        ckpt_path="logs/pi/autodecoder_model/version_27/checkpoints/epoch=1197-step=519932.ckpt"
        )

    



