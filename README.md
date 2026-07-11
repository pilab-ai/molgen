# MolGen: 3D Molecule Generation via Auto-Decoder and Latent Flow Matching

MolGen is a two-stage framework for **conditional 3D molecule generation**. It learns a structured latent space of molecules using an auto-decoder, then trains a conditional flow-matching model in that latent space to generate new molecules given chemical and physical conditions.

## Overview

| Stage | Model | Input → Output |
|-------|-------|----------------|
| **1. Auto-Decoder** | Slot-FiLM Transformer (`TMolDecoder`) | Per-molecule latent code → atom types, 3D coordinates, existence mask |
| **2. Latent Flow** | FiLM-MLP (`LatentFlow`) | Noise + conditions → latent code (decoded by Stage 1) |

An alternative **Bayesian Flow Network (BFN)** model is also provided for direct 3D molecule generation without the two-stage pipeline.

## Architecture

### TMolDecoder (Auto-Decoder)

- **Slot embeddings**: Learnable `R^{max_atoms × hidden}` matrix, one slot per potential atom position.
- **FiLM conditioning**: A per-molecule latent vector `z` modulates each transformer layer via Feature-wise Linear Modulation (scale + shift).
- **Output heads**: Atom existence (binary), atom type (classification), 3D coordinates (regression).
- **Loss**: BCE (existence) + CE (atom type) + Smooth L1 (coordinates).

### LatentFlow (Conditional Flow Matching)

- **Backbone**: Stack of residual MLP blocks with FiLM time-conditioning.
- **Conditions**: Repeat-unit embedding (polyBERT), log(polymerization degree), temperature.
- **Training**: Predicts velocity field for optimal-transport flow matching in latent space.
- **Sampling**: Euler or Heun (2nd-order) ODE solver.
- **PCA whitening** (optional): Reduces and whitens latent codes before flow training for better geometry.

### MoleculeBFN (Bayesian Flow Network)

- **Joint continuous–discrete**: Gaussian BFN for coordinates + categorical BFN for atom types.
- **Backbone**: Standard transformer encoder with sinusoidal time embedding.
- **Schedule**: `β(t) = β₁t²` for coordinates, `α(t) = α₁t²` for atom types.
- **Sampling**: Iterative Bayesian updating over discrete time steps.

## Datasets

| Name | Description | Elements | Conditions |
|------|-------------|----------|------------|
| **PI** | Polyimide MD trajectories (LAMMPS) | H, C, N, O, F, S (6) | Repeat unit, degree, temperature |
| **QM9** | Small-molecule quantum chemistry benchmark | H, C, N, O, F (5) | — |
| **SDBS** | Organic spectral database | 72 element types | — |

### PI Dataset Pipeline

```
diamine.csv + dianhydride.csv
  → RDKit reaction SMARTS → polyimide repeat unit SMILES
  → polyBERT encoder → repeat-unit embeddings
  + LAMMPS MD trajectories (varying degree & temperature)
  → PI InMemoryDataset (graph + 3D coords + conditions)
```

## Project Structure

```
molgen/
├── main.py                          # Entry point (Stage 1 training)
├── run.sh                           # SLURM launch script
├── pyproject.toml                   # Dependencies (uv-managed)
├── config/
│   ├── pi.yaml                      # Polyimide auto-decoder config
│   ├── flow3.yaml                   # Flow matching config (with PCA whitening)
│   ├── sdbs.yaml                    # SDBS auto-decoder config
│   ├── qm9.yaml                     # QM9 config
│   └── config.yaml
└── src/
    ├── model/
    │   ├── decoder.py               # TMolDecoder (slot-FiLM auto-decoder)
    │   ├── flow.py                  # LatentFlow (conditional flow matching MLP)
    │   └── bfn.py                   # MoleculeBFN (Bayesian Flow Network)
    ├── pipelines/
    │   ├── train_autodecoder.py     # Stage 1 pipeline
    │   ├── export_dataset_with_latent.py  # Export latents after Stage 1
    │   ├── train_flow.py            # Stage 2 pipeline (basic)
    │   └── train_flow3.py           # Stage 2 pipeline (PCA whitening + OT coupling)
    ├── trainers/
    │   ├── autodecoder_trainer.py   # Lightning module for auto-decoder
    │   ├── flow_trainer.py          # Lightning module for flow (with conditions)
    │   └── flow_trainer3.py         # Lightning module for flow (PCA + OT + advanced losses)
    └── utils/
        ├── create_data_pi.py        # PI dataset from MD trajectories
        ├── create_data_pi_repeat_unit.py  # Polyimide repeat unit generation
        ├── create_data_pi_latent.py       # PI dataset with latent codes
        ├── create_data_qm9.py        # QM9 dataset
        └── create_data_sdbs.py       # SDBS dataset
```

## Usage

### Setup

```bash
# Install dependencies (requires uv)
uv sync
```

### Stage 1: Train Auto-Decoder

```bash
uv run python main.py
```

Configuration is in `config/pi.yaml`. Key parameters:

- `model.code_dim`: Latent code dimensionality (default: 1024)
- `model.hidden_dim`: Transformer hidden dimension (default: 1024)
- `trainer.max_epochs`: Training epochs (default: 2000)
- `trainer.devices`: Number of GPUs for DDP (default: 4)

### Export Latent Codes

After Stage 1 training, export the learned latent codes:

```python
from src.pipelines.export_dataset_with_latent import run as export_latents
export_latents(
    dataset=dataset,
    best_ckpt_path="logs/pi/autodecoder_model/version_XX/checkpoints/epoch=XXX-step=XXXXXX.ckpt",
    save_dir=cfg["data"]["dir"],
)
```

### Stage 2: Train Latent Flow

```bash
# With PCA whitening and advanced features
uv run python -c "
import yaml
from src.pipelines.train_flow3 import run
with open('config/flow3.yaml') as f:
    cfg = yaml.safe_load(f)
run(cfg)
"
```

### Generate Molecules

After Stage 2 training, sample new latent codes and decode them:

```python
# Load trained flow model and auto-decoder
# Sample z from noise via flow → decode with auto-decoder → 3D molecule
```

### SLURM Launch

```bash
sbatch run.sh
```

## Key Features

- **PCA whitening** of latent codes for better flow geometry
- **Optimal Transport (OT) coupling** for straighter flow paths
- **Heun solver** (2nd-order) for high-quality sampling
- **Early stopping** with configurable patience
- **DDP** multi-GPU training via PyTorch Lightning
- **Mixed precision** (bf16-mixed) for faster training

## Dependencies

- Python ≥ 3.13
- PyTorch ≥ 2.7 (CUDA 12.6)
- PyTorch Geometric ≥ 2.7
- Lightning ≥ 2.6
- RDKit ≥ 2025.9.3
- MDAnalysis ≥ 2.10
- sentence-transformers ≥ 5.5.1 (polyBERT)
- fairchem-core ≥ 2.19.0

## License

Not yet specified.
