# GenPnCFormer

> Generative design of phononic crystal (PnC) structures using a latent diffusion model conditioned on band-gap specifications.

GenPnCFormer combines a Conditional VAE with a Classifier-Free Guidance (CFG) Diffusion Transformer to generate PnC unit-cell geometries that satisfy user-specified bandgap targets. A pre-trained PnCFormer surrogate evaluates generated structures in place of expensive FEM simulations.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Data Preparation](#data-preparation)
6. [Training](#training)
7. [Evaluation](#evaluation)
8. [Inverse Design](#inverse-design)
9. [Configuration](#configuration)
10. [Citation](#citation)

---

## Overview

Given a **band mask** (desired pass / bandgap / defect pattern across 500 frequency bins at 0–50 kHz), GenPnCFormer generates phononic crystal layer thicknesses that realise that pattern. The pipeline has three stages:

| Stage | Module | Purpose |
|-------|--------|---------|
| **Stage 1** | Conditional VAE | Compresses PnC layer geometries into a low-dimensional latent space |
| **Stage 2** | Latent Diffusion (DiT) | Learns the conditional distribution `p(z | band_mask, material, n_cells)` |
| **Stage 3** | Surrogate / TMM | Evaluates generated structures to compute mBOF & DMA metrics |

## Architecture

```
Band Mask (500 bins)  ──► Transformer Encoder ──► Cross-Attention Context
                                                        │
Noise z_T ──► DiT Backbone (AdaLN-Zero) ──────────────►├──► z_0
                        ▲                               │
            (t, material, n_cells) ──► AdaLN            │
                                                        │
z_0 ──► VAE Decoder ──► Layer Thicknesses ──► PnCFormer / TMM ──► Predicted Band Mask
```

## Project Structure

```
GenPnCFormer/
├── config.py              # Central hyperparameter hub (CFG dataclass)
├── main.py                # Entry point: train → cache → evaluate
├── data_utils.py          # Data I/O, caching, feature engineering
├── vae.py                 # Conditional VAE (encoder + decoder + training)
├── diffusion.py           # Latent diffusion model (DiT / U-Net, DDPM, DDIM, CFG)
├── eval.py                # Inference, mBOF/DMA metrics, bulk visualization
├── benchmark.py           # Conditioning-mode ablation (AdaLN / AdaLN-Zero / MHCA)
├── inverse_design.py      # Target-driven inverse design pipeline
├── data_generation/       # Raw data generation scripts & TMM solver
│   └── tmm_torch.py       # GPU-accelerated Transfer Matrix Method
├── surrogate/             # PnCFormer surrogate model (frozen weights)
│   ├── models/
│   │   └── pncformer.py
│   └── best_model/        # Pre-trained checkpoints (UDR / TR)
├── data/                  # HDF5 raw data & NPZ caches (git-ignored)
├── checkpoints/           # Trained model weights (git-ignored)
└── .gitignore
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CUDA 12.x recommended)
- NumPy, SciPy, h5py, matplotlib, tqdm

### Environment Setup (using `uv`)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy scipy h5py matplotlib tqdm
```

## Data Preparation

Raw data is stored as HDF5 files under `data/{CA,SA,TA,AC,AS,AT}/`. Each file contains:
- `design_variable`: layer thicknesses (E, ρ, length)
- `dispersion_relation_unitcell` / `dispersion_relation_supercell`: UDR / SDR
- `frequencies`: frequency axis

On first run, `main.py` automatically builds NPZ caches (`data/cache_v2.3.1/`) with 80/10/10 train/valid/test splits.

## Training

```bash
# Full pipeline: VAE → latent cache → DDPM → visualization → evaluation
python main.py

# Use exact physics (TMM) instead of surrogate for evaluation
python main.py --use_tmm
```

The training pipeline:
1. **VAE** (Stage 1): Trains for 100 epochs, saves `vae_model_best.pt`
2. **Latent Cache**: Encodes all data through the trained VAE encoder
3. **DDPM** (Stage 2): Trains for 200 epochs with CFG, saves `ddpm_transformer_best.pt`
4. **Visualization**: Generates dispersion comparison plots
5. **Evaluation**: Computes mBOF and DMA metrics

### Resuming Training

If a valid checkpoint exists, the script will prompt whether to resume or skip training.

## Evaluation

```bash
# Run evaluation only (skip training)
python main.py --test_only

# With TMM solver
python main.py --test_only --use_tmm
```

### Metrics

| Metric | Description |
|--------|-------------|
| **mBOF** | Mean Bandgap Overlap Fraction — IoU between predicted and target bandgap intervals |
| **DMA** | Defect Mode Accuracy — fraction of defect peaks matched within δ=0.5 kHz tolerance |

Results are reported by material type (CA, SA, TA, AC, AS, AT) and by unit-cell count (4–7).

## Inverse Design

```python
from inverse_design import run_inverse_design

# Design a PnC with a specific bandgap target
run_inverse_design(cfg, device, target_band_mask, material_conds, n_cells)
```

The inverse design module generates multiple candidate structures and ranks them by mBOF score.

## Configuration

All hyperparameters are centralized in `config.py` via the `CFG` dataclass:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `latent_dim` | 128 | VAE latent space dimension |
| `d_model` | 128 | VAE Transformer width |
| `transformer_width` | 256 | DiT backbone width |
| `transformer_depth` | 8 | DiT backbone depth |
| `timesteps` | 1000 | DDPM diffusion steps |
| `epochs_vae` | 100 | VAE training epochs |
| `epochs_diffusion` | 200 | DDPM training epochs |
| `batch_size` | 128 | Training batch size |
| `cond_mode` | `adaln-zero` | Conditioning mode (adaln / adaln-zero / mhca) |

See [`config.py`](config.py) for the complete list.

## Citation

```bibtex
@article{genpcnformer2026,
  title={GenPnCFormer: Generative Design of Phononic Crystals via Latent Diffusion},
  author={...},
  year={2026}
}
```
