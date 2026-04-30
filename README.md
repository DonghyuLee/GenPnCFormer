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
│
├── data_generation/       # Raw dataset generation (TMM-based simulation)
│   ├── __init__.py        # Public API exports
│   ├── data_types.py      # Material, Case, SimulationConfig dataclasses
│   ├── cases.py           # Generates all (cells, defect) case combinations
│   ├── core.py            # GPU-batched TMM simulation per case → HDF5 output
│   ├── tmm_torch.py       # PyTorch Transfer Matrix Method solver
│   ├── tmm_build.py       # TMM build utilities
│   ├── io_utils.py        # HDF5 file naming & writing
│   ├── run.py             # CLI entry point for data generation
│   └── visualize.py       # Dispersion / transmittance visualization
│
├── surrogate/             # PnCFormer surrogate model (frozen weights)
│   ├── models/
│   │   └── pncformer.py
│   └── best_model/        # Pre-trained checkpoints (UDR / TR)
│
├── data/                  # HDF5 raw data & NPZ caches (git-ignored)
├── checkpoints/           # Trained model weights (git-ignored)
└── .gitignore
```

## Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1 (CUDA 12.x recommended)
- NumPy, SciPy, h5py, matplotlib, tqdm, joblib

### Environment Setup (using `uv`)

```bash
# Create and activate virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
uv pip install numpy scipy h5py matplotlib tqdm joblib
```

## Data Preparation

### Physical Background

A **phononic crystal (PnC)** is a periodic stack of alternating layers of two materials (A and B). Each *unit cell* consists of one A-layer and one B-layer. A **defect** is introduced by replacing a B-layer's thickness with a different value, which creates a localized mode inside the bandgap.

The Transfer Matrix Method (TMM) computes three physical quantities for a given geometry:

| Quantity | Symbol | Description |
|----------|--------|-------------|
| **Unit-cell Dispersion Relation** | UDR | Bloch wave dispersion of a single AB cell |
| **Supercell Dispersion Relation** | SDR | Dispersion of the full N-cell structure (reveals bandgap & defect modes) |
| **Transmittance** | TR | Power transmission coefficient across the structure |

### Step 1 — Generate Raw HDF5 Data

The `data_generation/` package uses a GPU-accelerated TMM solver to simulate all structural configurations.

```bash
# Generate all 6 material pairs × 28 structure cases × 20,000 samples
python -m data_generation.run
```

This produces HDF5 files under `data/{MATERIAL_PREFIX}/`:

#### Material Pairs

| Prefix | Material A | Material B | E_A (GPa) | ρ_A (kg/m³) | E_B (GPa) | ρ_B (kg/m³) |
|--------|-----------|-----------|-----------|-------------|-----------|-------------|
| `CA` | Copper | Aluminum | 110 | 8960 | 70 | 2700 |
| `SA` | Steel | Aluminum | 200 | 7850 | 70 | 2700 |
| `TA` | Titanium | Aluminum | 116 | 4500 | 70 | 2700 |
| `AC` | Aluminum | Copper | 70 | 2700 | 110 | 8960 |
| `AS` | Aluminum | Steel | 70 | 2700 | 200 | 7850 |
| `AT` | Aluminum | Titanium | 70 | 2700 | 116 | 4500 |

#### Structure Cases

For each material pair, 28 cases are generated covering 4–7 unit cells with 0, 1, or 2 defects:

- **No defect**: `{PREFIX}{N}00.h5` (e.g., `CA400.h5` = 4 cells, no defect)
- **Single defect at position d**: `{PREFIX}{N}{d}0.h5` (e.g., `CA520.h5` = 5 cells, defect at cell 2)
- **Double defects at positions i, j**: `{PREFIX}{N}{i}{j}.h5` (e.g., `CA724.h5` = 7 cells, defects at cells 2 & 4)

Defect positions follow these rules:
- Positions range from cell 2 to cell N−1 (1-indexed)
- Double defects must be non-adjacent (gap ≥ 2)

#### HDF5 Contents

Each `.h5` file contains 20,000 samples with layer thicknesses randomly sampled from U(5mm, 100mm):

| Dataset Key | Shape | Description |
|-------------|-------|-------------|
| `design_variable` | `[20000, 2N, 3]` | Layer parameters `[E (GPa), ρ (kg/m³), length (m)]` |
| `dispersion_relation_unitcell` | `[20000, 500]` | UDR at 500 frequency bins (0.1–50 kHz) |
| `dispersion_relation_supercell` | `[20000, 500]` | SDR at 500 frequency bins |
| `transmittance` | `[20000, 500]` | Power transmittance |
| `frequencies` | `[20000, 500]` | Frequency axis (kHz), identical per sample |

#### Simulation Parameters

| Parameter | Value |
|-----------|-------|
| Samples per case | 20,000 |
| Layer thickness range | 5–100 mm (uniform random) |
| Frequency range | 0.1–50 kHz (step: 0.1 kHz → 500 bins) |
| GPU batch size | 5,000 |
| Random seed | 7 |

### Step 2 — Build NPZ Caches

On first run, `main.py` automatically reads the HDF5 files and builds NPZ caches for efficient training:

```
data/cache_v2.3.1/
├── CA_train_dispersion.npz    # 80% of CA data
├── CA_valid_dispersion.npz    # 10%
├── CA_test_dispersion.npz     # 10%
├── SA_train_dispersion.npz
├── ...
└── AT_test_dispersion.npz
```

Each NPZ cache contains:

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `X` | `[N, 14, 3]` | float16 | Design variables padded to `max_cells=14` layers |
| `U` | `[N, 500]` | float16 | Unit-cell dispersion relation |
| `D` | `[N, 500]` | float16 | Supercell dispersion relation |
| `M` | `[N, 500]` | int8 | Band mask (0=Pass, 1=Gap, 2=Defect, 3=Don't Care) |
| `F` | `[N, 500]` | float16 | Frequencies |

The **band mask** `M` is computed automatically from UDR and SDR using `data_utils.build_band_mask()`, which identifies pass-band, bandgap, and defect-mode regions. Don't Care (class 3) masking is applied during cache building to focus training on one or two bandgap regions per sample.

### Step 3 — Build VAE Latent Caches

After VAE training, `main.py` encodes all samples through the VAE encoder and saves latent representations:

```
data/cache_v2.3.1/
├── CA_train_vae_latent.npz    # Latent vectors for DDPM training
├── CA_valid_vae_latent.npz
├── ...
```

| Key | Shape | Dtype | Description |
|-----|-------|-------|-------------|
| `Z` | `[N, 128]` | float16 | VAE latent mean (μ) |
| `C_mat` | `[N, 4]` | float16 | Material conditions `[E1, ρ1, E2, ρ2]` |
| `N_cells` | `[N]` | int64 | Number of layers per sample |
| `M` | `[N, 500]` | int64 | Band mask |
| `F` | `[N, 500]` | float16 | Frequencies |

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
  title={GenPnCFormer: Latent Diffusion Transformer for Inverse Design of Variable Phononic Crystals},
  author={D Lee, T Kim, JH Han, S Kim, BD Youn, SH Jo},
  year={2026}
}
```
