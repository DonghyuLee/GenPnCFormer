import random
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
# ==============================================================================

SEED = 7 # Same as random seed in data_generation folder
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

@dataclass
class CFG:
    # Common Config
    data_dir: str = "data"
    cache_dir: str = "data/cache_v2.2.0"
    save_dir: str = "./checkpoints/v2.2.0"
    target_folders: list = ("CA", "SA", "TA", "AC", "AS", "AT") # 💡 Multi-Material Support

    # Data Config
    sample_size: int = 20000
    k_points: int = 500
    n_classes: int = 3
    max_cells: int = 14
    
    # Model Config
    dropout: float = 0.00
    batch_size: int = 256 
    num_workers: int = 8 
    
    # VAE 
    latent_dim: int = 128
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 4
    lr_vae: float = 4e-4 
    kld_beta: float = 1e-4 
    epochs_vae: int = 100 
      
    # Diffusion 
    diffusion_backbone: str = "transformer" # "unet" or "transformer"
    unet_width: int = 64
    unet_depth: int = 4
    transformer_width: int = 256
    transformer_depth: int = 8
    transformer_heads: int = 8
    
    # Encoder Config (BandMask)
    encoder_dim: int = 128     
    encoder_depth: int = 4    
    encoder_heads: int = 4    
    
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    defect_sample_weight: float = 3.0 
    lr_diffusion: float = 1e-4 
    warmup_epochs: int = 2 
    curriculum_switch_epoch: int = 100 
    latent_scale_factor: float = 5.04 
    epochs_diffusion: int = 50 
    cond_mode: str = "adaln-zero" 
   
