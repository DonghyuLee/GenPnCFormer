import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
from tqdm import tqdm
from config import CFG
from data_utils import layers_to_six_features_torch
from surrogate.models.pncformer import PnCFormer
import matplotlib.pyplot as plt

# --- Dataset Definition ---
class SurrogateDataset(Dataset):
    def __init__(self, npz_path, target_key):
        """
        target_key: 'U' (UDR), 'D' (SDR), 'T' (Transmittance)
        """
        self.target_key = target_key
        try:
            data = np.load(npz_path)
            self.X = data["X"].astype(np.float32) # [N, L, 3] features (E/1000, rho/1000, L)
            self.F = data["F"].astype(np.float32) # [N, K]
            if target_key in data:
                self.Y = data[target_key].astype(np.float32) # [N, K]
            else:
                self.Y = None
                print(f"[Warning] Key {target_key} not in {npz_path}")
                
            # Compute N_cells for padding mask
            # X[:, :, 2] is length.
            self.lengths = self.X[:, :, 2]
                
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            self.X, self.F, self.Y = None, None, None

    def __len__(self):
        return len(self.X) if self.X is not None else 0

    def __getitem__(self, idx):
        return {
            "x_seq": torch.from_numpy(self.X[idx]),
            "freqs": torch.from_numpy(self.F[idx]),
            "target": torch.from_numpy(self.Y[idx]),
            "n_cells": torch.tensor((self.lengths[idx] > 1e-6).sum(), dtype=torch.long)
        }

def train_one_model(model_type, target_key, cfg, device):
    print(f"\n=== Training Surrogate Model: {model_type} (Target: {target_key}) ===")
    
    # 1. Select Cache File Suffix
    if target_key in ["U", "D"]:
        suffix_tr = "train_dispersion.npz"
        suffix_va = "valid_dispersion.npz"
    elif target_key == "T":
        suffix_tr = "train_transmittance.npz"
        suffix_va = "valid_transmittance.npz"
    else:
        raise ValueError(f"Unknown target key: {target_key}")

    target_folders = getattr(cfg, "target_folders", ["CA", "SA", "TA"]) # Multimat
    train_paths = [os.path.join(cfg.cache_dir, f"{f}_{suffix_tr}") for f in target_folders]
    valid_paths = [os.path.join(cfg.cache_dir, f"{f}_{suffix_va}") for f in target_folders]
    
    # Check if ALL cache files exist
    missing_cache = False
    for p in train_paths:
        if not os.path.exists(p):
            missing_cache = True
            print(f"Missing cache: {p}")
            break
            
    if missing_cache:
        print("Triggering cache build (build_cache_multimat)...")
        from data_utils import build_cache_multimat
        build_cache_multimat(cfg)
    else:
        print(f"Found existing cache for Target '{target_key}'. Skipping build.")
    
    train_ds = ConcatDataset([SurrogateDataset(p, target_key) for p in train_paths if os.path.exists(p)])
    val_ds   = ConcatDataset([SurrogateDataset(p, target_key) for p in valid_paths if os.path.exists(p)])
    
    if len(train_ds) == 0:
        print("No training data found.")
        return

    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,  num_workers=0, pin_memory=False)
    val_dl   = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False, num_workers=0, pin_memory=False)
    
    # 2. Model
    model = PnCFormer(
        x_input_dim=6,
        f_input_dim=1,
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        num_encoder_layers=cfg.n_layers,
        num_decoder_layers=cfg.n_layers,
        dropout=cfg.dropout
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4) # Slightly lower LR
    criterion = nn.MSELoss()
    
    # Use Cosine Annealing Scheduler
    epochs = 50 
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    # 3. Train
    best_loss = float('inf')
    save_dir = getattr(cfg, "surrogate_model_dir", "./surrogate/best_model")
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"Transformer_{model_type}_v2.1.pth")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_dl, desc=f"Ep {epoch+1}/{epochs}")
        for batch in pbar:
            x = batch["x_seq"].to(device)     # [B, L, 3] (E, rho, L)
            f = batch["freqs"].to(device)     # [B, K]
            y = batch["target"].to(device)    # [B, K]
            
            # Transform [B, L, 3] -> [B, L, 6]
            x = layers_to_six_features_torch(x) # [B, L, 6]
            
            # Padding Mask
            n_cells = batch["n_cells"].to(device).clamp(min=1)
            idx = torch.arange(x.size(1), device=device).unsqueeze(0)
            bool_mask = (idx >= n_cells.unsqueeze(1)) # True=Pad
            
            pred = model(x, f, src_key_padding_mask=bool_mask)
            
            loss = criterion(pred, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=loss.item())
            
        avg_train_loss = total_loss / len(train_dl)
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_dl:
                x = batch["x_seq"].to(device)
                f = batch["freqs"].to(device)
                y = batch["target"].to(device)
                n_cells = batch["n_cells"].to(device).clamp(min=1)
                idx = torch.arange(x.size(1), device=device).unsqueeze(0)
                bool_mask = (idx >= n_cells.unsqueeze(1))
                
                # Transform [B, L, 3] -> [B, L, 6]
                x = layers_to_six_features_torch(x) # [B, L, 6]
                
                pred = model(x, f, src_key_padding_mask=bool_mask)
                loss = criterion(pred, y)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_dl)
        
        # Scheduler Step (Cosine does not need metric)
        scheduler.step()
        
        print(f"Epoch {epoch+1}: Train MSE={avg_train_loss:.6f}, Val MSE={avg_val_loss:.6f}")
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            state = {"model": model.state_dict(), "config": cfg}
            torch.save(state, save_path)
            print(f"Saved Best {model_type} -> {save_path}")
            
    print(f"Finished {model_type}. Best Val MSE: {best_loss:.6f}")

def main():
    cfg = CFG()
    # Force settings if needed
    cfg.batch_size = 128
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # 1. Train UDR (Unitcell Dispersion)
    train_one_model("UDR", "U", cfg, device)
    
    # 2. Train SDR (Supercell Dispersion)
    train_one_model("SDR", "D", cfg, device)
    
    # 3. Train Transmittance
    train_one_model("TR", "T", cfg, device)

if __name__ == "__main__":
    main()
