import os
import math
import numpy as np
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class VAE_Encoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d_in = 6  # (E, rho, len, Z, v, t)
        d_model = cfg.d_model
        
        self.input_proj = nn.Linear(d_in, d_model)
        self.pos_enc = PositionalEncoding(d_model, cfg.dropout, max_len=cfg.max_cells + 2)
        
        # [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=cfg.n_heads, batch_first=True,
            dim_feedforward=4*d_model, activation="gelu", dropout=cfg.dropout
        )
        self.tr_enc = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        

        self.fc_mean = nn.Linear(d_model, cfg.latent_dim) # From d_model (pooled) to latent
        self.fc_logvar = nn.Linear(d_model, cfg.latent_dim)
        self.norm_pooled = nn.LayerNorm(d_model)
        


    def forward(self, x_seq, n_cells):
        B = x_seq.size(0)
        valid_len = n_cells.long().clamp(min=1)

        x = self.input_proj(x_seq)  # [B, L, D]

        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]
        x = torch.cat((cls_tokens, x), dim=1)          # [B, L+1, D]

        x = self.pos_enc(x)

        # Padding mask (CLS at idx=0 is always valid)
        L_plus_1 = x.size(1)
        idx = torch.arange(L_plus_1, device=x.device).unsqueeze(0)  # [1, L+1]
        src_key_padding_mask = idx >= (valid_len.unsqueeze(1) + 1)

        # Guard against ALL-masked rows to prevent NaN from softmax(-inf)
        src_key_padding_mask = src_key_padding_mask.clone()
        all_masked = src_key_padding_mask.all(dim=-1)  # [B]
        if all_masked.any():
            src_key_padding_mask[all_masked, 0] = False  # Keep CLS valid

        x = self.tr_enc(x, src_key_padding_mask=src_key_padding_mask)

        pooled = x[:, 0, :]  # CLS pooling

        pooled_norm = self.norm_pooled(pooled)
        mu = self.fc_mean(pooled_norm)
        logvar = self.fc_logvar(pooled_norm)
        return mu, logvar



class VAE_Decoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model; self.max_cells = cfg.max_cells; d_out = 1
        self.pos_embed = nn.Parameter(torch.randn(1, self.max_cells, self.d_model) * 0.02)
        self.latent_proj = nn.Linear(cfg.latent_dim, self.d_model)
        self.material_proj = nn.Sequential(nn.Linear(4, self.d_model), nn.GELU(), nn.Linear(self.d_model, self.d_model))
        self.ncells_embed = nn.Embedding(cfg.max_cells + 1, self.d_model)
        dec_layer = nn.TransformerDecoderLayer(
            d_model=self.d_model, nhead=cfg.n_heads, dim_feedforward=4*self.d_model,
            dropout=cfg.dropout, batch_first=True, activation="gelu"
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=cfg.n_layers)
        self.norm_out = nn.LayerNorm(self.d_model)
        
        self.output_head = nn.Sequential(
            nn.Linear(self.d_model, d_out),
            nn.Tanh() 
        )

    def forward(self, z, material_conds, n_cells):
        B = z.size(0); valid_len = n_cells.long().clamp(min=1)
        query = self.pos_embed.expand(B, -1, -1)
        z_token = self.latent_proj(z); mat_embed = self.material_proj(material_conds)
        nc_embed = self.ncells_embed(n_cells); c_token = mat_embed + nc_embed
        memory = torch.stack([z_token, c_token], dim=1)
        L = self.max_cells; idx = torch.arange(L, device=z.device).unsqueeze(0)
        tgt_key_padding_mask = idx >= valid_len.unsqueeze(1)  # True = ignored

        # Guard against ALL-masked rows to prevent NaN
        tgt_key_padding_mask = tgt_key_padding_mask.clone()
        all_masked = tgt_key_padding_mask.all(dim=-1)  # [B]
        if all_masked.any():
            tgt_key_padding_mask[all_masked, 0] = False

        out = self.decoder(
            tgt=query, memory=memory, tgt_mask=None,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        out = self.norm_out(out)
        recon_lengths = self.output_head(out)  # [B, L, 1] in [-1, 1]
        recon_lengths = recon_lengths.masked_fill(tgt_key_padding_mask.unsqueeze(-1), 0.0)
        return recon_lengths



class ConditionalVAE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = VAE_Encoder(cfg)
        self.decoder = VAE_Decoder(cfg)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar); eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_seq, material_conds, n_cells):
        mu, logvar = self.encoder(x_seq, n_cells)
        z = self.reparameterize(mu, logvar)
        
        recon_lengths = self.decoder(z, material_conds, n_cells)
        
        return recon_lengths, mu, logvar

    def sample(self, z, material_conds, n_cells):
        return self.decoder(z, material_conds, n_cells)




LENGTH_MIN = 0.005
LENGTH_MAX = 0.100
LENGTH_RANGE = LENGTH_MAX - LENGTH_MIN

def scale_to_tanh(lengths):
    """Scale [LENGTH_MIN, LENGTH_MAX] to [-1, 1]."""
    lengths_01 = (lengths - LENGTH_MIN) / LENGTH_RANGE
    lengths_02 = lengths_01 * 2.0

    return lengths_02 - 1.0

def unscale_from_tanh(lengths_tanh):
    """Inverse of scale_to_tanh: [-1, 1] -> [LENGTH_MIN, LENGTH_MAX]."""
    lengths_02 = lengths_tanh + 1.0
    lengths_01 = lengths_02 / 2.0

    return (lengths_01 * LENGTH_RANGE) + LENGTH_MIN


class VAEDataset(Dataset):
    def __init__(self, npz_path: str, max_cells: int):
        print(f"Loading VAE data from {npz_path}...")
        self.max_cells = max_cells
        try:
            data = np.load(npz_path)

            X_orig = data["X"].astype(np.float32) # [N, L, 3]
            try:
                self.M = data["M"].astype(np.int64) # [N, K]
            except Exception:
                # If M is not available (e.g. old cache), create dummy or error
                print("Warning: M (Band Mask) not found in npz. Weighted sampling will drift.")
                self.M = np.zeros((X_orig.shape[0], 1), dtype=np.int64)
            
            try:
                self.F = data["F"].astype(np.float32) # [N, K]
            except Exception:
                # If F is not available, default linspace (will happen if old cache)
                print("Warning: F (Freqs) not found in npz. Using default.")
                K = self.M.shape[1] if self.M.ndim > 1 else 500
                self.F = np.linspace(0, np.pi, K, dtype=np.float32).reshape(1, -1).repeat(X_orig.shape[0], axis=0)

            N_orig, L_orig, C_orig = X_orig.shape
            
            if C_orig != 3:
                print(f"Warning: Expected 3 features (E, rho, length) but got {C_orig}.")

            self.X = np.zeros((N_orig, self.max_cells, 3), dtype=np.float32)
            use_len = min(L_orig, self.max_cells)
            self.X[:, :use_len, :] = X_orig[:, :use_len, :]
            
            # Recompute n_cells from valid (nonzero) lengths
            lengths = self.X[:, :, 2]
            valid_mask = (lengths > 0.0)
            self.N_cells = valid_mask.sum(axis=1).astype(np.int64) # [N]
            self.N_cells = np.clip(self.N_cells, 1, self.max_cells)
            
            print(f"Loaded and processed {self.X.shape[0]} samples.")
            
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            self.X = np.zeros((0, self.max_cells, 3), dtype=np.float32)
            self.N_cells = np.zeros((0,), dtype=np.int64)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        feats3 = self.X[idx]      # [max_cells, 3]
        n_cells = self.N_cells[idx] # int
        
        lengths_seq_orig = feats3[:, 2:3]  # [max_cells, 1]
        
        lengths_seq_scaled = scale_to_tanh(lengths_seq_orig)
        
        # Material conditions
        E1, rho1 = 0.0, 0.0; E2, rho2 = 0.0, 0.0
        if n_cells > 0: E1, rho1 = feats3[0, 0], feats3[0, 1]
        if n_cells > 1: E2, rho2 = feats3[1, 0], feats3[1, 1]
        else: E2, rho2 = E1, rho1
        material_conds = np.array([E1, rho1, E2, rho2], dtype=np.float32)
        
        # Construct 6-feature input: (E, rho, L, Z, v, t)
        X1 = feats3[:, 0]
        X2 = feats3[:, 1]
        X3 = feats3[:, 2] # Raw length
        
        # Derived features (valid where E>0 and rho>0)
        valid_mask = (X1 > 1e-6) & (X2 > 1e-6)
        
        X4 = np.zeros_like(X1)
        X4[valid_mask] = np.sqrt(X1[valid_mask] * X2[valid_mask])  # impedance Z
        
        X5 = np.zeros_like(X1)
        X5[valid_mask] = np.sqrt(X1[valid_mask] / X2[valid_mask])  # velocity v
        
        X6 = np.zeros_like(X1)
        valid_X6 = valid_mask & (X5 > 1e-9)
        X6[valid_X6] = X3[valid_X6] / X5[valid_X6]  # travel time t
        
        x_seq_6 = np.stack([X1, X2, X3, X4, X5, X6], axis=1)
        
        return {
            "x_seq": torch.from_numpy(x_seq_6.astype(np.float32)),  # [L, 6]
            "lengths": torch.from_numpy(lengths_seq_scaled),          # Target [L, 1]
            "material_conds": torch.from_numpy(material_conds),
            "N_cells": torch.tensor(n_cells, dtype=torch.long),
            "band_mask": torch.from_numpy(self.M[idx]),
            "freqs": torch.from_numpy(self.F[idx])
        }


def vae_loss_fn(recon_lengths, lengths, mu, logvar, valid_len, kld_beta=1e-5):

    B, L, _ = lengths.shape
    
    # Masked reconstruction loss (MSE)
    idx = torch.arange(L, device=lengths.device).unsqueeze(0)
    mask = (idx < valid_len.unsqueeze(1)).unsqueeze(-1) # [B, L, 1]
    
    squared_error = F.mse_loss(recon_lengths, lengths, reduction='none')
    squared_error_masked = squared_error * mask
    recon_loss = squared_error_masked.sum() / (mask.sum() + 1e-8)

    # KLD loss
    kld_loss_unsummed = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    kld_loss = torch.sum(kld_loss_unsummed, dim=1).mean()
    total_loss = recon_loss + kld_beta * kld_loss
    
    return total_loss, recon_loss, kld_loss


@torch.no_grad()
def validate_vae(model, val_dl, device, kld_beta):
    model.eval()
    total_loss, total_recon, total_kld = 0.0, 0.0, 0.0
    num_batches = 0
    
    for batch in val_dl:
        lengths = batch["lengths"].to(device)
        x_seq = batch["x_seq"].to(device)
        material_conds = batch["material_conds"].to(device)
        Ncells = batch["N_cells"].to(device)

        recon_lengths, mu, logvar = model(x_seq, material_conds, Ncells)
        
        loss, recon_loss, kld_loss = vae_loss_fn(
            recon_lengths, lengths, mu, logvar, Ncells, 
            kld_beta=kld_beta
        )
        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kld += kld_loss.item()
        num_batches += 1
        
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_recon = total_recon / num_batches if num_batches > 0 else 0.0
    avg_kld = total_kld / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, avg_recon, avg_kld



def train_vae(cfg, device, train_paths=None, valid_paths=None):
    print("\n--- Starting VAE Training (Stage 1) ---")
    
    if train_paths is None or valid_paths is None:
        raise ValueError("train_vae requires train_paths and valid_paths")
        
    EPOCHS = getattr(cfg, "epochs_vae", 30)
    
    FIXED_KLD_BETA = cfg.kld_beta
    
    WEIGHT_DECAY = getattr(cfg, "weight_decay", 1e-4)
    LR = getattr(cfg, "lr_vae", getattr(cfg, "lr", 1e-4))
    print(f"Using Fixed KLD Beta: {FIXED_KLD_BETA:.1e}")

    # Data loaders
    tr_datasets = [VAEDataset(p, cfg.max_cells) for p in train_paths]
    va_datasets = [VAEDataset(p, cfg.max_cells) for p in valid_paths]
    
    tr_ds = ConcatDataset(tr_datasets)
    va_ds = ConcatDataset(va_datasets)
    
    train_dl = DataLoader(tr_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=getattr(cfg, "num_workers", 0), pin_memory=False, drop_last=True)
    val_dl = DataLoader(va_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=getattr(cfg, "num_workers", 0), pin_memory=False)

    # Model & optimizer
    model = ConditionalVAE(cfg).to(device)
    # foreach=False avoids PyTorch 2.x _multi_tensor_adamw bug
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY, foreach=False)
    
    best_val_loss = float('inf')
    os.makedirs(cfg.save_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.save_dir, "vae_model_best.pt")
    
    for epoch in range(EPOCHS):
        
        current_kld_beta = FIXED_KLD_BETA

        model.train()
        pbar = tqdm(train_dl, desc=f"[VAE] E {epoch+1}/{EPOCHS}, β={current_kld_beta:.1e}")
        
        for batch in pbar:
            lengths = batch["lengths"].to(device)
            x_seq = batch["x_seq"].to(device) # [B, L, 3]
            material_conds = batch["material_conds"].to(device)
            Ncells = batch["N_cells"].to(device)

            recon_lengths, mu, logvar = model(x_seq, material_conds, Ncells)
            loss, recon_loss, kld_loss = vae_loss_fn(
                recon_lengths, lengths, mu, logvar, Ncells, 
                kld_beta=current_kld_beta
            )
            
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            
            pbar.set_postfix(
                loss=f"{loss.item():.4f}", 
                recon=f"{recon_loss.item():.4f}", 
                kld=f"{kld_loss.item():.2f}"
            )
            
        # Validation
        val_loss, val_recon, val_kld = validate_vae(model, val_dl, device, current_kld_beta)
        
        print(f"[VAE] Epoch {epoch+1} summary: "
              f"Val Loss: {val_loss:.6f} (Recon: {val_recon:.6f}, KLD: {val_kld:.2f})")


        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved best VAE model checkpoint to {ckpt_path} (Val Loss: {val_loss:.6f})")
            
    print(f"\n--- VAE Training Finished. Best model saved to {ckpt_path} ---")
    return ckpt_path




@torch.no_grad()
def build_vae_latent_cache(split: str, cfg: Any, vae_model, device, paths: list):
    """
    Build latent cache for EACH file in the list.
    Saves to: {original_name}_vae_latent.npz
    """
    print(f"[LatentCache] Building for '{split}' ({len(paths)} files)...")

    for in_path in paths:
        # data/cache/CA_train_dispersion.npz -> data/cache/CA_train_vae_latent.npz
        out_path = in_path.replace("_dispersion.npz", "_vae_latent.npz")
        if os.path.exists(out_path):
             print(f" -> Skipping {os.path.basename(out_path)} (Already exists).")
             continue
             
        print(f" -> Processing {os.path.basename(in_path)}...")
    

        dataset = VAEDataset(in_path, cfg.max_cells)
        data_loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=getattr(cfg, "num_workers", 0), pin_memory=False)
        
        try:
            M_all = np.load(in_path)["M"].astype(np.int64)
            if M_all.shape[0] != len(dataset):
                raise ValueError(f"Mask (M) count ({M_all.shape[0]}) does not match dataset count ({len(dataset)}).")

            if "F" in np.load(in_path):
                 F_all = np.load(in_path)["F"].astype(np.float32)
            else:
                 print("[VAE Latent Cache] Warning: F not found in input cache. Using default linspace.")
                 K = M_all.shape[1]
                 F_all = np.linspace(0, np.pi, K, dtype=np.float32).reshape(1, -1).repeat(M_all.shape[0], axis=0)
    
        except Exception as e:
            print(f"Error loading Band Masks (M) or Freqs (F) from {in_path}: {e}")
            continue
            
        vae_model.eval()
        
        Z_all_mu, C_mat_all, C_n_all = [], [], []

        
        for batch in tqdm(data_loader, desc=f"[VAE Encode '{split}']"):
            x_seq = batch["x_seq"].to(device)
            material_conds = batch["material_conds"].to(device)
            n_cells = batch["N_cells"].to(device)
            mu, logvar = vae_model.encoder(x_seq, n_cells)
            # Use mu (not reparameterized z) for stable latent cache
            Z_all_mu.append(mu.detach().cpu().numpy())
            C_mat_all.append(material_conds.detach().cpu().numpy())
            C_n_all.append(n_cells.detach().cpu().numpy())
    
        Z_all_mu = np.concatenate(Z_all_mu, axis=0)
        C_mat_all = np.concatenate(C_mat_all, axis=0)
        C_n_all = np.concatenate(C_n_all, axis=0)
        print(f"Z: {Z_all_mu.shape}, C_mat: {C_mat_all.shape}, N_cells: {C_n_all.shape}, M: {M_all.shape}")
        
        np.savez(
            out_path, 
            Z=Z_all_mu.astype(np.float16),
            C_mat=C_mat_all.astype(np.float16),
            N_cells=C_n_all.astype(np.int64),
            M=M_all,
            F=F_all.astype(np.float16)
        )
        
        print(f"[VAE Latent Cache] Saved new cache to: {out_path}")
        
    return True
