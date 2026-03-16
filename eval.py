import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from matplotlib.ticker import MaxNLocator
from matplotlib.patches import Rectangle

from diffusion import UNet1D, DDPM, DiffusionTransformer
from vae import VAE_Decoder, scale_to_tanh, unscale_from_tanh
from data_utils import build_band_mask

# surrogate (PnCFormer)
from surrogate.models.pncformer import PnCFormer

import gc


# ==============================================================================
# Utils: masks & padding
# ==============================================================================

def get_mask(n_cells, L):
    """[B] n_cells -> [B, L, 1] mask (True=valid)."""
    idx = torch.arange(L, device=n_cells.device).unsqueeze(0)
    mask = (idx < n_cells.unsqueeze(1)).unsqueeze(-1)
    return mask

def zero_pad_layers(layers: torch.Tensor, valid_mask_1D: torch.Tensor) -> torch.Tensor:
    """layers [B,L,3] * valid_mask_1D [B,L] -> pad구간 0."""
    return layers * valid_mask_1D.unsqueeze(-1).to(layers.dtype)


# ==============================================================================
# Test Dataset (uses F from npz if available)
# ==============================================================================

class TestDataset(Dataset):
    def __init__(self, npz_path: str, max_cells: int, mat_name: str = ""):
        print(f"Loading Test data from {npz_path}...")
        self.max_cells = max_cells
        self.mat_name  = mat_name

        # safe defaults
        self.Y_lengths_true_scaled = np.zeros((0, self.max_cells, 1), dtype=np.float32)
        self.C_materials = np.zeros((0, 4), dtype=np.float32)
        self.N_cells = np.zeros((0,), dtype=np.int64)
        self.M = np.zeros((0, 1), dtype=np.int64)
        self.F = np.zeros((0, 1), dtype=np.float32)

        try:
            z = np.load(npz_path)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return

        try:
            self.M = z["M"].astype(np.int64)          # [N, K]
            X_orig = z["X"].astype(np.float32)        # [N, L, 3]
            N_orig, K = self.M.shape
            if X_orig.shape[0] != N_orig:
                raise ValueError("X and M sample counts mismatch")

            # pad/truncate X to max_cells
            X_padded = np.zeros((N_orig, self.max_cells, 3), dtype=np.float32)
            use_len = min(X_orig.shape[1], self.max_cells)
            X_padded[:, :use_len, :] = X_orig[:, :use_len, :]

            # target lengths (scaled to tanh-range)
            Y_lengths_orig = X_padded[:, :, 2:3]
            self.Y_lengths_true_scaled = scale_to_tanh(Y_lengths_orig)

            # n_cells from positive length
            lengths_for_n = X_padded[:, :, 2]
            valid_mask = (lengths_for_n > 0.0)
            self.N_cells = np.clip(valid_mask.sum(axis=1).astype(np.int64), 1, self.max_cells)

            # materials [E1, rho1, E2, rho2] from first two cells
            self.C_materials = np.zeros((N_orig, 4), dtype=np.float32)
            for i in range(N_orig):
                feats3 = X_padded[i]
                n_cells = self.N_cells[i]
                E1, rho1 = feats3[0, 0], feats3[0, 1]
                if n_cells > 1:
                    E2, rho2 = feats3[1, 0], feats3[1, 1]
                else:
                    E2, rho2 = E1, rho1
                self.C_materials[i] = [E1, rho1, E2, rho2]

            # frequencies
            if "F" in z.files:
                self.F = z["F"].astype(np.float32)  # [N, K] or [K]
                if self.F.ndim == 1:
                    self.F = np.tile(self.F.reshape(1, -1), (N_orig, 1))
                if self.F.shape[1] != K:
                    self.F = self.F[:, :K]
            else:
                base_f = np.linspace(0.0, np.pi, K, dtype=np.float32)
                self.F = np.tile(base_f, (N_orig, 1))

            print(f"Loaded and processed {N_orig} test samples.")
        except Exception as e:
            print(f"Error parsing arrays from {npz_path}: {e}")

    def __len__(self):
        return self.Y_lengths_true_scaled.shape[0]

    def __getitem__(self, idx):
        return {
            "lengths_true_scaled": torch.from_numpy(self.Y_lengths_true_scaled[idx]),  # [L,1] tanh-scaled
            "material_conds": torch.from_numpy(self.C_materials[idx]),                 # [4]
            "n_cells": torch.tensor(self.N_cells[idx], dtype=torch.long),              # []
            "band_mask": torch.from_numpy(self.M[idx]),                                # [K]
            "freqs": torch.from_numpy(self.F[idx]),                                    # [K]
            "mat_name": self.mat_name,                                                 # str
        }


# ==============================================================================
# Length losses (geometry; optional diagnostics)
# ==============================================================================

def masked_mse_loss(pred_scaled, target_scaled, mask):
    se = F.mse_loss(pred_scaled, target_scaled, reduction='none')
    se = se * mask
    return se.sum() / (mask.sum() + 1e-8)

def masked_mae_loss(pred_scaled, target_scaled, mask):
    ae = F.l1_loss(pred_scaled, target_scaled, reduction='none')
    ae = ae * mask
    return ae.sum() / (mask.sum() + 1e-8)

def masked_mape_loss(pred_scaled, target_scaled, mask):
    pred = unscale_from_tanh(pred_scaled)
    targ = unscale_from_tanh(target_scaled)
    ape = torch.abs((targ - pred) / (targ + 1e-8))
    ape = ape * mask
    return (ape.sum() / (mask.sum() + 1e-8)) * 100.0


# ==============================================================================
# Band-mask metrics
# ==============================================================================

def compute_bandmask_metrics(pred_mask: torch.Tensor,
                             target_mask: torch.Tensor):
    with torch.no_grad():
        pred = pred_mask.long()
        target = target_mask.long()
        valid_mask = (target != 3)  # Don't Care mask

        # Accuracy
        if valid_mask.sum() > 0:
            acc = (pred[valid_mask] == target[valid_mask]).float().mean().item()
        else:
            acc = 0.0

        pred_def = (pred[valid_mask] == 2)
        targ_def = (target[valid_mask] == 2)
        tp = (pred_def & targ_def).sum().item()
        fp = (pred_def & (~targ_def)).sum().item()
        fn = ((~pred_def) & targ_def).sum().item()

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8) if (tp + fp + fn) > 0 else 0.0

    return {"acc": acc, "defect_prec": prec, "defect_rec": rec, "defect_f1": f1}


# ==============================================================================
# Feature builder: layers -> 6 features (E, ρ already scaled), zero-padding semantics
# ==============================================================================

def layers_to_six_features_torch(layers: torch.Tensor) -> torch.Tensor:
    """
    layers: [B,L,3] with columns [E_scaled, rho_scaled, length].
    Zero-padding rule: invalid cells produce zero features (no eps tricks).
    """
    modulus = layers[..., 0]
    density = layers[..., 1]
    length  = layers[..., 2]

    X1 = modulus
    X2 = density
    X3 = length

    mul_valid = (modulus > 0) & (density > 0)
    div_valid = mul_valid

    X4 = torch.zeros_like(modulus)
    X4[mul_valid] = torch.sqrt(modulus[mul_valid] * density[mul_valid])

    X5 = torch.zeros_like(modulus)
    X5[div_valid] = torch.sqrt(modulus[div_valid] / density[div_valid])

    X6 = torch.zeros_like(modulus)
    valid_X6 = div_valid & (X5 > 0)
    X6[valid_X6] = length[valid_X6] / X5[valid_X6]

    # print(X1, X2, X3, X4, X5, X6)

    return torch.stack([X1, X2, X3, X4, X5, X6], dim=-1)  # [B,L,6]


# ==============================================================================
# Surrogate band-solver (PnCFormer; uses batch freqs; mask-safety)
# ==============================================================================

class SurrogateBandSolver:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device

        # models
        self.model_udr = PnCFormer(
            x_input_dim=6, f_input_dim=1,
            d_model=cfg.d_model, nhead=cfg.n_heads,
            num_encoder_layers=cfg.n_layers, num_decoder_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(device)
        self.model_trans = PnCFormer(
            x_input_dim=6, f_input_dim=1,
            d_model=cfg.d_model, nhead=cfg.n_heads,
            num_encoder_layers=cfg.n_layers, num_decoder_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(device)
        self.model_sdr = PnCFormer(
            x_input_dim=6, f_input_dim=1,
            d_model=cfg.d_model, nhead=cfg.n_heads,
            num_encoder_layers=cfg.n_layers, num_decoder_layers=cfg.n_layers,
            dropout=cfg.dropout,
        ).to(device)

        base_dir = getattr(cfg, "surrogate_model_dir", "./surrogate/best_model")
        udr_ckpt = os.path.join(base_dir, "Transformer_UDR_v1.4.pth")
        tr_ckpt  = os.path.join(base_dir, "Transformer_TR_v1.4.pth")
        sdr_ckpt = os.path.join(base_dir, "Transformer_SDR_v1.4.pth")

        print(f"[Surrogate] Loading UDR from {udr_ckpt}")
        self._load_weights(self.model_udr, udr_ckpt)
        print(f"[Surrogate] Loading Transmittance from {tr_ckpt}")
        self._load_weights(self.model_trans, tr_ckpt)
        print(f"[Surrogate] Loading SDR from {sdr_ckpt}")
        self._load_weights(self.model_sdr, sdr_ckpt)

        self.model_udr.eval()
        self.model_trans.eval()
        self.model_sdr.eval()

        # Note: enable_nested_tensor=False is already set in PnCFormer.__init__
        # (pncformer.py line 66). The real segfault fix is in PnCFormer.forward:
        # pass bool mask (not float -inf) to TransformerDecoder.memory_key_padding_mask.

    def _load_weights(self, model, ckpt_path: str):
        try:
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
        except TypeError:
            state = torch.load(ckpt_path, map_location=self.device)
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]
        model.load_state_dict(state)

    def _build_layers_and_features(self,
                                   lengths_unscaled: torch.Tensor,  # [B,L,1]
                                   material_conds: torch.Tensor,    # [B,4] (E1,ρ1,E2,ρ2) already scaled like dataset
                                   n_cells: torch.Tensor):          # [B]
        device = lengths_unscaled.device
        B, L, _ = lengths_unscaled.shape

        length = lengths_unscaled.squeeze(-1)  # [B,L]
        E1, rho1 = material_conds[:, 0].unsqueeze(1), material_conds[:, 1].unsqueeze(1)
        E2, rho2 = material_conds[:, 2].unsqueeze(1), material_conds[:, 3].unsqueeze(1)

        idx = torch.arange(L, device=device).unsqueeze(0).expand(B, L)
        use_mat2 = (idx % 2 == 1)

        modulus = torch.where(use_mat2, E2, E1)      # [B,L]
        density = torch.where(use_mat2, rho2, rho1)  # [B,L]

        # valid mask (True=valid)
        idx_cells = torch.arange(L, device=device).unsqueeze(0)
        valid_mask = (idx_cells < n_cells.unsqueeze(1))  # [B,L]

        # form layers and hard zero-pad
        layers = torch.stack([modulus, density, length], dim=-1)  # [B,L,3]
        layers = zero_pad_layers(layers, valid_mask)              # pad->0

        # features (respect zero-padding)
        X_feat = layers_to_six_features_torch(layers)             # [B,L,6]
        return X_feat, valid_mask

    @torch.no_grad()
    def __call__(self, lengths_unscaled, material_conds, n_cells, freqs):
        """
        freqs: [B,K] or [B,K,1]; model expects [B,K,1]
        """
        B, L, _ = lengths_unscaled.shape

        X_feat, valid_mask = self._build_layers_and_features(lengths_unscaled, material_conds, n_cells)
        X_feat = X_feat.to(self.device)
        valid_mask = valid_mask.to(self.device).bool()     # True=valid

        # ✅ f는 [B, K] 로 유지 (절대 unsqueeze 하지 않음)
        f = freqs.to(self.device)
        if f.dim() == 3 and f.size(-1) == 1:
            f = f.squeeze(-1)  # 안전장치: [B,K,1]로 들어오면 [B,K]로 되돌림

        # PnCFormer expects src_key_padding_mask where True = Padding (Ignored)
        # valid_mask is True = Valid (Keep)
        # So we must pass ~valid_mask
        padding_mask = ~valid_mask
        
        udr_pred = self.model_udr(X_feat, f, src_key_padding_mask=padding_mask)   # [B,K]
        sdr_pred = self.model_sdr(X_feat, f, src_key_padding_mask=padding_mask)   # [B,K]
        
        # T_pred is used for Transmittance if needed, but not for mask
        # T_pred   = self.model_trans(X_feat, f, src_key_padding_mask=padding_mask) 

        band_masks = []
        for i in range(B):
            u_i = udr_pred[i].detach().cpu().numpy()
            s_i = sdr_pred[i].detach().cpu().numpy()
            band_masks.append(build_band_mask(u_i, s_i).astype(np.int64)) # Pass (UDR, SDR)
        # Keep result on CPU — metrics only need numpy anyway
        return torch.from_numpy(np.stack(band_masks, 0))


# ==============================================================================
# Inference & Evaluation
# ==============================================================================

@torch.no_grad()
def run_inference_and_evaluation(cfg, device,
                                 min_width: float = 0.0,
                                 w_cfg: float = 5.0,
                                 ddim_steps: int = 50,
                                 eta: float = 0.0,
                                 test_paths: list | None = None,
                                 diffusion_path: str | None = None,
                                 vae_path: str | None = None):
    """
    1) DDPM 샘플링 → VAE 디코딩
    2) Surrogate(PnCFormer)로 band mask 예측
    3) condition band mask와 비교
       - Bandgap (Class 1): IoU metrics
       - Defect (Class 2): Tolerance metrics
    """
    print("\n--- Starting Inference & Evaluation (VAE+DDPM + Surrogate) ---")

    # --- Checkpoints ---
    ddpm_ckpt_path = diffusion_path if diffusion_path else os.path.join(cfg.save_dir, "ddpm_transformer_best.pt")
    vae_ckpt_path  = vae_path if vae_path else os.path.join(cfg.save_dir, "vae_model_best.pt")

    # --- DDPM ---
    ddpm_state = torch.load(ddpm_ckpt_path, map_location=device, weights_only=True)
    
    # 💡 Model Selection
    # Disable nested tensor prototype to prevent C++ memory/state corruption during long loops
    if hasattr(torch.backends.cuda, 'enable_nested_tensor'):
        torch.backends.cuda.enable_nested_tensor = False
        print("[Eval] Disabled PyTorch NestedTensor prototype to ensure stability.")

    backbone_type = getattr(cfg, "diffusion_backbone", "transformer")
    print(f"[Eval] Using diffusion backbone: {backbone_type}")
    
    if backbone_type == "transformer":
        model = DiffusionTransformer(
            latent_dim=cfg.latent_dim,
            width=cfg.transformer_width,
            depth=cfg.transformer_depth,
            heads=cfg.transformer_heads,
            dropout=cfg.dropout,
            cfg=cfg
        ).to(device)
    else:
        model = UNet1D(cfg.latent_dim, cfg.unet_width, cfg.unet_depth, cfg.dropout, cfg).to(device)
        
    ddpm = DDPM(model, cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)
    if "ddpm" in ddpm_state:
        ddpm.load_state_dict(ddpm_state["ddpm"])
    else:
        print("[Eval] Loading generic state_dict for DDPM...")
        ddpm.load_state_dict(ddpm_state)
    ddpm.eval()
    print("Loaded trained DDPM successfully.")

    # --- VAE Decoder ---
    vae_decoder = VAE_Decoder(cfg).to(device)
    full_vae_state = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    decoder_keys = {k.replace("decoder.", "", 1): v for k, v in full_vae_state.items() if k.startswith("decoder.")}
    vae_decoder.load_state_dict(decoder_keys)
    vae_decoder.eval()
    print("Loaded trained VAE Decoder.")

    # --- Surrogate Solver ---
    surrogate_solver = SurrogateBandSolver(cfg, device)

    # --- Data ---
    # test_paths must be passed or we iterate
    if test_paths:
        paths = test_paths
    elif hasattr(cfg, "test_paths") and cfg.test_paths:
        paths = cfg.test_paths
    else:
        # Fallback if not passed (though we should pass it from main)
        # Try to guess
        paths = [
            os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz")
            for mat in ["CA", "SA", "TA", "AC", "AS", "AT"]
            if os.path.exists(os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz"))
        ]

    if not paths:
        print("[Eval] No test files found. Using hardcoded CA fallback.")
        paths = [os.path.join(cfg.cache_dir, "CA_test_with_mask.npz")]

    print(f"[Eval] Loading test data from {len(paths)} files: { [os.path.basename(p) for p in paths] }")

    # Build per-material dataset (keep mat_name tag)
    mat_names_order = ["TA", "CA", "SA", "AT", "AC", "AS"]  # display order
    def _mat_from_path(p):
        base = os.path.basename(p)          # e.g. "CA_test_dispersion.npz"
        return base.split("_")[0]           # → "CA"

    test_datasets = [
        TestDataset(p, cfg.max_cells, mat_name=_mat_from_path(p))
        for p in paths
    ]

    # --- Accumulators ---
    num_samples  = 0
    DMA_DELTA    = 0.5  # tolerance δ in frequency units (kHz)

    # helper: empty sub-accumulator dict
    def _acc():
        return {"mbof_iou": 0.0, "mbof_n": 0, "dma_match": 0.0, "dma_n": 0}

    # Total
    acc_total = _acc()
    # Per-material  (keys: mat name strings)
    all_mats   = list({_mat_from_path(p) for p in paths})
    acc_by_mat = {m: _acc() for m in all_mats}
    # Per-n_cells  (keys: 4,5,6,7 → layers 8,10,12,14)
    # n_cells in dataset = number of layers (AB pairs × 2), so cells=4 → n_cells=8
    # Group by unit-cell count = n_cells // 2
    acc_by_uc  = {uc: _acc() for uc in [4, 5, 6, 7]}

    W_CFG = getattr(cfg, "w_cfg_inference", w_cfg)
    print(f"Running inference with CFG scale (w) = {W_CFG}")
    print(f"Sampling method: DDIM (steps={ddim_steps}, eta={eta})")
    print(f"Metrics: mBOF (bandgap IoU)  |  DMA(δ={DMA_DELTA:.2f}) (defect accuracy)")

    def _update_acc(acc, mbof_m, dma_m):
        acc["mbof_iou"]   += mbof_m["iou_sum"]
        acc["mbof_n"]     += mbof_m["n_samples"]
        acc["dma_match"]  += dma_m["match_sum"]
        acc["dma_n"]      += dma_m["n_samples"]

    def _fmt(acc):
        mbof = acc["mbof_iou"] / max(acc["mbof_n"], 1)
        dma  = acc["dma_match"] / max(acc["dma_n"],  1)
        return mbof, dma, acc["mbof_n"], acc["dma_n"]

    # Iterate per-material dataset to keep mat_name accessible
    total_batches = sum(
        (len(ds) + cfg.batch_size - 1) // cfg.batch_size for ds in test_datasets
    )
    pbar = tqdm(total=total_batches, desc="[Inference & Eval]")

    for ds in test_datasets:
        mat = ds.mat_name
        dl  = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

        for batch in dl:
            # inputs
            material_conds = batch["material_conds"].to(device)       # [B, 4]
            n_cells = torch.clamp(batch["n_cells"].to(device), min=1) # [B]
            band_mask = batch["band_mask"].to(device)                 # [B, K]
            freqs = batch["freqs"].to(device)                         # [B, K]
            B = material_conds.size(0)
            num_samples += B

            # 1) DDPM → z (DDIM Sampling)
            z_sampled = ddpm.sample_ddim(
                B=B,
                band_mask=band_mask,
                material_conds=material_conds,
                n_cells=n_cells,
                z_dim=cfg.latent_dim,
                device=device,
                w=W_CFG,
                ddim_steps=ddim_steps,
                eta=eta
            )

            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                scale_factor = getattr(cfg, "latent_scale_factor", 1.0)
                z_sampled = z_sampled / scale_factor
                lengths_pred_scaled   = vae_decoder(z_sampled, material_conds, n_cells)
                lengths_pred_unscaled = unscale_from_tanh(lengths_pred_scaled)

            lengths_fp32 = lengths_pred_unscaled.float().detach()
            material_conds_fp32 = material_conds.float()
            band_mask_pred = surrogate_solver(
                lengths_fp32, material_conds_fp32, n_cells, freqs,
            )  # [B, K] — CPU

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # ── mBOF + DMA (total) ────────────────────────────────────────
            mbof_m = _mbof_batch(band_mask_pred, band_mask, freqs, min_width)
            dma_m  = _dma_batch(band_mask_pred,  band_mask, freqs, DMA_DELTA, min_width)

            _update_acc(acc_total,          mbof_m, dma_m)
            _update_acc(acc_by_mat[mat],    mbof_m, dma_m)

            # ── per-n_cells breakdown ─────────────────────────────────────
            nc_np = n_cells.cpu().numpy()              # [B] layer count (8/10/12/14)
            bm_cpu = band_mask.cpu()
            freq_cpu = freqs.cpu()
            bp_cpu   = band_mask_pred                  # already on CPU

            for uc in [4, 5, 6, 7]:
                layer_target = uc * 2                  # 4→8, 5→10, 6→12, 7→14
                sel = (nc_np == layer_target)          # bool [B]
                if not sel.any():
                    continue
                sel_t = torch.from_numpy(sel)
                mbof_uc = _mbof_batch(
                    bp_cpu[sel_t], bm_cpu[sel_t], freq_cpu[sel_t], min_width)
                dma_uc  = _dma_batch(
                    bp_cpu[sel_t], bm_cpu[sel_t], freq_cpu[sel_t], DMA_DELTA, min_width)
                _update_acc(acc_by_uc[uc], mbof_uc, dma_uc)

            # Progress bar
            cur_mbof, cur_dma, _, _ = _fmt(acc_total)
            pbar.set_postfix(mat=mat, mBOF=f"{cur_mbof:.4f}", DMA=f"{cur_dma:.4f}")
            pbar.update(1)

            # Explicit VRAM cleanup
            del z_sampled, lengths_pred_scaled, lengths_pred_unscaled
            del lengths_fp32, material_conds_fp32, band_mask_pred
            del material_conds, n_cells, band_mask, freqs
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    pbar.close()

    if num_samples == 0:
        print("No test samples evaluated.")
        return

    # ── Print Results ──────────────────────────────────────────────────────
    def _row(mbof, dma, mbof_n, dma_n):
        return f"{mbof:.4f} (n={mbof_n:>6})   {dma:.4f} (n={dma_n:>6})"

    sep  = "-" * 65
    hdr  = f"{'':8s}  {'mBOF':>26s}   {'DMA(δ=0.50)':>26s}"

    print(f"\n{'='*65}")
    print(f"  Evaluation Results   (total samples: {num_samples})")
    print(f"{'='*65}")

    # 1) By material
    print(f"\n  [1] By Material")
    print(hdr); print(sep)
    for m in mat_names_order:
        if m not in acc_by_mat:
            continue
        mbof, dma, mn, dn = _fmt(acc_by_mat[m])
        print(f"  {m:<6s}    {_row(mbof, dma, mn, dn)}")
    print(sep)
    mbof_t, dma_t, mn_t, dn_t = _fmt(acc_total)
    print(f"  {'Total':<6s}    {_row(mbof_t, dma_t, mn_t, dn_t)}")

    # 2) By unit-cell count
    print(f"\n  [2] By Unit-Cell Count")
    print(hdr); print(sep)
    for uc in [4, 5, 6, 7]:
        mbof, dma, mn, dn = _fmt(acc_by_uc[uc])
        print(f"  {uc} cells   {_row(mbof, dma, mn, dn)}")
    print(sep)
    print(f"  {'Total':<6s}    {_row(mbof_t, dma_t, mn_t, dn_t)}")
    print(f"{'='*65}\n")

    return {"mBOF": mbof_t, "DMA": dma_t,
            "by_mat": acc_by_mat, "by_uc": acc_by_uc}


# ==============================================================================
# Qualitative viz: Transmittance (original vs generated)
# ==============================================================================

@torch.no_grad()
def visualize_dispersion_comparison(cfg, device,
                                       save_dir: str = "vis_results",
                                       w_cfg: float = 5.0,
                                       ddim_steps: int = 50,
                                       test_paths: list | None = None,
                                       diffusion_path: str | None = None,
                                       vae_path: str | None = None):
    """
    User Request:
      - Bulk export: 6 Materials (CA..AT) * 28 Structures * 20 Samples = 3360 images.
      - File: save_dir/{MAT}/{MAT}{STRUCT}_{sample_idx}.png
      - Layout: Left=BandMask, Right=Transmittance
      - Style: Box graphs, no text/labels, Red(solid)/Blue(dotted), consistent fonts.
    """
    

    print(f"\n--- Starting Bulk Visualization ---")
    print(f"Output Directory: {save_dir}")
    os.makedirs(save_dir, exist_ok=True)

    # --- Load DDPM & VAE ---
    ddpm_ckpt_path = diffusion_path if diffusion_path else os.path.join(cfg.save_dir, "ddpm_transformer_best.pt")
    vae_ckpt_path  = vae_path if vae_path else os.path.join(cfg.save_dir, "vae_model_best.pt")

    ddpm_state = torch.load(ddpm_ckpt_path, map_location=device, weights_only=True)
    
    # model selection
    backbone_type = getattr(cfg, "diffusion_backbone", "transformer")
    if backbone_type == "transformer":
        model = DiffusionTransformer(
            latent_dim=cfg.latent_dim,
            width=cfg.transformer_width,
            depth=cfg.transformer_depth,
            heads=cfg.transformer_heads,
            dropout=cfg.dropout,
            cfg=cfg
        ).to(device)
    else:
        model = UNet1D(cfg.latent_dim, cfg.unet_width, cfg.unet_depth, cfg.dropout, cfg).to(device)
        
    ddpm = DDPM(model, cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)
    if "ddpm" in ddpm_state:
        ddpm.load_state_dict(ddpm_state["ddpm"])
    else:
        print("[Viz] Loading generic state_dict for DDPM...")
        ddpm.load_state_dict(ddpm_state)
    ddpm.eval()

    vae_decoder = VAE_Decoder(cfg).to(device)
    full_vae = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    dec_keys = {k.replace("decoder.", "", 1): v for k, v in full_vae.items() if k.startswith("decoder.")}
    vae_decoder.load_state_dict(dec_keys); vae_decoder.eval()

    # --- Surrogate (UDR & TR) ---
    base_dir = getattr(cfg, "surrogate_model_dir", "./surrogate/best_model")
    udr_ckpt = os.path.join(base_dir, "Transformer_UDR_v1.4.pth")
    tr_ckpt  = os.path.join(base_dir, "Transformer_TR_v1.4.pth")
    sdr_ckpt = os.path.join(base_dir, "Transformer_SDR_v1.4.pth")

    def _load_pnc(path):
        m = PnCFormer(x_input_dim=6, f_input_dim=1,
                      d_model=cfg.d_model, nhead=cfg.n_heads,
                      num_encoder_layers=cfg.n_layers, num_decoder_layers=cfg.n_layers,
                      dropout=cfg.dropout).to(device)
        try:
            st = torch.load(path, map_location=device, weights_only=True)
        except TypeError:
            st = torch.load(path, map_path=device)
        if isinstance(st, dict):
            st = st.get("state_dict", st.get("model", st))
        m.load_state_dict(st); m.eval()
        return m

    model_udr = _load_pnc(udr_ckpt)
    model_tr  = _load_pnc(tr_ckpt)
    model_sdr = _load_pnc(sdr_ckpt)

    import sys, traceback
    
    # --- Load test arrays (Fixed Order: CA->SA->TA->AC->AS->AT) ---
    materials = ["CA", "SA", "TA", "AC", "AS", "AT"]
    target_paths = [os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz") for mat in materials]
    
    print(f"[Viz] Loading visualization data from {len(target_paths)} files (Fixed Order)...")
    
    X_list, M_list, F_list, D_list = [], [], [], []
    # 💡 Track which materials were successfully loaded and their sizes
    loaded_materials = []  # (mat_name, n_samples) in load order
    for mat, p in zip(materials, target_paths):
        if not os.path.exists(p):
            print(f"[Warning] File not found: {p}. Skipping material {mat}.")
            continue
        try:
            z = np.load(p)
            X = z["X"].astype(np.float32)
            M = z["M"].astype(np.int64)
            # D = TMM Supercell Dispersion Relation (SDR) — exact simulation result
            D = z["D"].astype(np.float32) if "D" in z.files else np.zeros_like(X[:, :, 0])
            if "F" in z.files:
                F = z["F"].astype(np.float32)
                if F.ndim == 1: F = np.tile(F.reshape(1, -1), (X.shape[0], 1))
            else:
                K = M.shape[1]
                F = np.tile(np.linspace(0.0, np.pi, K, dtype=np.float32), (X.shape[0], 1))
            print(f"Loaded {os.path.basename(p)}: shape={X.shape}")
            X_list.append(X); M_list.append(M); F_list.append(F); D_list.append(D)
            loaded_materials.append((mat, X.shape[0]))
        except Exception as e:
            print(f"Error loading {p}: {e}")

    if not X_list:
        print("[Viz] No data loaded."); return

    X_all = np.concatenate(X_list, axis=0)
    M_all = np.concatenate(M_list, axis=0)
    F_all = np.concatenate(F_list, axis=0)
    D_all = np.concatenate(D_list, axis=0)  # TMM SDR: [N_total, K]
    print(f"[Viz] Total samples loaded: {X_all.shape[0]}")

    # Full structures list (matches sorted h5 filenames per material)
    structures_all = [400, 420, 430, 500, 520, 524, 530, 540, 600, 620, 624, 625, 630, 635,
                      640, 650, 700, 720, 724, 725, 726, 730, 735, 736, 740, 746, 750, 760]
    BLOCK_SIZE = 2000  # Each h5 file contributes exactly 2000 test samples (20000 * 0.1)

    # 💡 Build per-material structure->block mapping.
    #    The cache is built by iterating sorted h5 files and concatenating their
    #    test splits IN ORDER.  So block[i] = i-th h5 file's test samples.
    #    If a file was skipped during cache build (e.g. SA760), the block count
    #    is less than 28 — we detect this by comparing n_samples // BLOCK_SIZE.
    mat_struct_map = {}  # mat_name -> list of (struct_val, global_start_idx)
    cum_global = 0
    for mat_name_l, n_samp in loaded_materials:
        actual_n_blocks = n_samp // BLOCK_SIZE  # e.g. CA=28, SA=27
        present_structs = structures_all[:actual_n_blocks]  # first N structs (sorted h5 order)
        skipped = structures_all[actual_n_blocks:]          # trailing structs that were missing
        if skipped:
            print(f"[Viz] {mat_name_l}: {len(skipped)} file(s) missing from cache: "
                  f"{[f'{mat_name_l}{s}' for s in skipped]}")
        entries = []
        for blk_i, sv in enumerate(present_structs):
            entries.append((sv, cum_global + blk_i * BLOCK_SIZE))
        mat_struct_map[mat_name_l] = entries
        cum_global += n_samp
        print(f"[Viz] {mat_name_l}: {actual_n_blocks} blocks × {BLOCK_SIZE} = {n_samp} samples "
              f"(global offset {cum_global - n_samp})")

    # --- Helpers ---
    def zero_pad_layers(layers, valid_mask_1D):
        return layers * valid_mask_1D.unsqueeze(-1).to(layers.dtype)

    def layers_to_six_features_torch(layers: torch.Tensor) -> torch.Tensor:
        modulus = layers[..., 0]; density = layers[..., 1]; length = layers[..., 2]
        X1 = modulus; X2 = density; X3 = length
        mul_valid = (modulus > 0) & (density > 0); div_valid = mul_valid
        X4 = torch.zeros_like(modulus); X4[mul_valid] = torch.sqrt(modulus[mul_valid]*density[mul_valid])
        X5 = torch.zeros_like(modulus); X5[div_valid] = torch.sqrt(modulus[div_valid]/density[div_valid])
        X6 = torch.zeros_like(modulus); mask6 = div_valid & (X5 > 0); X6[mask6] = length[mask6]/X5[mask6]
        return torch.stack([X1, X2, X3, X4, X5, X6], dim=-1)

    def make_AB_layers(material_conds, lengths_unscaled, n_cells):
        idx = torch.arange(cfg.max_cells, device=device).unsqueeze(0)
        use_mat2 = (idx % 2 == 1)
        E1, r1 = material_conds[:, 0:1], material_conds[:, 1:2]
        E2, r2 = material_conds[:, 2:3], material_conds[:, 3:4]
        Ls = lengths_unscaled.squeeze(-1)
        E = torch.where(use_mat2, E2, E1).expand_as(Ls)
        R = torch.where(use_mat2, r2, r1).expand_as(Ls)
        valid_mask = (idx < n_cells.unsqueeze(1))
        layers = torch.stack([E, R, Ls], dim=-1)
        layers = zero_pad_layers(layers, valid_mask.squeeze(0))
        return layers, valid_mask

    def tr_from_layers(layers, f, valid_mask):
        X = layers_to_six_features_torch(layers)
        padding_mask = ~valid_mask
        return model_tr(X, f, src_key_padding_mask=padding_mask).squeeze(0).detach().cpu().numpy()

    def band_from_layers(layers, f, valid_mask):
        X = layers_to_six_features_torch(layers)
        padding_mask = ~valid_mask
        U = model_udr(X, f, src_key_padding_mask=padding_mask).squeeze(0).detach().cpu().numpy()
        S = model_sdr(X, f, src_key_padding_mask=padding_mask).squeeze(0).detach().cpu().numpy()
        return build_band_mask(U, S).astype(np.int64)

    def sdr_from_layers(layers, f, valid_mask):
        X = layers_to_six_features_torch(layers)
        padding_mask = ~valid_mask
        return model_sdr(X, f, src_key_padding_mask=padding_mask).squeeze(0).detach().cpu().numpy()

    # --- Helper for Single Sample Visualization ---
    def _viz_one_sample(layers_np, freqs_np, M_cond, sdr_true_tmm, save_path, fig=None, axes=None):
        # 1) Prepare Inputs
        n_cells_i = int(np.clip((layers_np[:, 2] > 0.0).sum(), 1, cfg.max_cells))
        mat_cond_arr = np.array([layers_np[0,0], layers_np[0,1],
                        (layers_np[1,0] if n_cells_i>1 else layers_np[0,0]),
                        (layers_np[1,1] if n_cells_i>1 else layers_np[0,1])], dtype=np.float32)
        material_conds = torch.from_numpy(mat_cond_arr).to(device).unsqueeze(0)
        n_cells_t      = torch.tensor([n_cells_i], dtype=torch.long, device=device)
        f_t            = torch.from_numpy(freqs_np).to(device).unsqueeze(0)

        # 2) Ground Truth
        layers_true = torch.from_numpy(layers_np).to(device).unsqueeze(0)
        pad = torch.zeros((1, cfg.max_cells, 3), device=device, dtype=layers_true.dtype)
        L0 = min(cfg.max_cells, layers_true.size(1)); pad[:, :L0, :] = layers_true[:, :L0, :]
        layers_true = pad
        idx_cells = torch.arange(cfg.max_cells, device=device).unsqueeze(0)
        valid_mask_true = (idx_cells < n_cells_t.unsqueeze(1))
        layers_true = zero_pad_layers(layers_true, valid_mask_true.squeeze(0))
        # GT SDR already provided as TMM array — no surrogate call needed
        # (Only Gen SDR is computed via surrogate)

        # 3) Generation
        z_sampled = ddpm.sample_ddim(B=1,
                    band_mask=torch.from_numpy(M_cond).to(device).unsqueeze(0),
                    material_conds=material_conds,
                    n_cells=n_cells_t,
                    z_dim=cfg.latent_dim,
                    device=device, w=w_cfg, ddim_steps=ddim_steps)
        
        scale_factor = getattr(cfg, "latent_scale_factor", 1.0)
        z_sampled = z_sampled / scale_factor
        
        lengths_pred_scaled = vae_decoder(z_sampled, material_conds, n_cells_t)
        lengths_pred = unscale_from_tanh(lengths_pred_scaled)
        layers_gen, valid_mask_gen = make_AB_layers(material_conds, lengths_pred, n_cells_t)
        
        S_gen    = sdr_from_layers(layers_gen, f_t, valid_mask_gen)
        band_gen = band_from_layers(layers_gen, f_t, valid_mask_gen)

        # GT: use TMM-based SDR and band mask from cache (exact simulation, not surrogate)
        S_true = sdr_true_tmm   # [K] numpy array — already float32 from cache

        # GT band mask: M_cond is the TMM-based conditioning mask from data build
        band_true = M_cond.copy()

        # Apply Don't-Care regions from conditioning mask to Gen prediction only
        band_gen[M_cond == 3] = 3

        # 4) Plotting
        # Left: Length Comparison, Middle: Band Mask, Right: Transmittance
        # Custom width ratios, e.g., similar horizontal length
        if fig is None or axes is None:
            fig, local_axes = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={'width_ratios': [1, 1, 1]})
            axLen, axL, axR = local_axes
        else:
            axLen, axL, axR = axes
            axLen.clear()
            axL.clear()
            axR.clear()
            fig.texts.clear()
        
        # --- Pre-calc Lengths ---
        L_gt = layers_np[:, 2]; L_gt = L_gt[L_gt > 1e-6]
        L_gen = layers_gen[0, :, 2].detach().cpu().numpy(); L_gen = L_gen[L_gen > 1e-6]
        
        # --- 1. Length Comparison (Left) ---
        axLen = axes[0]
        # X-axis: Layer Index 1..N
        idx_gt  = np.arange(1, len(L_gt)+1)
        idx_gen = np.arange(1, len(L_gen)+1)
        
        axLen.plot(idx_gt, L_gt,   color='r', linestyle='-', marker='o', markersize=3)
        axLen.plot(idx_gen, L_gen, color='b', linestyle=':', marker='x', markersize=4)
        
        # Remove labels and titles as requested
        axLen.set_ylabel("") 
        axLen.set_xlabel("")
        axLen.tick_params(direction='in', which='both', top=False, right=False,
                          labelbottom=False, labelleft=False)
        
        # Fixed scale X: 0-15 (ticks 0,5,10,15), Y: 0-0.10
        axLen.set_xlim(0, 15)
        axLen.set_xticks([0, 5, 10, 15])
        axLen.set_ylim(0.0, 0.10)
        
        # --- 2. Band Mask (Middle, Split Box) ---
        axL = axes[1]
        K = len(freqs_np)
        
        edges = np.empty(K+1, dtype=np.float32)
        mids  = (freqs_np[1:] + freqs_np[:-1]) / 2.0
        edges[1:K] = mids
        edges[0] = freqs_np[0] - (mids[0]-freqs_np[0]) if K>1 else freqs_np[0]
        edges[K] = freqs_np[-1] + (freqs_np[-1]-mids[-1]) if K>1 else freqs_np[-1]
        
        def _draw_mask_box(ax, mask_vals, y_bottom, height=0.8):
            for kk in range(K):
                c = mask_vals[kk]
                edge = 'none'; lw = 0.0
                if c == 2: 
                    continue # Draw defects on top later
                elif c == 1: 
                    col = (1.0, 0.8, 0.6) # Gap: Light Orange
                    edge = 'none'; lw = 0.0
                elif c == 3:
                    col = "lightgray" # Don't Care
                    edge = 'none'; lw = 0.0
                else: 
                    col = "white" # Pass
                    edge = 'none'; lw = 0.0
                
                width = edges[kk+1] - edges[kk]
                rect = Rectangle((edges[kk], y_bottom), width, height, 
                                 facecolor=col, edgecolor=edge, linewidth=lw, zorder=1)
                ax.add_patch(rect)

            # Draw defects on top as 1.5-thick lines (matching dispersion band line)
            for kk in range(K):
                if mask_vals[kk] == 2:
                    x_c = (edges[kk] + edges[kk+1]) / 2.0
                    ax.plot([x_c, x_c], [y_bottom, y_bottom+height], color=(0.0, 0.5, 0.0), linewidth=1.5, zorder=3)

            # Box Frame for the strip
            ax.plot([edges[0], edges[-1], edges[-1], edges[0], edges[0]], 
                    [y_bottom, y_bottom, y_bottom+height, y_bottom+height, y_bottom], 
                    color='k', linewidth=1.0)

        # Bottom Box (Gen): y=[0.0, 0.9]
        _draw_mask_box(axL, band_gen,  0.0, 0.9)
        # Top Box (GT surrogate): y=[1.0, 1.9]
        _draw_mask_box(axL, band_true, 1.0, 0.9)
        
        axL.set_xlim(edges[0], edges[-1])
        axL.set_ylim(0.0, 1.9) # Tight fit to top
        axL.set_yticks([])
        
        # Ticks only on bottom, but clean visual
        axL.tick_params(axis='x', direction='in', top=False, labelbottom=False)
        axL.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        
        # Restore full box frame to match neighbors' height visual
        for spine in axL.spines.values(): spine.set_visible(True)
        
        # --- 3. Supercell Dispersion Relation (Right) ---
        axR = axes[2]
        
        # Post-Processing: Snap local extrema at zone boundaries (0 or pi)
        def force_band_edges(S_new, thresh=0.15):
            S = S_new.copy()
            for i in range(1, len(S)-1):
                if S[i] >= S[i-1] and S[i] >= S[i+1] and (np.pi - S[i]) < thresh:
                    S[i] = np.pi
                elif S[i] <= S[i-1] and S[i] <= S[i+1] and S[i] < thresh:
                    S[i] = 0.0
            if S[0] < thresh: S[0] = 0.0
            elif (np.pi - S[0]) < thresh: S[0] = np.pi
            if S[-1] < thresh: S[-1] = 0.0
            elif (np.pi - S[-1]) < thresh: S[-1] = np.pi
            return S
            
        S_true = force_band_edges(S_true)
        S_gen  = force_band_edges(S_gen)

        # Highlight Bandgaps in light orange to match middle plot
        for kk in range(K):
            if M_cond[kk] == 1:
                axR.axhspan(edges[kk], edges[kk+1], color=(1.0, 0.8, 0.6), alpha=0.5, lw=0, zorder=-1)
                
        axR.plot(S_true, freqs_np, color='r', linestyle='-', linewidth=1.5)
        axR.plot(S_gen, freqs_np,  color='b', linestyle=':', linewidth=1.5)
        
        axR.set_xlim(0, np.pi)
        axR.set_ylim(freqs_np.min(), freqs_np.max())
        
        # Remove top and right ticks as requested
        axR.tick_params(direction='in', which='both', top=False, right=False, 
                        labelbottom=False, labelleft=False)
        for spine in axR.spines.values(): spine.set_visible(True)
        axR.set_xticks([0, np.pi/2, np.pi])
        axR.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        axR.yaxis.set_major_locator(MaxNLocator(nbins=5))
        
        # --- Text: Design Parameters ---
        def fmt_arr(arr):
            return "[" + ", ".join([f"{x:.3f}" for x in arr]) + "]"
        valid_m = (M_cond != 3)
        acc = np.mean(band_gen[valid_m] == M_cond[valid_m]) * 100 if valid_m.sum() > 0 else 0.0
        param_str = f"GT: {fmt_arr(L_gt)}\nGen: {fmt_arr(L_gen)}\nAcc: {acc:.1f}%"
        
        fig.text(0.5, 0.02, param_str, ha='center', va='bottom', fontsize=8, family='monospace')
        # Increased wspace from 0.15 to 0.35 for wider spacing between graphs
        fig.subplots_adjust(wspace=0.35, hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.25)
        
        fig.savefig(save_path, dpi=300)
        
        if axes is None:
            plt.close(fig)

    # --- Loop Logic ---
    cnt_saved = 0
    plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

    # Create a single figure to reuse
    global_fig, global_axes = plt.subplots(1, 3, figsize=(12, 3), gridspec_kw={'width_ratios': [1, 1, 1]})

    # RESUME LOGIC — skip materials/structures before these targets
    TARGET_MAT = "CA"
    TARGET_STRUCT = 400
    mat_order = {m: i for i, m in enumerate(materials)}
    struct_order = {s: i for i, s in enumerate(structures_all)}

    for mat_name, entries in mat_struct_map.items():
        mat_dir = os.path.join(save_dir, mat_name)
        os.makedirs(mat_dir, exist_ok=True)

        # RESUME: skip materials before target
        if mat_order.get(mat_name, 0) < mat_order.get(TARGET_MAT, 0):
            print(f"[Viz] Skipping {mat_name} (before {TARGET_MAT})")
            continue

        for struct_val, block_start in entries:
            # RESUME: skip structures before target (only in target material)
            if mat_name == TARGET_MAT and struct_order[struct_val] < struct_order[TARGET_STRUCT]:
                continue

            for k in range(20):
                # Spread 20 samples evenly over the BLOCK_SIZE window
                local_offset = k * (BLOCK_SIZE // 20)  # = k * 100 when BLOCK_SIZE=2000
                global_idx = block_start + local_offset

                if global_idx >= X_all.shape[0]:
                    print(f"Skipping {mat_name}{struct_val}_{k:02d}: global_idx={global_idx} "
                          f">= total={X_all.shape[0]}")
                    continue

                # File name: {MAT}{STRUCT}_{k:02d}.png  (k = 00..19)
                fname = f"{mat_name}{struct_val}_{k:02d}.png"
                save_path = os.path.join(mat_dir, fname)

                print(f"DEBUG: {fname} <- global_idx={global_idx} "
                      f"(block_start={block_start}, k={k}, offset={local_offset})")
                sys.stdout.flush()

                try:
                    _viz_one_sample(X_all[global_idx], F_all[global_idx], M_all[global_idx],
                                    D_all[global_idx],   # TMM SDR for GT
                                    save_path, fig=global_fig, axes=global_axes)
                    cnt_saved += 1
                    if cnt_saved % 20 == 0:
                        print(f"Saved {cnt_saved} images (last: {fname})")
                        sys.stdout.flush()
                except Exception as e:
                    print(f"[Error] {fname} (global={global_idx}): {e}")
                    traceback.print_exc()
                    sys.stdout.flush()

    print(f"Done. Total saved: {cnt_saved}")


    # --------------------------------------------------------------------------------------

@torch.no_grad()
def _defect_metrics_batch(pred_mask: torch.Tensor, target_mask: torch.Tensor):
    """
    pred_mask, target_mask: [B, K], 값 {0,1,2}
    배치 단위로 TP/FP/FN/총개수와 결함(2) 분포를 리턴
    """
    pred = pred_mask.long()
    targ = target_mask.long()

    pred_def = (pred == 2)
    targ_def = (targ == 2)

    tp = (pred_def & targ_def).sum().item()
    fp = (pred_def & (~targ_def)).sum().item()
    fn = ((~pred_def) & targ_def).sum().item()

    # 결함 비율 파악용
    pred_def_cnt = pred_def.sum().item()
    targ_def_cnt = targ_def.sum().item()
    total_elems  = pred.numel()

    return dict(
        tp=tp, fp=fp, fn=fn,
        pred_def_cnt=pred_def_cnt,
        targ_def_cnt=targ_def_cnt,
        total=total_elems
    )

def _mask_to_intervals_1d(freqs: np.ndarray,
                          mask: np.ndarray,
                          target_val: int = 2,
                          min_width: float = 0.0):
    """
    freqs: [K] 주파수 샘플 (비균일 가능)
    mask : [K] 정수 라벨 (defect=2 가정)
    target_val: 구간을 만들 라벨 값 (기본 2)
    min_width: 이 길이(주파수 폭) 미만 구간은 버림
    return: [(start_f, end_f), ...]  (좌폐우개 구간, end>start)
    """
    assert freqs.ndim == 1 and mask.ndim == 1 and freqs.shape[0] == mask.shape[0]
    K = freqs.shape[0]
    if K == 0:
        return []

    # 주파수 bin edge 계산 (비균일 지원)
    # edges[j] ~ freqs[j]와 freqs[j+1]의 중간값, 양 끝은 외삽
    edges = np.empty(K + 1, dtype=freqs.dtype)
    mids = (freqs[1:] + freqs[:-1]) / 2.0
    edges[1:K] = mids
    edges[0]   = freqs[0] - (mids[0] - freqs[0]) if K > 1 else freqs[0] - 1e-6
    edges[K]   = freqs[-1] + (freqs[-1] - mids[-1]) if K > 1 else freqs[-1] + 1e-6

    # 타겟 라벨 연속 구간 탐지
    is_t = (mask == target_val).astype(np.int32)
    # run-length: 변화 지점
    diff = np.diff(np.concatenate(([0], is_t, [0])))
    starts = np.where(diff ==  1)[0]   # 포함 index
    ends   = np.where(diff == -1)[0] - 1  # 포함 index

    intervals = []
    for s, e in zip(starts, ends):
        start_f = edges[s]      # s bin의 왼쪽 edge
        end_f   = edges[e + 1]  # e bin의 오른쪽 edge
        if end_f > start_f and (end_f - start_f) >= min_width:
            intervals.append((float(start_f), float(end_f)))
    return intervals


def _interval_iou(a: tuple, b: tuple) -> float:
    """IoU of two 1-D intervals."""
    (a0, a1), (b0, b1) = a, b
    inter = max(0.0, min(a1, b1) - max(a0, b0))
    uni   = max(a1, b1) - min(a0, b0)
    return (inter / uni) if uni > 0 else 0.0


@torch.no_grad()
def _mbof_batch(pred_mask: torch.Tensor,
                gt_mask:   torch.Tensor,
                freqs:     torch.Tensor,
                min_width: float = 0.0) -> dict:
    """
    Mean Bandgap Overlap Fidelity (mBOF) — accumulator for one batch.

        mBOF = (1/n_test) * Σ_i  IoU(I_i, Î_i)

    Per-sample IoU = (sum of matched-pair IoUs) / (number of GT bandgaps).
    Only samples with ≥1 GT bandgap (label=1) are counted.
    Returns cumulative iou_sum and n_samples for cross-batch accumulation.
    """
    pred_np = pred_mask.detach().cpu().numpy().copy()
    gt_np   = gt_mask.detach().cpu().numpy()
    freq_np = freqs.detach().cpu().numpy()
    pred_np[gt_np == 3] = 0  # zero-out predictions in don't-care regions

    iou_sum, n_samples = 0.0, 0
    for b in range(pred_np.shape[0]):
        gt_ivs = _mask_to_intervals_1d(freq_np[b], gt_np[b], 1, min_width)
        if len(gt_ivs) == 0:
            continue
        pred_ivs = _mask_to_intervals_1d(freq_np[b], pred_np[b], 1, min_width)

        if len(pred_ivs) == 0:
            sample_iou = 0.0
        else:
            used_p, used_g = set(), set()
            pairs = sorted(
                [(_interval_iou(p, g), pi, gi)
                 for gi, g in enumerate(gt_ivs)
                 for pi, p in enumerate(pred_ivs)
                 if _interval_iou(p, g) > 0],
                reverse=True
            )
            matched = []
            for iou, pi, gi in pairs:
                if pi in used_p or gi in used_g:
                    continue
                used_p.add(pi); used_g.add(gi)
                matched.append(iou)
            # Unmatched GT intervals contribute 0; normalise by total GT count
            sample_iou = sum(matched) / len(gt_ivs)

        iou_sum   += sample_iou
        n_samples += 1
    return {"iou_sum": iou_sum, "n_samples": n_samples}


@torch.no_grad()
def _dma_batch(pred_mask: torch.Tensor,
               gt_mask:   torch.Tensor,
               freqs:     torch.Tensor,
               delta:     float = 0.5,
               min_width: float = 0.0) -> dict:
    """
    Defect-band Matching Accuracy DMA(δ) — accumulator for one batch.

        DMA(δ) = (1/n_test) * Σ_i  [ |f_defect,i − f̂_defect,i| ≤ δ ]

    Per-sample score = (number of GT defects matched within δ) / (total GT defects).
    Only samples with ≥1 GT defect band (label=2) are counted.
    """
    pred_np = pred_mask.detach().cpu().numpy().copy()
    gt_np   = gt_mask.detach().cpu().numpy()
    freq_np = freqs.detach().cpu().numpy()
    pred_np[gt_np == 3] = 0

    match_sum, n_samples = 0.0, 0
    for b in range(pred_np.shape[0]):
        gt_ivs = _mask_to_intervals_1d(freq_np[b], gt_np[b], 2, min_width)
        if len(gt_ivs) == 0:
            continue
        pred_ivs = _mask_to_intervals_1d(freq_np[b], pred_np[b], 2, min_width)

        g_centers = [(g[0] + g[1]) / 2.0 for g in gt_ivs]
        p_centers = [(p[0] + p[1]) / 2.0 for p in pred_ivs]

        pairs = sorted(
            [(abs(gc - pc), pi, gi)
             for gi, gc in enumerate(g_centers)
             for pi, pc in enumerate(p_centers)
             if abs(gc - pc) <= delta],
            key=lambda x: x[0]
        )
        used_p, used_g = set(), set()
        matched = 0
        for _, pi, gi in pairs:
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            matched += 1

        match_sum += matched / len(gt_ivs)
        n_samples += 1
    return {"match_sum": match_sum, "n_samples": n_samples}


def _greedy_match_intervals(pred_intervals, gt_intervals, iou_thr=0.5):
    """
    예측/정답 구간 세트를 IoU 최대 기준으로 그리디 매칭.
    return:
      tp, fp, fn, matched_ious(list)
    """
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 0, 0, 0, []

    used_p = set()
    used_g = set()
    pairs = []
    # 모든 쌍의 IoU 계산 후 큰 것부터 매칭
    for gi, g in enumerate(gt_intervals):
        for pi, p in enumerate(pred_intervals):
            iou = _interval_iou(p, g)
            if iou > 0:
                pairs.append((iou, pi, gi))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_ious = []
    for iou, pi, gi in pairs:
        if iou < iou_thr:
            break
        if (pi in used_p) or (gi in used_g):
            continue
        used_p.add(pi); used_g.add(gi)
        matched_ious.append(iou)

    tp = len(matched_ious)
    fp = len(pred_intervals) - tp
    fn = len(gt_intervals) - tp
    return tp, fp, fn, matched_ious


def _match_intervals_tolerance(pred_intervals, gt_intervals, tol=0.5):
    """
    Match intervals based on CENTER distance <= tol.
    Greedy match by closest distance.
    tol: Tolerance for center distance (in frequency units, e.g. rad)
    """
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 0, 0, 0, []

    # Calculate centers
    p_centers = [(p[0] + p[1])/2.0 for p in pred_intervals]
    g_centers = [(g[0] + g[1])/2.0 for g in gt_intervals]
    
    pairs = []
    # Calculate all distances
    for gi, gc in enumerate(g_centers):
        for pi, pc in enumerate(p_centers):
            dist = abs(gc - pc)
            if dist <= tol:
                pairs.append((dist, pi, gi))
    
    # Sort by distance (ASCENDING) -> greedy match closest pairs
    pairs.sort(key=lambda x: x[0])

    used_p = set()
    used_g = set()
    matched_lists = [] # Store matched (pred, gt) or distances? user wants metrics. just count.
    
    for dist, pi, gi in pairs:
        if (pi in used_p) or (gi in used_g):
            continue
        used_p.add(pi)
        used_g.add(gi)
        matched_lists.append(dist)
        
    tp = len(matched_lists)
    fp = len(pred_intervals) - tp
    fn = len(gt_intervals) - tp
    
    return tp, fp, fn, matched_lists


@torch.no_grad()
def interval_iou_metrics_batch(pred_mask: torch.Tensor,
                               gt_mask: torch.Tensor,
                               freqs: torch.Tensor,
                               target_val: int = 2,
                               iou_thr: float = 0.5,
                               min_width: float = 0.0):
    """
    pred_mask, gt_mask: [B, K] (정수 라벨)
    freqs:              [B, K] (실제 주파수 샘플)
    interval IoU 기반 집계:
      - 구간 매칭 TP/FP/FN (IoU>=thr)
      - Precision / Recall / F1
      - 평균 IoU (매칭된 쌍만)
      - 샘플당 구간 개수 평균 등
    """
    assert pred_mask.shape == gt_mask.shape == freqs.shape
    B, K = pred_mask.shape

    tot_tp = tot_fp = tot_fn = 0
    all_matched_ious = []
    tot_pred_int = tot_gt_int = 0

    pred_mask_np = pred_mask.detach().cpu().numpy().copy()
    gt_mask_np   = gt_mask.detach().cpu().numpy()
    freqs_np     = freqs.detach().cpu().numpy()
    
    # Ignore Don't Care (3) regions to prevent False Positives
    pred_mask_np[gt_mask_np == 3] = 0

    for b in range(B):
        gis = _mask_to_intervals_1d(freqs_np[b], gt_mask_np[b],   target_val, min_width)
        if len(gis) == 0:
            continue # Skip samples where ground truth does not contain the target feature
        pis = _mask_to_intervals_1d(freqs_np[b], pred_mask_np[b], target_val, min_width)
        tp, fp, fn, matched_ious = _greedy_match_intervals(pis, gis, iou_thr=iou_thr)
        tot_tp += tp; tot_fp += fp; tot_fn += fn
        tot_pred_int += len(pis); tot_gt_int += len(gis)
        all_matched_ious.extend(matched_ious)

    prec = tot_tp / (tot_tp + tot_fp + 1e-8)
    rec  = tot_tp / (tot_tp + tot_fn + 1e-8)
    f1   = 2 * prec * rec / (prec + rec + 1e-8) if (tot_tp + tot_fp + tot_fn) > 0 else 0.0
    mean_iou = (float(np.mean(all_matched_ious)) if len(all_matched_ious) > 0 else 0.0)

    return {
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "precision": prec, "recall": rec, "f1": f1,
        "mean_iou": mean_iou,
        "pred_intervals": tot_pred_int, "gt_intervals": tot_gt_int,
        "matched_pairs": len(all_matched_ious)
    }


@torch.no_grad()
def defect_tolerance_metrics_batch(pred_mask: torch.Tensor,
                                   gt_mask: torch.Tensor,
                                   freqs: torch.Tensor,
                                   tol: float = 0.5,
                                   min_width: float = 0.0):
    """
    Class 2 (Defect) metrics using CENTER TOLERANCE matching.
    """
    B, K = pred_mask.shape
    tot_tp = tot_fp = tot_fn = 0
    tot_pred_int = tot_gt_int = 0
    
    pred_mask_np = pred_mask.detach().cpu().numpy().copy()
    gt_mask_np   = gt_mask.detach().cpu().numpy()
    freqs_np     = freqs.detach().cpu().numpy()
    
    # Ignore Don't Care (3) regions to prevent False Positives
    pred_mask_np[gt_mask_np == 3] = 0
    
    target_val = 2 # Defect

    for b in range(B):
        gis = _mask_to_intervals_1d(freqs_np[b], gt_mask_np[b],   target_val, min_width)
        if len(gis) == 0:
            continue # Skip samples where ground truth does not contain the target feature
        pis = _mask_to_intervals_1d(freqs_np[b], pred_mask_np[b], target_val, min_width)
        
        tp, fp, fn, _ = _match_intervals_tolerance(pis, gis, tol=tol)
        
        tot_tp += tp; tot_fp += fp; tot_fn += fn
        tot_pred_int += len(pis); tot_gt_int += len(gis)

    return {
        "tp": tot_tp, "fp": tot_fp, "fn": tot_fn,
        "pred_intervals": tot_pred_int,
        "gt_intervals": tot_gt_int
    }


@torch.no_grad()
def compare_condition_vs_surrogate_truth(
    cfg,
    device,
    max_batches: int | None = None,
    iou_thr: float = 0.5,
    min_width: float = 0.0,
):
    """
    Test set에 대해:
      - surrogate(UDR/TR)로 '기존 설계안'의 응답 → build_band_mask → surrogate-truth mask
      - condition으로 제공한 band mask와 직접 비교

    출력:
      1) 포인트 단위(pnt-level): Acc / Defect Prec / Recall / F1
      2) 구간 단위(interval-level): Precision / Recall / F1 / mean IoU
         (IoU 임계치 iou_thr로 매칭, min_width 미만 구간은 무시)
      3) 결함 비율 요약
    """
    # 1) Surrogate 준비
    surrogate_solver = SurrogateBandSolver(cfg, device)

    # 2) Data
    test_ds = TestDataset(os.path.join(cfg.cache_dir, "CA_test_with_mask.npz"), cfg.max_cells)
    test_dl = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=0)

    # -------------------------
    # 누적 통계 (포인트 단위)
    # -------------------------
    total_prec = 0.0
    total_rec  = 0.0
    total_f1   = 0.0
    num_samples = 0

    # 결함 분포/혼동 일부 집계
    total_tp = total_fp = total_fn = 0
    total_pred_def = total_targ_def = 0
    total_elems = 0

    # -------------------------
    # 누적 통계 (interval IoU)
    # -------------------------
    i_tot_tp = i_tot_fp = i_tot_fn = 0
    i_tot_matched = 0
    i_all_matched_ious_sum = 0.0  # mean IoU 계산용 (배치별이 아닌 전체 매칭쌍 평균)
    i_tot_pred_int = i_tot_gt_int = 0

    pbar = tqdm(test_dl, desc="[Cond vs SurrogateTruth + Interval IoU]")

    for b_idx, batch in enumerate(pbar):
        lengths_true_scaled = batch["lengths_true_scaled"].to(device)  # [B,L,1]
        material_conds = batch["material_conds"].to(device)            # [B,4]
        n_cells = torch.clamp(batch["n_cells"].to(device), min=1)      # [B]
        band_mask_cond = batch["band_mask"].to(device)                 # [B,K]
        freqs = batch["freqs"].to(device)                              # [B,K]
        B = lengths_true_scaled.size(0)

        # surrogate-truth mask (기존 설계안 길이 사용)
        lengths_true_unscaled = unscale_from_tanh(lengths_true_scaled)  # [B,L,1]
        band_mask_truth = surrogate_solver(
            lengths_unscaled=lengths_true_unscaled,
            material_conds=material_conds,
            n_cells=n_cells,
            freqs=freqs,   # [B,K]
        )  # [B,K]

        # ---------- 포인트 단위 메트릭 ----------
        metrics = compute_bandmask_metrics(band_mask_truth, band_mask_cond)
        total_prec += metrics["defect_prec"] * B
        total_rec  += metrics["defect_rec"] * B
        total_f1   += metrics["defect_f1"] * B
        num_samples += B

        # 결함 분포/혼동 집계
        d = _defect_metrics_batch(band_mask_truth, band_mask_cond)
        total_tp       += d["tp"]
        total_fp       += d["fp"]
        total_fn       += d["fn"]
        total_pred_def += d["pred_def_cnt"]
        total_targ_def += d["targ_def_cnt"]
        total_elems    += d["total"]

        # ---------- Interval IoU 메트릭 ----------
        i_metrics = interval_iou_metrics_batch(
            pred_mask=band_mask_truth,   # surrogate-truth vs condition (둘 중 어떤걸 pred로 둬도 TP/FP/FN은 동일)
            gt_mask=band_mask_cond,
            freqs=freqs,                 # [B,K]
            target_val=2,
            iou_thr=iou_thr,
            min_width=min_width,
        )
        i_tot_tp       += i_metrics["tp"]
        i_tot_fp       += i_metrics["fp"]
        i_tot_fn       += i_metrics["fn"]
        i_tot_pred_int += i_metrics["pred_intervals"]
        i_tot_gt_int   += i_metrics["gt_intervals"]

        # mean IoU 누적 (매칭된 쌍 수 * 평균값 = 총 합)
        i_all_matched_ious_sum += i_metrics["mean_iou"] * i_metrics["matched_pairs"]
        i_tot_matched          += i_metrics["matched_pairs"]

        pbar.set_postfix(
            pF1=f"{(total_f1/num_samples):.4f}",  # point-level F1
            iF1=f"{( (2*(i_tot_tp)/(2*i_tot_tp + i_tot_fp + i_tot_fn + 1e-8)) ):.4f}",  # interval-level F1 계산식과 동일
        )

        if (max_batches is not None) and (b_idx + 1 >= max_batches):
            break

    if num_samples == 0:
        print("No test samples.")
        return

    # ---------- 최종 집계 (포인트 단위) ----------
    avg_prec = total_prec / num_samples
    avg_rec  = total_rec  / num_samples
    avg_f1   = total_f1   / num_samples

    pred_def_ratio = total_pred_def / max(total_elems, 1)
    targ_def_ratio = total_targ_def / max(total_elems, 1)

    # ---------- 최종 집계 (interval IoU) ----------
    i_prec = i_tot_tp / (i_tot_tp + i_tot_fp + 1e-8)
    i_rec  = i_tot_tp / (i_tot_tp + i_tot_fn + 1e-8)
    i_f1   = 2 * i_prec * i_rec / (i_prec + i_rec + 1e-8) if (i_tot_tp + i_tot_fp + i_tot_fn) > 0 else 0.0
    i_mean_iou = (i_all_matched_ious_sum / i_tot_matched) if i_tot_matched > 0 else 0.0

    print("\n--- Condition vs Surrogate-Truth (band mask) ---")
    print(f"Total Test Samples: {num_samples}")
    
    print("\n[Metrics] 1) Bandgap (Class 1) - Only for samples with Target Bandgap")
    print(f"[Point]   Defect Prec:  {avg_prec:.4f}")
    print(f"[Point]   Defect Rec:   {avg_rec:.4f}")
    print(f"[Point]   Defect F1:    {avg_f1:.4f}")
    print(f"          TP / FP / FN: {total_tp} / {total_fp} / {total_fn}")
    print(f"          Defect ratio  (surrogate-truth): {pred_def_ratio*100:.2f}%")
    print(f"          Defect ratio  (condition mask) : {targ_def_ratio*100:.2f}%")

    print(f"\n[Interval IoU]  thr={iou_thr:.2f}, min_width={min_width:.5f}")
    print(f"          Precision:    {i_prec:.4f}")
    print(f"          Recall:       {i_rec:.4f}")
    print(f"          F1:           {i_f1:.4f}")
    print(f"          Mean IoU*:    {i_mean_iou:.4f}   (*matched pairs only)")
    print(f"          Pred/GT intervals: {i_tot_pred_int} / {i_tot_gt_int}")
    print(f"          Matched pairs:     {i_tot_matched}")
