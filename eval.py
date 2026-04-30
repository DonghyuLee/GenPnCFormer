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
from data_utils import build_band_mask, zero_pad_layers, layers_to_six_features_torch
from data_generation.tmm_torch import TorchTMM

# surrogate (PnCFormer)
from surrogate.models.pncformer import PnCFormer

import gc


# ---------------------------------------------------------------------------
# Test Dataset
# ---------------------------------------------------------------------------


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

# ---------------------------------------------------------------------------
# Feature builder: layers -> 6 features
# ---------------------------------------------------------------------------

# NOTE: layers_to_six_features_torch is imported from data_utils.
# A module-level alias is kept for backward compatibility with inverse_design.py.


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

        # f must stay [B, K] — never unsqueeze
        f = freqs.to(self.device)
        if f.dim() == 3 and f.size(-1) == 1:
            f = f.squeeze(-1)  # safety: [B,K,1] -> [B,K]

        # PnCFormer expects src_key_padding_mask where True = Padding (Ignored)
        # valid_mask is True = Valid (Keep)
        # So we must pass ~valid_mask
        padding_mask = ~valid_mask

        udr_pred = self.model_udr(X_feat, f, src_key_padding_mask=padding_mask)   # [B,K]
        sdr_pred = self.model_sdr(X_feat, f, src_key_padding_mask=padding_mask)   # [B,K]

        # Synchronize CUDA before GPU->CPU transfer
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        udr_np = udr_pred.detach().cpu().numpy()   # [B,K] — CPU copy
        sdr_np = sdr_pred.detach().cpu().numpy()   # [B,K] — CPU copy

        # Free GPU tensors immediately to prevent GC releasing during CUDA execution
        del udr_pred, sdr_pred, X_feat, f, padding_mask

        band_masks = []
        for i in range(B):
            band_masks.append(build_band_mask(udr_np[i], sdr_np[i]).astype(np.int64))

        return torch.from_numpy(np.stack(band_masks, 0))

class TMMBandSolver:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.freq_start_hz = 100.0
        self.freq_stop_hz = 50000.0
        self.freq_step_hz = 100.0

    @torch.no_grad()
    def __call__(self, lengths_unscaled, material_conds, n_cells, freqs):
        B, L, _ = lengths_unscaled.shape
        lengths_c = lengths_unscaled.squeeze(-1).clone()

        layer_idx = torch.arange(L, device=lengths_c.device).unsqueeze(0)
        valid_mask = (layer_idx < n_cells.to(lengths_c.device).unsqueeze(1))
        lengths_c = lengths_c * valid_mask.float()

        m_A = float(material_conds[0, 0].item()) * 1e9
        r_A = float(material_conds[0, 1].item())
        m_B = float(material_conds[0, 2].item()) * 1e9
        r_B = float(material_conds[0, 3].item())
        
        tmm = TorchTMM(
            modulus_A=m_A, density_A=r_A,
            modulus_B=m_B, density_B=r_B,
            batch_size=B,
            freq_start_hz=self.freq_start_hz,
            freq_stop_hz=self.freq_stop_hz,
            freq_step_hz=self.freq_step_hz,
            device=self.device
        )
        
        K = tmm.f.shape[1]

        I_cell = torch.zeros((B, K, 2, 2), dtype=torch.complex128, device=self.device)
        I_cell[..., 0, 0] = 1.0; I_cell[..., 1, 1] = 1.0

        if L >= 2:
            TM_A = tmm.TM(m_A, r_A, lengths_c[:, 0].unsqueeze(-1).double())
            TM_B = tmm.TM(m_B, r_B, lengths_c[:, 1].unsqueeze(-1).double())
            T_udr = torch.matmul(TM_B, TM_A)
            x_udr = torch.clamp(torch.real(T_udr[..., 0, 0] + T_udr[..., 1, 1]) / 2.0, -1.0, 1.0)
            UDR_pred = torch.acos(x_udr).cpu().numpy()
        else:
            UDR_pred = np.zeros((B, K))

        num_cells = L // 2
        cell_tms = []
        for c in range(num_cells):
            idx_A, idx_B = c * 2, c * 2 + 1
            l_A = lengths_c[:, idx_A].unsqueeze(-1).double()
            l_B = lengths_c[:, idx_B].unsqueeze(-1).double()
            valid_A = (l_A.squeeze(-1) > 0)
            valid_B = (l_B.squeeze(-1) > 0)
            TM_layer_A = tmm.TM(m_A, r_A, l_A)
            TM_layer_B = tmm.TM(m_B, r_B, l_B)
            TM_layer_A[~valid_A] = I_cell[~valid_A]
            TM_layer_B[~valid_B] = I_cell[~valid_B]
            cell_tms.append(torch.matmul(TM_layer_B, TM_layer_A))

        T_sdr = I_cell.clone()
        for cell_tm in reversed(cell_tms):
            T_sdr = torch.matmul(T_sdr, cell_tm)

        x_sdr = torch.clamp(torch.real(T_sdr[..., 0, 0] + T_sdr[..., 1, 1]) / 2.0, -1.0, 1.0)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        UDR_np = torch.acos(x_udr).cpu().numpy() if L >= 2 else np.zeros((B, K))
        SDR_np = torch.acos(x_sdr).cpu().numpy()

        del tmm, I_cell, cell_tms, T_sdr, x_sdr
        if L >= 2:
            del TM_A, TM_B, T_udr, x_udr
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        band_masks = []
        for i in range(B):
            band_masks.append(build_band_mask(UDR_np[i], SDR_np[i]).astype(np.int64))

        return torch.from_numpy(np.stack(band_masks, 0))

@torch.no_grad()
def run_inference_and_evaluation(cfg, device,
                                 min_width: float = 0.0,
                                 w_cfg: float = 5.0,
                                 ddim_steps: int = 50,
                                 eta: float = 0.0,
                                 test_paths: list | None = None,
                                 diffusion_path: str | None = None,
                                 vae_path: str | None = None,
                                 use_tmm: bool = False):

    evaluator_name = "TMM" if use_tmm else "Surrogate"
    ddpm_ckpt_path = diffusion_path if diffusion_path else os.path.join(cfg.save_dir, "ddpm_transformer_best.pt")
    vae_ckpt_path  = vae_path if vae_path else os.path.join(cfg.save_dir, "vae_model_best.pt")

    if hasattr(torch.backends.cuda, 'enable_nested_tensor'):
        torch.backends.cuda.enable_nested_tensor = False

    if hasattr(torch.backends, 'mha'):
        torch.backends.mha.set_fastpath_enabled(False)

    ddpm_state = torch.load(ddpm_ckpt_path, map_location=device, weights_only=True)
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
        ddpm.load_state_dict(ddpm_state)
    ddpm.eval()

    vae_decoder = VAE_Decoder(cfg).to(device)
    full_vae_state = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    decoder_keys = {k.replace("decoder.", "", 1): v for k, v in full_vae_state.items() if k.startswith("decoder.")}
    vae_decoder.load_state_dict(decoder_keys)
    vae_decoder.eval()

    if use_tmm:
        band_solver = TMMBandSolver(cfg, device)
    else:
        band_solver = SurrogateBandSolver(cfg, device)

    if test_paths:
        paths = test_paths
    elif hasattr(cfg, "test_paths") and cfg.test_paths:
        paths = cfg.test_paths
    else:
        paths = [
            os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz")
            for mat in ["CA", "SA", "TA", "AC", "AS", "AT"]
            if os.path.exists(os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz"))
        ]

    if not paths:
        paths = [os.path.join(cfg.cache_dir, "CA_test_with_mask.npz")]

    test_datasets = [
        TestDataset(p, cfg.max_cells, mat_name=_mat_from_path(p))
        for p in paths
    ]

    num_samples  = 0
    DMA_DELTA    = 0.5

    def _acc():
        return {"mbof_iou": 0.0, "mbof_n": 0, "dma_match": 0.0, "dma_n": 0}

    acc_total = _acc()
    all_mats   = list({_mat_from_path(p) for p in paths})
    acc_by_mat = {m: _acc() for m in all_mats}
    acc_by_uc  = {uc: _acc() for uc in [4, 5, 6, 7]}

    W_CFG = getattr(cfg, "w_cfg_inference", w_cfg)

    def _update_acc(acc, mbof_m, dma_m):
        acc["mbof_iou"]   += mbof_m["iou_sum"]
        acc["mbof_n"]     += mbof_m["n_samples"]
        acc["dma_match"]  += dma_m["match_sum"]
        acc["dma_n"]      += dma_m["n_samples"]

    def _fmt(acc):
        mbof = acc["mbof_iou"] / max(acc["mbof_n"], 1)
        dma  = acc["dma_match"] / max(acc["dma_n"],  1)
        return mbof, dma, acc["mbof_n"], acc["dma_n"]

    total_batches = sum(
        (len(ds) + cfg.batch_size - 1) // cfg.batch_size for ds in test_datasets
    )
    pbar = tqdm(total=total_batches)

    for ds in test_datasets:
        mat = ds.mat_name
        dl  = DataLoader(ds, batch_size=cfg.batch_size, shuffle=False,
                         num_workers=0, pin_memory=False)

        for batch_idx, batch in enumerate(dl):
            material_conds = batch["material_conds"].to(device)
            n_cells = torch.clamp(batch["n_cells"].to(device), min=1)
            band_mask = batch["band_mask"].to(device)
            freqs = batch["freqs"].to(device)
            B = material_conds.size(0)
            num_samples += B

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

            lengths_fp32 = torch.clamp(lengths_pred_unscaled.float().detach(), 0.0, 10.0)
            material_conds_fp32 = material_conds.float()
            
            if torch.isnan(lengths_fp32).any() or torch.isinf(lengths_fp32).any():
                lengths_fp32 = torch.nan_to_num(lengths_fp32, nan=0.0, posinf=0.0, neginf=0.0)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            band_mask_pred = band_solver(
                lengths_fp32, material_conds_fp32, n_cells, freqs,
            )

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            mbof_m = _mbof_batch(band_mask_pred, band_mask, freqs, min_width)
            dma_m  = _dma_batch(band_mask_pred,  band_mask, freqs, DMA_DELTA, min_width)

            _update_acc(acc_total,          mbof_m, dma_m)
            _update_acc(acc_by_mat[mat],    mbof_m, dma_m)

            nc_np = n_cells.cpu().numpy()
            bm_cpu = band_mask.cpu()
            freq_cpu = freqs.cpu()
            bp_cpu   = band_mask_pred

            for uc in [4, 5, 6, 7]:
                layer_target = uc * 2
                sel = (nc_np == layer_target)
                if not sel.any():
                    continue
                sel_t = torch.from_numpy(sel)
                mbof_uc = _mbof_batch(
                    bp_cpu[sel_t], bm_cpu[sel_t], freq_cpu[sel_t], min_width)
                dma_uc  = _dma_batch(
                    bp_cpu[sel_t], bm_cpu[sel_t], freq_cpu[sel_t], DMA_DELTA, min_width)
                _update_acc(acc_by_uc[uc], mbof_uc, dma_uc)

            pbar.set_postfix(mat=mat)
            pbar.update(1)

            del z_sampled, lengths_pred_scaled, lengths_pred_unscaled
            del lengths_fp32, material_conds_fp32, band_mask_pred
            del material_conds, n_cells, band_mask, freqs

            if batch_idx % 50 == 49:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    pbar.close()

    if num_samples == 0:
        return

    mbof_t, dma_t, mn_t, dn_t = _fmt(acc_total)
    return {"mBOF": mbof_t, "DMA": dma_t,
            "by_mat": acc_by_mat, "by_uc": acc_by_uc}

@torch.no_grad()
def visualize_dispersion_comparison(cfg, device,
                                       save_dir: str = "vis_results",
                                       w_cfg: float = 5.0,
                                       ddim_steps: int = 50,
                                       test_paths: list | None = None,
                                       diffusion_path: str | None = None,
                                       vae_path: str | None = None,
                                       use_tmm: bool = False):
    os.makedirs(save_dir, exist_ok=True)
    ddpm_ckpt_path = diffusion_path if diffusion_path else os.path.join(cfg.save_dir, "ddpm_transformer_best.pt")
    vae_ckpt_path  = vae_path if vae_path else os.path.join(cfg.save_dir, "vae_model_best.pt")

    ddpm_state = torch.load(ddpm_ckpt_path, map_location=device, weights_only=True)
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
        ddpm.load_state_dict(ddpm_state)
    ddpm.eval()

    vae_decoder = VAE_Decoder(cfg).to(device)
    full_vae = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    dec_keys = {k.replace("decoder.", "", 1): v for k, v in full_vae.items() if k.startswith("decoder.")}
    vae_decoder.load_state_dict(dec_keys); vae_decoder.eval()

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
            st = torch.load(path, map_location=device)
        if isinstance(st, dict):
            st = st.get("state_dict", st.get("model", st))
        m.load_state_dict(st); m.eval()
        return m

    if not use_tmm:
        model_udr = _load_pnc(udr_ckpt)
        model_tr  = _load_pnc(tr_ckpt)
        model_sdr = _load_pnc(sdr_ckpt)
    else:
        model_udr = model_tr = model_sdr = None

    import sys, traceback
    
    # --- Load test arrays (Fixed Order: CA->SA->TA->AC->AS->AT) ---
    materials = ["CA", "SA", "TA", "AC", "AS", "AT"]
    target_paths = [os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz") for mat in materials]
    
    print(f"[Viz] Loading visualization data from {len(target_paths)} files (Fixed Order)...")
    
    X_list, M_list, F_list, D_list = [], [], [], []
    # Track which materials were loaded
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

    # Build per-material structure->block mapping.
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
        # Do NOT squeeze valid_mask — keep [B, L] shape for any batch size
        layers = zero_pad_layers(layers, valid_mask)
        return layers, valid_mask


    def _tmm_udr_sdr(layers, m_A, r_A, m_B, r_B):
        """Compute UDR/SDR in the same order as get_dispersion_relation_supercell.
        SDR: build per-cell TM (TM_B @ TM_A), then chain multiply in reverse order.
        """
        B = layers.shape[0]; L = layers.shape[1]
        lengths_c = layers[..., 2]
        tmm = TorchTMM(m_A, r_A, m_B, r_B, B, 100.0, 50000.0, 100.0, device=device)
        K = tmm.f.shape[1]
        I_cell = torch.zeros((B, K, 2, 2), dtype=torch.complex128, device=device)
        I_cell[..., 0, 0] = 1.0; I_cell[..., 1, 1] = 1.0

        # UDR: TM_B @ TM_A (same as reference)
        TM_A0 = tmm.TM(m_A, r_A, lengths_c[:, 0].unsqueeze(-1).double())
        TM_B1 = tmm.TM(m_B, r_B, lengths_c[:, 1].unsqueeze(-1).double())
        T_udr = torch.matmul(TM_B1, TM_A0)
        x_udr = torch.clamp(torch.real(T_udr[..., 0, 0] + T_udr[..., 1, 1]) / 2., -1., 1.)
        U_np = torch.acos(x_udr).cpu().numpy()[0]  # [K]

        # SDR: per-cell (TM_B[2c+1] @ TM_A[2c]), reverse chain multiply
        num_cells = L // 2
        cell_tms = []
        for c in range(num_cells):
            idx_A, idx_B = c * 2, c * 2 + 1
            l_A = lengths_c[:, idx_A].unsqueeze(-1).double()
            l_B = lengths_c[:, idx_B].unsqueeze(-1).double()
            valid_A = (l_A.squeeze(-1) > 1e-8)
            valid_B = (l_B.squeeze(-1) > 1e-8)
            TM_layer_A = tmm.TM(m_A, r_A, l_A)
            TM_layer_B = tmm.TM(m_B, r_B, l_B)
            # Replace invalid layers with identity matrix
            TM_layer_A[~valid_A] = I_cell[~valid_A]
            TM_layer_B[~valid_B] = I_cell[~valid_B]
            cell_tm = torch.matmul(TM_layer_B, TM_layer_A)  # BA order
            cell_tms.append(cell_tm)

        # Reverse chain multiply: T = I @ cell[N-1] @ ... @ cell[0]
        T_sdr = I_cell.clone()
        for cell_tm in reversed(cell_tms):
            T_sdr = torch.matmul(T_sdr, cell_tm)

        x_sdr = torch.clamp(torch.real(T_sdr[..., 0, 0] + T_sdr[..., 1, 1]) / 2., -1., 1.)
        S_np = torch.acos(x_sdr).cpu().numpy()[0]  # [K]
        return U_np, S_np

    def band_from_layers(layers, f, valid_mask, mat_conds):
        if use_tmm:
            m_A, r_A = mat_conds[0,0].item() * 1e9, mat_conds[0,1].item()
            m_B, r_B = mat_conds[0,2].item() * 1e9, mat_conds[0,3].item()
            U_np, S_np = _tmm_udr_sdr(layers, m_A, r_A, m_B, r_B)
            return build_band_mask(U_np, S_np).astype(np.int64)
        X = layers_to_six_features_torch(layers)
        padding_mask = ~valid_mask
        U = model_udr(X, f, src_key_padding_mask=padding_mask).squeeze(0).detach().cpu().numpy()
        S = model_sdr(X, f, src_key_padding_mask=padding_mask).squeeze(0).detach().cpu().numpy()
        return build_band_mask(U, S).astype(np.int64)

    def sdr_from_layers(layers, f, valid_mask, mat_conds):
        if use_tmm:
            m_A, r_A = mat_conds[0,0].item() * 1e9, mat_conds[0,1].item()
            m_B, r_B = mat_conds[0,2].item() * 1e9, mat_conds[0,3].item()
            _, S_np = _tmm_udr_sdr(layers, m_A, r_A, m_B, r_B)
            return S_np
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
        
        S_gen    = sdr_from_layers(layers_gen, f_t, valid_mask_gen, material_conds)
        band_gen = band_from_layers(layers_gen, f_t, valid_mask_gen, material_conds)

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
            return "[" + ", ".join([f"{x:.6f}" for x in arr]) + "]"
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



    for mat_name, entries in mat_struct_map.items():
        mat_dir = os.path.join(save_dir, mat_name)
        os.makedirs(mat_dir, exist_ok=True)

        for struct_val, block_start in entries:
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

def _mask_to_intervals_1d(freqs: np.ndarray,
                          mask: np.ndarray,
                          target_val: int = 2,
                          min_width: float = 0.0):
    """
    freqs: [K] frequency samples (non-uniform possible)
    mask : [K] integer labels (defect=2 assumed)
    target_val: label value for intervals (default 2)
    min_width: intervals narrower than this (in freq units) are discarded
    return: [(start_f, end_f), ...]  (half-open intervals)
    """
    assert freqs.ndim == 1 and mask.ndim == 1 and freqs.shape[0] == mask.shape[0]
    K = freqs.shape[0]
    if K == 0:
        return []

    # Compute frequency bin edges (supports non-uniform spacing)
    # edges[j] ~ midpoint of freqs[j] and freqs[j+1], extrapolated at boundaries
    edges = np.empty(K + 1, dtype=freqs.dtype)
    mids = (freqs[1:] + freqs[:-1]) / 2.0
    edges[1:K] = mids
    edges[0]   = freqs[0] - (mids[0] - freqs[0]) if K > 1 else freqs[0] - 1e-6
    edges[K]   = freqs[-1] + (freqs[-1] - mids[-1]) if K > 1 else freqs[-1] + 1e-6

    # Detect contiguous regions of the target label
    is_t = (mask == target_val).astype(np.int32)
    # Run-length: change points
    diff = np.diff(np.concatenate(([0], is_t, [0])))
    starts = np.where(diff ==  1)[0]   # inclusive index
    ends   = np.where(diff == -1)[0] - 1  # inclusive index

    intervals = []
    for s, e in zip(starts, ends):
        start_f = edges[s]      # left edge of bin s
        end_f   = edges[e + 1]  # right edge of bin e
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
        # Robust frequency indexing: use shared vector if 1D, else per-sample if 2D
        f_b = freq_np if freq_np.ndim == 1 else freq_np[b]
        
        gt_ivs = _mask_to_intervals_1d(f_b, gt_np[b], 1, min_width)
        if len(gt_ivs) == 0:
            continue
        pred_ivs = _mask_to_intervals_1d(f_b, pred_np[b], 1, min_width)

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
        # Robust frequency indexing: use shared vector if 1D, else per-sample if 2D
        f_b = freq_np if freq_np.ndim == 1 else freq_np[b]

        gt_ivs = _mask_to_intervals_1d(f_b, gt_np[b], 2, min_width)
        if len(gt_ivs) == 0:
            continue
        pred_ivs = _mask_to_intervals_1d(f_b, pred_np[b], 2, min_width)

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

