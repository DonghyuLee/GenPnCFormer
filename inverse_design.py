import os
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm
from sklearn.decomposition import PCA
from scipy.interpolate import CubicSpline

from config import CFG
from vae import VAE_Decoder
from diffusion import DDPM, DiffusionTransformer, UNet1D
from eval import unscale_from_tanh, TestDataset, _mask_to_intervals_1d, _interval_iou
from data_utils import build_band_mask
from data_generation.tmm_torch import TorchTMM


# ── FEM dispersion data loading ──────────────────────────────────────────────
FEM_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "FEM")

# Mapping: scenario index (0-based) → (figure number, number of FEM bands)
FEM_SCENARIO_MAP = {
    0: {"fig": 10, "numFreq": 10},
    1: {"fig": 11, "numFreq": 14},
    2: {"fig": 12, "numFreq": 10},
    3: {"fig": 13, "numFreq": 12},
    4: {"file": "Figure_13_2_Data.txt", "numFreq": 13},  # 5d: SA 25-42kHz / 28&36kHz
}


def load_fem_dispersion(fem_info: dict, num_k: int = 16):
    """Load FEM dispersion data from COMSOL export file.

    COMSOL exports data band-by-band (C-order, 16 rows per band).
    At kx=0, optical branches report freq≈0 due to degenerate eigenvalues;
    we exclude kx=0 for bands where this artifact occurs.

    Returns:
        list of (kx_arr, freq_arr) tuples — one per band,
        with kx normalized to [0, pi].
    """
    num_freq = fem_info["numFreq"]
    if "file" in fem_info:
        filepath = os.path.join(FEM_DATA_DIR, fem_info["file"])
    else:
        filepath = os.path.join(FEM_DATA_DIR, f"Figure_{fem_info['fig']}_Data.txt")
    if not os.path.exists(filepath):
        return None
    data = np.loadtxt(filepath)
    kx_raw = data[:num_k, 0]
    kx_all = kx_raw / kx_raw.max() * np.pi  # (num_k,) 0→pi
    freq_bands = data[:, 1].reshape(num_freq, num_k)  # C-order: (num_freq, num_k)

    freq_matrix = freq_bands.copy()

    # Greedy nearest-neighbor band tracking starting from kx[1]
    # (kx=0 can have COMSOL degenerate-pair ordering issues).
    # 1) Sort bands by kx[1] value as the reliable starting point.
    order = np.argsort(freq_matrix[:, 1])
    freq_matrix = freq_matrix[order]

    tracked = np.zeros_like(freq_matrix)
    tracked[:, 1] = freq_matrix[:, 1]   # anchor at kx[1]
    # Back-extrapolate kx[0] linearly from kx[1] and kx[2]
    for b in range(num_freq):
        slope = (freq_matrix[b, 2] - freq_matrix[b, 1]) / (kx_all[2] - kx_all[1])
        tracked[b, 0] = freq_matrix[b, 1] - slope * (kx_all[1] - kx_all[0])

    # Forward greedy tracking from kx[2] onward
    for k in range(2, num_k):
        available = list(freq_matrix[:, k])
        prev = tracked[:, k - 1]
        assigned = []
        used = [False] * num_freq
        for p in prev:
            best_i = min(
                (i for i in range(num_freq) if not used[i]),
                key=lambda i: abs(available[i] - p)
            )
            assigned.append(available[best_i])
            used[best_i] = True
        tracked[:, k] = assigned

    return [(kx_all, tracked[b]) for b in range(num_freq)]


# ── interval matching helpers (previously in eval.py) ────────────────────────
def _greedy_match_intervals(pred_intervals, gt_intervals, iou_thr=0.5):
    """Greedy matching by maximum IoU. Returns (tp, fp, fn, matched_ious)."""
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 0, 0, 0, []
    used_p, used_g, pairs = set(), set(), []
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
    return tp, len(pred_intervals) - tp, len(gt_intervals) - tp, matched_ious


def _match_intervals_tolerance(pred_intervals, gt_intervals, tol=0.5):
    """Greedy matching by center distance <= tol. Returns (tp, fp, fn, dists)."""
    if len(pred_intervals) == 0 and len(gt_intervals) == 0:
        return 0, 0, 0, []
    p_c = [(p[0]+p[1])/2.0 for p in pred_intervals]
    g_c = [(g[0]+g[1])/2.0 for g in gt_intervals]
    pairs = [(abs(gc-pc), pi, gi)
             for gi, gc in enumerate(g_c)
             for pi, pc in enumerate(p_c) if abs(gc-pc) <= tol]
    pairs.sort(key=lambda x: x[0])
    used_p, used_g, matched = set(), set(), []
    for dist, pi, gi in pairs:
        if (pi in used_p) or (gi in used_g):
            continue
        used_p.add(pi); used_g.add(gi)
        matched.append(dist)
    tp = len(matched)
    return tp, len(pred_intervals) - tp, len(gt_intervals) - tp, matched


def build_scenario_masks(k_points=500, f_start=100.0, f_step=100.0):
    def freq_to_idx(freq_khz):
        return max(0, min(k_points - 1, int(round((freq_khz - 0.1) / 0.1))))

    scenarios = []

    # 1) Single bandgap matching (Case A)
    mask1 = np.full(k_points, 3, dtype=np.int32)
    f1, f2 = 25, 35
    idx1, idx2 = freq_to_idx(f1), freq_to_idx(f2)
    pad_start = max(0, idx1 - 20)
    pad_end = min(k_points, idx2 + 20)
    mask1[pad_start:idx1] = 0
    mask1[idx2:pad_end] = 0
    mask1[idx1:idx2] = 1
    scenarios.append({
        "name": "1_SingleBandgap_25_35kHz", "mask": mask1, "type": "bandgap", "targets": [(f1, f2)],
        "mat": "CA", "nc": 6
    })

    # 2) double bandgap matching 
    mask2 = np.full(k_points, 3, dtype=np.int32)
    for (f1, f2) in [(15, 20), (35, 40)]:
        idx1, idx2 = freq_to_idx(f1), freq_to_idx(f2)
        pad_start = max(0, idx1 - 20)
        pad_end = min(k_points, idx2 + 20)
        mask2[pad_start:idx1] = 0
        mask2[idx2:pad_end] = 0
        mask2[idx1:idx2] = 1
    scenarios.append({
        "name": "2_DoubleBandgap_15_20_35_40kHz", "mask": mask2, "type": "bandgap", "targets": [(15, 20), (35, 40)],
        "mat": "TA", "nc": 5
    })

    # 3) single defect-band matching (in a 30-40 bandgap)
    mask3 = np.full(k_points, 3, dtype=np.int32)
    f1, f2 = 30, 40
    idx1, idx2 = freq_to_idx(f1), freq_to_idx(f2)
    pad_start = max(0, idx1 - 20)
    pad_end = min(k_points, idx2 + 20)
    mask3[pad_start:idx1] = 0
    mask3[idx2:pad_end] = 0
    mask3[idx1:idx2] = 1
    f = 35
    d_idx = freq_to_idx(f)
    mask3[d_idx] = 2
    scenarios.append({
        "name": "3_SingleDefect_35kHz_in_30_40", "mask": mask3, "type": "defect", "targets": [35],
        "mat": "CA", "nc": 6
    })

    # 4) double defect-band matching (in a 20-40 bandgap)
    mask4 = np.full(k_points, 3, dtype=np.int32)
    f1, f2 = 20, 40
    idx1, idx2 = freq_to_idx(f1), freq_to_idx(f2)
    pad_start = max(0, idx1 - 20)
    pad_end = min(k_points, idx2 + 20)
    mask4[pad_start:idx1] = 0
    mask4[idx2:pad_end] = 0
    mask4[idx1:idx2] = 1
    for f in [25, 35]:
        d_idx = freq_to_idx(f)
        mask4[d_idx] = 2
    scenarios.append({
        "name": "4_DoubleDefect_25_35kHz_in_20_40", "mask": mask4, "type": "defect", "targets": [25, 35],
        "mat": "SA", "nc": 7
    })

    # 5-A) SA 22-42kHz / defects 26,35 kHz
    mask5a = np.full(k_points, 3, dtype=np.int32)
    i1,i2=freq_to_idx(22),freq_to_idx(42)
    mask5a[max(0,i1-20):i1]=0; mask5a[i2:min(k_points,i2+20)]=0; mask5a[i1:i2]=1
    for f in [26,35]: mask5a[freq_to_idx(f)]=2
    scenarios.append({"name":"5a_SA_26_35_in_22_42","mask":mask5a,"type":"defect","targets":[26,35],"mat":"SA","nc":7})

    # 5-B) SA 20-40kHz / defects 23,33 kHz
    mask5b = np.full(k_points, 3, dtype=np.int32)
    i1,i2=freq_to_idx(20),freq_to_idx(40)
    mask5b[max(0,i1-20):i1]=0; mask5b[i2:min(k_points,i2+20)]=0; mask5b[i1:i2]=1
    for f in [23,33]: mask5b[freq_to_idx(f)]=2
    scenarios.append({"name":"5b_SA_23_33_in_20_40","mask":mask5b,"type":"defect","targets":[23,33],"mat":"SA","nc":7})

    # 5-C) SA 22-40kHz / defects 27,35 kHz
    mask5c = np.full(k_points, 3, dtype=np.int32)
    i1,i2=freq_to_idx(22),freq_to_idx(40)
    mask5c[max(0,i1-20):i1]=0; mask5c[i2:min(k_points,i2+20)]=0; mask5c[i1:i2]=1
    for f in [27,35]: mask5c[freq_to_idx(f)]=2
    scenarios.append({"name":"5c_SA_27_35_in_22_40","mask":mask5c,"type":"defect","targets":[27,35],"mat":"SA","nc":7})

    # 5-D) SA 25-42kHz / defects 28,36 kHz
    mask5d = np.full(k_points, 3, dtype=np.int32)
    i1,i2=freq_to_idx(25),freq_to_idx(42)
    mask5d[max(0,i1-20):i1]=0; mask5d[i2:min(k_points,i2+20)]=0; mask5d[i1:i2]=1
    for f in [28,36]: mask5d[freq_to_idx(f)]=2
    scenarios.append({"name":"5d_SA_28_36_in_25_42","mask":mask5d,"type":"defect","targets":[28,36],"mat":"SA","nc":7})

    # 5-E) SA 18-38kHz / defects 22,32 kHz
    mask5e = np.full(k_points, 3, dtype=np.int32)
    i1,i2=freq_to_idx(18),freq_to_idx(38)
    mask5e[max(0,i1-20):i1]=0; mask5e[i2:min(k_points,i2+20)]=0; mask5e[i1:i2]=1
    for f in [22,32]: mask5e[freq_to_idx(f)]=2
    scenarios.append({"name":"5e_SA_22_32_in_18_38","mask":mask5e,"type":"defect","targets":[22,32],"mat":"SA","nc":7})

    # 6) single defect-band matching — SA, 32 kHz in 25-40 kHz bandgap
    mask6 = np.full(k_points, 3, dtype=np.int32)
    f1, f2 = 25, 40
    idx1, idx2 = freq_to_idx(f1), freq_to_idx(f2)
    pad_start = max(0, idx1 - 20)
    pad_end = min(k_points, idx2 + 20)
    mask6[pad_start:idx1] = 0
    mask6[idx2:pad_end] = 0
    mask6[idx1:idx2] = 1
    f = 32
    d_idx = freq_to_idx(f)
    mask6[d_idx] = 2
    scenarios.append({
        "name": "6_SingleDefect_32kHz_in_25_40", "mask": mask6, "type": "defect", "targets": [32],
        "mat": "SA", "nc": 7
    })

    return scenarios


def check_success(pred_mask: np.ndarray, scenario: dict, freqs: np.ndarray,
                  iou_thr: float = 0.50, tol_kHz: float = 0.5):
    """
    Interval-based success check (consistent with eval.py metrics).
    - Bandgap : greedy IoU matching between predicted and target intervals, threshold iou_thr
    - Defect  : greedy center-tolerance matching, tolerance tol_kHz
    Don't Care (class 3) regions are zeroed out before interval extraction.
    Returns: (is_success: bool, mBOF: float, max_tol: float)
    """
    target_mask = scenario["mask"].copy()
    pred_clean  = pred_mask.copy()
    # Ignore Don't Care in both
    pred_clean[target_mask == 3] = 0

    if scenario["type"] == "bandgap":
        target_val = 1
        pis = _mask_to_intervals_1d(freqs, pred_clean, target_val)
        gis = _mask_to_intervals_1d(freqs, target_mask, target_val)
        if len(gis) == 0:
            return False, 0.0, 0.0
        # Calculate mBOF: sum of greedy-matched IoUs (IoU > 0) divided by number of GT intervals
        tp, fp, fn, matched_ious = _greedy_match_intervals(pis, gis, iou_thr=1e-8)
        sample_mbof = sum(matched_ious) / len(gis)
        return (sample_mbof >= iou_thr), sample_mbof, 0.0
        
    elif scenario["type"] == "defect":
        # 1) Defect: center-tolerance matching (DMA metric logic)
        target_val = 2
        pis_d = _mask_to_intervals_1d(freqs, pred_clean, target_val)
        gis_d = _mask_to_intervals_1d(freqs, target_mask, target_val)
        if len(gis_d) == 0:
            return False, 0.0, 999.0
        tp_d, fp_d, fn_d, matched_dists = _match_intervals_tolerance(pis_d, gis_d, tol=tol_kHz)
        dma_success = (tp_d == len(gis_d))
        max_dist = max([float(d) for d in matched_dists]) if len(matched_dists) > 0 else 999.0
        
        # 2) Background Bandgap: mBOF checking for the surrounding gap
        target_val = 1
        pis_b = _mask_to_intervals_1d(freqs, pred_clean, target_val)
        gis_b = _mask_to_intervals_1d(freqs, target_mask, target_val)
        if len(gis_b) > 0:
            tp_b, fp_b, fn_b, matched_ious = _greedy_match_intervals(pis_b, gis_b, iou_thr=1e-8)
            sample_mbof = sum(matched_ious) / len(gis_b)
            mbof_success = (sample_mbof >= iou_thr)
        else:
            mbof_success = True
            sample_mbof = 1.0
            
        return (dma_success and mbof_success), sample_mbof, max_dist
        
    return False, 0.0, 999.0


def batched_tmm_evaluation(lengths_c, nc, modulus_A, density_A, modulus_B, density_B, device):
    """
    lengths_c: [B, L, 1] tensor of lengths in meters
    nc: integer, number of layers
    Returns: UDR [B, 500], SDR [B, 500], masks [B, 500]
    """
    B = lengths_c.shape[0]
    L_max = nc
    
    tmm = TorchTMM(
        modulus_A=modulus_A, density_A=density_A,
        modulus_B=modulus_B, density_B=density_B,
        batch_size=B,
        freq_start_hz=100.0, 
        freq_stop_hz=50000.0, 
        freq_step_hz=100.0,
        device=device
    )
    
    K = tmm.f.shape[1]

    # Identity matrix template
    I_cell = torch.zeros((B, K, 2, 2), dtype=torch.complex128, device=device)
    I_cell[..., 0, 0] = 1.0; I_cell[..., 1, 1] = 1.0

    # --- 1) UDR: TM_B @ TM_A (same order as get_dispersion_relation_unitcell) ---
    if L_max >= 2:
        TM_A0 = tmm.TM(modulus_A, density_A, lengths_c[:, 0].double())
        TM_B1 = tmm.TM(modulus_B, density_B, lengths_c[:, 1].double())
        T_udr = torch.matmul(TM_B1, TM_A0)   # BA order
        x_udr = torch.clamp(torch.real(T_udr[..., 0, 0] + T_udr[..., 1, 1]) / 2.0, -1.0, 1.0)
        UDR_pred = torch.acos(x_udr).cpu().numpy()  # [B, K]
    else:
        UDR_pred = np.zeros((B, K))

    # --- 2) SDR: per-cell (TM_B @ TM_A), reverse chain multiply ---
    # Same method as get_dispersion_relation_supercell
    num_cells = L_max // 2
    cell_tms = []
    for c in range(num_cells):
        idx_A, idx_B = c * 2, c * 2 + 1
        l_A = lengths_c[:, idx_A].double()
        l_B = lengths_c[:, idx_B].double()
        valid_A = (l_A.squeeze(-1) > 0)
        valid_B = (l_B.squeeze(-1) > 0)
        TM_layer_A = tmm.TM(modulus_A, density_A, l_A)
        TM_layer_B = tmm.TM(modulus_B, density_B, l_B)
        TM_layer_A[~valid_A] = I_cell[~valid_A]
        TM_layer_B[~valid_B] = I_cell[~valid_B]
        cell_tms.append(torch.matmul(TM_layer_B, TM_layer_A))   # BA order

    # Reverse chain multiply: T = I @ cell[N-1] @ ... @ cell[0]
    T_sdr = I_cell.clone()
    for cell_tm in reversed(cell_tms):
        T_sdr = torch.matmul(T_sdr, cell_tm)

    x_sdr = torch.clamp(torch.real(T_sdr[..., 0, 0] + T_sdr[..., 1, 1]) / 2.0, -1.0, 1.0)
    SDR_pred = torch.acos(x_sdr).cpu().numpy()  # [B, K]

    # --- 3) Convert to masks ---
    masks = np.zeros((B, K), dtype=np.int32)
    for i in range(B):
        masks[i] = build_band_mask(UDR_pred[i], SDR_pred[i])

    return UDR_pred, SDR_pred, masks


def run_inverse_design(cfg, mode="adaln-zero"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n=========================================")
    print(f" Running Inverse Design ({mode.upper()}) ")
    print(f"=========================================\n")
    
    # 1. Load VAE Decoder
    vae_ckpt_path = os.path.join(cfg.save_dir, "vae_model_best.pt")
    vae_decoder = VAE_Decoder(cfg).to(device)
    full_vae_state = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
    dec_keys = {k.replace("decoder.", "", 1): v for k, v in full_vae_state.items() if k.startswith("decoder.")}
    vae_decoder.load_state_dict(dec_keys)
    vae_decoder.eval()

    # 2. Load Diffusion Model
        
    ddpm_dir = f"{cfg.save_dir}_{mode}"
    ddpm_ckpt_path = os.path.join(ddpm_dir, f"ddpm_{getattr(cfg, 'diffusion_backbone', 'transformer')}_best.pt")
    ddpm_state = torch.load(ddpm_ckpt_path, map_location=device, weights_only=True)
    
    if getattr(cfg, "diffusion_backbone", "transformer") == "transformer":
        model = DiffusionTransformer(
            latent_dim=cfg.latent_dim, width=cfg.transformer_width,
            depth=cfg.transformer_depth, heads=cfg.transformer_heads,
            dropout=cfg.dropout, cfg=cfg).to(device)
    else:
        model = UNet1D(cfg.latent_dim, cfg.unet_width, cfg.unet_depth, cfg.dropout, cfg).to(device)
        
    ddpm = DDPM(model, cfg.timesteps, cfg.beta_start, cfg.beta_end).to(device)
    if "ddpm" in ddpm_state: ddpm.load_state_dict(ddpm_state["ddpm"])
    else: ddpm.load_state_dict(ddpm_state)
    ddpm.eval()

    # surrogate solver is now TMM, no models to load
    scenarios = build_scenario_masks()
    # Scenario nc values represent layers. Scenario constraints define max_cells (actually max layers) 
    cell_counts = [6, 5, 6, 7] # Not used for looping anymore since each scenario defines it.
    
    freqs = np.linspace(0.1, 50.0, 500)
    freqs_tensor = torch.tensor(freqs, dtype=torch.float32, device=device).unsqueeze(0) # [1, 500]
    
    res_base_dir = "inverse_design_results"
    os.makedirs(res_base_dir, exist_ok=True)

    for scenario in scenarios:
        print(f"\n>> 🎯 Scenario: {scenario['name']}")
        scenario_dir = os.path.join(res_base_dir, scenario["name"])
        os.makedirs(scenario_dir, exist_ok=True)
        
        target_mask_tensor = torch.tensor(scenario["mask"], dtype=torch.long, device=device).unsqueeze(0)

        found_solution = False

        mat = scenario["mat"]
        # Convert "cells" to "layers" since nc corresponds to layers in DDPM conditioning
        nc = scenario["nc"] * 2 
        
        # Use hardcoded material info rather than loading TestDataset
        MAT_CONDC = {
            "CA": [0.1099853515625, 8.9609375, 0.07000732421875, 2.69921875],
            "SA": [0.199951171875, 7.8515625, 0.07000732421875, 2.69921875],
            "TA": [0.11602783203125, 4.5, 0.07000732421875, 2.69921875],
            "AC": [0.07000732421875, 2.69921875, 0.1099853515625, 8.9609375],
            "AS": [0.07000732421875, 2.69921875, 0.199951171875, 7.8515625],
            "AT": [0.07000732421875, 2.69921875, 0.11602783203125, 4.5]
        }
        
        if mat not in MAT_CONDC:
            print(f"  [WARN] Missing hardcoded material keys for {mat}. Skipping.")
            continue
            
        mat_cond_tensor = torch.tensor([MAT_CONDC[mat]], dtype=torch.float32, device=device) # [1, 4]

        nc_tensor = torch.tensor([nc], dtype=torch.long, device=device)
        print(f"  Sampling [{mat}] with {nc} layers ({scenario['nc']} cells)...")
        
        mod_A_pa = mat_cond_tensor[0, 0].item() * 1e9
        rho_A = mat_cond_tensor[0, 1].item()
        mod_B_pa = mat_cond_tensor[0, 2].item() * 1e9
        rho_B = mat_cond_tensor[0, 3].item()
        
        for attempt in range(10):  # Retry 10 times
            if found_solution: break
            B = 100
            
            gen_start_time = time.time()
            with torch.no_grad():
                # Use mini-batch to avoid VRAM OOM with B=5000
                CHUNK = 500
                all_lengths = []
                all_masks   = []
                all_udr     = []
                all_sdr     = []
                all_z       = []
                
                for c_start in range(0, B, CHUNK):
                    c_end = min(c_start + CHUNK, B)
                    c_size = c_end - c_start
                    
                    with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                        z_c = ddpm.sample_ddim(
                            B=c_size,
                            band_mask=target_mask_tensor.expand(c_size, -1),
                            material_conds=mat_cond_tensor.expand(c_size, -1),
                            n_cells=nc_tensor.expand(c_size),
                            z_dim=cfg.latent_dim,
                            device=device,
                            w=5.0,
                            ddim_steps=50,
                            eta=0.0
                        )
                        z_c_scaled = z_c / getattr(cfg, "latent_scale_factor", 1.0)
                        lengths_c_scaled = vae_decoder(z_c_scaled, mat_cond_tensor.expand(c_size, -1), nc_tensor.expand(c_size))
                        lengths_c = unscale_from_tanh(lengths_c_scaled)
                    
                    # Evaluate using TMM
                    udr_c, sdr_c, mask_c = batched_tmm_evaluation(
                        lengths_c.detach(), nc, mod_A_pa, rho_A, mod_B_pa, rho_B, device
                    )
                    
                    all_lengths.append(lengths_c.detach().cpu())
                    all_masks.append(mask_c)
                    all_udr.append(udr_c)
                    all_sdr.append(sdr_c)
                    all_z.append(z_c.cpu()) # Save the latent vector for PCA
                    
                    del z_c, z_c_scaled, lengths_c_scaled, lengths_c
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()
                
                lengths_pred_unscaled = torch.cat(all_lengths, dim=0).to(device)
                band_mask_pred = np.concatenate(all_masks, axis=0) # [B, K] np array
                z_all = torch.cat(all_z, dim=0).numpy() # [B, 128]
                udr_all = np.concatenate(all_udr, axis=0)
                sdr_all = np.concatenate(all_sdr, axis=0)
                
                gen_end_time = time.time()
                eval_start_time = time.time()
                
                # Evaluate constraints
                # Evaluate constraints — collect metrics for ALL samples
                success_samples = []
                all_mbofs = []
                all_dmas  = []  # only for defect scenarios
                for i in range(B):
                    is_success, s_mbof, s_tol = check_success(band_mask_pred[i], scenario, freqs, iou_thr=0.5)
                    all_mbofs.append(s_mbof)
                    if scenario["type"] == "defect":
                        all_dmas.append(s_tol if s_tol < 900.0 else None)  # 999=unmatched, skip
                    if is_success:
                        success_samples.append((i, s_mbof, s_tol))
                eval_end_time = time.time()

                # ── Per-scenario statistics (all B samples) ──────────────────
                mbof_arr = np.array(all_mbofs)
                print(f"\n    📊 [Stats over {B} samples]")
                print(f"       mBOF  — max: {mbof_arr.max():.4f}  mean: {mbof_arr.mean():.4f}  min: {mbof_arr.min():.4f}")
                if scenario["type"] == "defect":
                    valid_dmas = [d for d in all_dmas if d is not None]
                    if valid_dmas:
                        dma_arr = np.array(valid_dmas)
                        print(f"       DMA(tol kHz) — max: {dma_arr.max():.4f}  mean: {dma_arr.mean():.4f}  min: {dma_arr.min():.4f}")
                    else:
                        print(f"       DMA — no defect-matched samples")
                # ─────────────────────────────────────────────────────────────

                if len(success_samples) > 0:
                    print(f"    ✅ [SUCCESS] Found {len(success_samples)} matching structural designs on retry {attempt+1}!")
                    print(f"    ⏱️  Generation Time: {gen_end_time - gen_start_time:.2f}s | Evaluation Time: {eval_end_time - eval_start_time:.2f}s")
                    
                    # 1. Select the "Best" sample (highest mBOF, then lowest tolerance)
                    success_samples.sort(key=lambda x: (-float(x[1]), float(x[2])))
                    best_sample = success_samples[0]

                    # 2. Rank 2 = worst mBOF (last after descending sort)
                    if len(success_samples) > 1:
                        top_samples = [best_sample, success_samples[-1]]
                    else:
                        top_samples = success_samples

                    success_indices = [x[0] for x in success_samples]
                    

                    
                    z_success = z_all[success_indices] # [S, 128]
                    z_fail = np.delete(z_all, success_indices, axis=0)
                    
                    pca = PCA(n_components=3)
                    z_pca = pca.fit_transform(z_all)
                    
                    fig = plt.figure(figsize=(8, 6))
                    ax = fig.add_subplot(111, projection='3d')
                    # Plot failed lightly
                    fail_indices = np.delete(np.arange(B), success_indices)
                    ax.scatter(z_pca[fail_indices, 0], z_pca[fail_indices, 1], z_pca[fail_indices, 2], 
                               c='dimgray', alpha=0.3, label='Failed', s=15)
                    # Plot success brightly
                    ax.scatter(z_pca[success_indices, 0], z_pca[success_indices, 1], z_pca[success_indices, 2], 
                               c='red', alpha=0.8, label='Success', s=20)
                    
                    # Plot most distant differently
                    top_indices = [x[0] for x in top_samples]
                    ax.scatter(z_pca[top_indices, 0], z_pca[top_indices, 1], z_pca[top_indices, 2], 
                               c='gold', alpha=1.0, label='Most Distant FEASIBLE', s=150, marker='*', zorder=10)
                    
                    ax.set_title(f"Latent Space PCA ({scenario['name']})")
                    ax.set_xlabel("PC1")
                    ax.set_ylabel("PC2")
                    ax.set_zlabel("PC3")
                    ax.legend()
                    plt.tight_layout()
                    plt.savefig(f"{scenario_dir}/{mat}_{nc}cells_latent_3d.png", dpi=200)
                    plt.close(fig)
                    
                    # Iterate through the top 5
                    for rank, (best_idx, b_mbof, b_tol) in enumerate(top_samples):
                        # Fetch best features for Visualization
                        best_lengths = lengths_pred_unscaled[best_idx:best_idx+1]
                        udr_pred = udr_all[best_idx]
                        sdr_pred_t = sdr_all[best_idx]
                        mask_np = band_mask_pred[best_idx]
                        len_np = best_lengths.cpu().numpy().squeeze()
                        
                        # Save Data
                        np.savez(f"{scenario_dir}/{mat}_{nc}cells_data_rank{rank+1}.npz", 
                                 lengths=len_np, target_mask=scenario["mask"], 
                                 pred_mask=mask_np, udr=udr_pred, sdr=sdr_pred_t,
                                 mbof=b_mbof, max_tol=b_tol)
                                 
                        # ─── Visualization (3-panel, eval.py style) ───────
                        from matplotlib.patches import Rectangle
                        from matplotlib.ticker import MaxNLocator
    
                        freqs_np_vis = freqs
    
                        # Force band edges (snap SDR to 0 or pi at extrema)
                        def _force_band_edges(S, thresh=0.15):
                            S = S.copy()
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
    
                        sdr_pred_snap = _force_band_edges(sdr_pred_t)
    
                        # Frequency bin edges
                        K = len(freqs_np_vis)
                        edges = np.empty(K+1, dtype=np.float32)
                        mids  = (freqs_np_vis[1:] + freqs_np_vis[:-1]) / 2.0
                        edges[1:K] = mids
                        edges[0]   = freqs_np_vis[0] - (mids[0] - freqs_np_vis[0]) if K > 1 else freqs_np_vis[0]
                        edges[K]   = freqs_np_vis[-1] + (freqs_np_vis[-1] - mids[-1]) if K > 1 else freqs_np_vis[-1]
    
                        fig, axes_3 = plt.subplots(1, 3, figsize=(12, 3),
                                                   gridspec_kw={'width_ratios': [1, 1, 1]})
                        axLen, axM, axD = axes_3
    
                        # ── Panel 1: Design Variables (predicted lengths only) ──
                        L_gen_raw = best_lengths.detach().cpu().numpy().squeeze()
                        if L_gen_raw.ndim == 2: L_gen_raw = L_gen_raw[:, 0]  # [L,1] → [L]
                        L_gen_plot = L_gen_raw[L_gen_raw > 1e-6][:nc]
                        idx_gen    = np.arange(1, len(L_gen_plot)+1)
                        axLen.plot(idx_gen, L_gen_plot, color='b', linestyle=':', marker='x', markersize=4)
                        axLen.set_xlim(0, 15); axLen.set_xticks([0, 5, 10, 15])
                        axLen.set_ylim(0.0, 0.10)
                        axLen.tick_params(direction='in', which='both', top=False, right=False,
                                          labelbottom=False, labelleft=False)
    
                        # ── Panel 2: Band Mask (target top, predicted bottom) ──
                        COLOR = {0: 'white', 1: (1.0, 0.8, 0.6), 2: (0.0, 0.5, 0.0), 3: 'lightgray'}
    
                        def _draw_mask_strip(ax, mask_vals, y_bottom, height=0.9):
                            # Background coloring (skip 2)
                            for kk in range(K):
                                c = int(mask_vals[kk])
                                if c == 2: continue # Don't draw rectangle for defect here
                                col = COLOR.get(c, 'white')
                                w = edges[kk+1] - edges[kk]
                                ax.add_patch(Rectangle((edges[kk], y_bottom), w, height,
                                                       facecolor=col, edgecolor='none', zorder=1))
                            
                            # Find intervals of class 2 to draw a single crisp line for each
                            is_defect = np.array(mask_vals == 2, dtype=np.int32)
                            diff = np.diff(np.concatenate(([0], is_defect, [0])))
                            starts = np.where(diff == 1)[0]
                            ends = np.where(diff == -1)[0]
                            for s, e in zip(starts, ends):
                                # Center frequency of the defect interval
                                f_center = (edges[s] + edges[e]) / 2.0
                                ax.plot([f_center, f_center], [y_bottom, y_bottom+height],
                                        color=(0, 0.5, 0), linewidth=1.5, zorder=3)
                                
                            ax.plot([edges[0], edges[-1], edges[-1], edges[0], edges[0]],
                                    [y_bottom, y_bottom, y_bottom+height, y_bottom+height, y_bottom],
                                    color='k', linewidth=1.0)
    
                        # Generated mask: apply Don't Care from target scenario
                        # so direct comparison is possible (same as eval.py)
                        mask_np_viz = mask_np.copy()
                        mask_np_viz[scenario["mask"] == 3] = 3

                        _draw_mask_strip(axM, mask_np_viz,          0.0, 0.9)  # bottom: predicted
                        _draw_mask_strip(axM, scenario["mask"], 1.0, 0.9)  # top: target
                        axM.set_xlim(edges[0], edges[-1])
                        axM.set_ylim(0.0, 1.9)
                        axM.set_yticks([])
                        axM.tick_params(axis='x', direction='in', top=False, labelbottom=False)
                        axM.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                        for sp in axM.spines.values(): sp.set_visible(True)
    
                        # ── Panel 3: SDR Dispersion Relation (predicted only) ──
                        for kk in range(K):
                            if mask_np[kk] == 1:
                                axD.axhspan(edges[kk], edges[kk+1], color=(1.0, 0.8, 0.6), alpha=0.5, lw=0, zorder=-1)
                        
                        # Draw defect positions as horizontal green lines
                        is_defect_d = np.array(mask_np == 2, dtype=np.int32)
                        diff_d = np.diff(np.concatenate(([0], is_defect_d, [0])))
                        starts_d = np.where(diff_d == 1)[0]
                        ends_d = np.where(diff_d == -1)[0]
                        for s, e in zip(starts_d, ends_d):
                            f_center = (edges[s] + edges[e]) / 2.0
                            # Draw a star at the center x-axis value for visibility
                            axD.plot(np.pi/2, f_center, marker='*', markersize=12, color='g', zorder=3)
    
                        axD.plot(sdr_pred_snap, freqs_np_vis, color='b', linestyle=':', linewidth=1.5)

                        # ── FEM dispersion overlay (solid black lines) ──
                        scenario_idx = scenarios.index(scenario)
                        if scenario_idx in FEM_SCENARIO_MAP:
                            fem_info = FEM_SCENARIO_MAP[scenario_idx]
                            fem_bands = load_fem_dispersion(
                                fem_info["fig"], fem_info["numFreq"])
                            if fem_bands is not None:
                                for fem_kx, fem_freq in fem_bands:
                                    axD.plot(fem_kx, fem_freq,
                                             'r-', linewidth=0.8,
                                             alpha=0.85, zorder=5)

                        axD.set_xlim(0, np.pi)
                        axD.set_ylim(freqs_np_vis.min(), freqs_np_vis.max())
                        axD.tick_params(direction='in', which='both', top=False, right=False,
                                        labelbottom=False, labelleft=False)
                        axD.set_xticks([0, np.pi/2, np.pi])
                        axD.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
                        for sp in axD.spines.values(): sp.set_visible(True)
                        axD.yaxis.set_major_locator(MaxNLocator(nbins=5))
    
                        rank_label = "BEST" if rank == 0 else "WORST"
                        metric_label = f"mBOF (IoU): {b_mbof:.4f}" if scenario["type"] == "bandgap" else f"DMA (Tol): {b_tol:.4f}kHz, mBOF: {b_mbof:.4f}"
                        caption = f"Mat: {mat}   Cells: {nc}   {metric_label} (Rank {rank+1}: {rank_label})\nLengths: {[f'{x:.3f}' for x in len_np.tolist()]}"

                        fig.text(0.5, 0.02, caption, ha='center', va='bottom', fontsize=8, family='monospace')
    
                        fig.subplots_adjust(wspace=0.35, hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.25)
                        plt.savefig(f"{scenario_dir}/{mat}_{nc}cells_plot_rank{rank+1}.png", dpi=300)
                        plt.close(fig)

                    found_solution = True
                    break
                    
        if not found_solution:
            print(f"    ❌ [FAILED] No configuration met the threshold for {scenario['name']}.")

if __name__ == "__main__":
    cfg = CFG()
    run_inverse_design(cfg, mode="adaln-zero")
