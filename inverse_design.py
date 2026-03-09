import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from tqdm.auto import tqdm

from config import CFG
from vae import VAE_Decoder
from diffusion import DDPM, DiffusionTransformer, UNet1D
from eval import SurrogateBandSolver, unscale_from_tanh, TestDataset, \
    _mask_to_intervals_1d, _greedy_match_intervals, _match_intervals_tolerance
from data_utils import visualize_sample_paper
from surrogate.models.pncformer import PnCFormer

def build_scenario_masks(k_points=500, f_start=100.0, f_step=100.0):
    """
    freq_grid = np.linspace(0.1, 50.0, 500)
    idx = int((f - 0.1)/0.1)
    """
    def freq_to_idx(freq_khz):
        return max(0, min(k_points - 1, int(round((freq_khz - 0.1) / 0.1))))

    scenarios = []

    # 1) Bandgap matching & widening
    for f1, f2 in [(25, 30), (25, 35), (25, 40)]:
        mask = np.full(k_points, 3, dtype=np.int32)
        idx1, idx2 = freq_to_idx(f1), freq_to_idx(f2)
        
        # 0-padding
        pad_start = max(0, idx1 - 20)
        pad_end = min(k_points, idx2 + 20)
        mask[pad_start:idx1] = 0
        mask[idx2:pad_end] = 0
        
        mask[idx1:idx2] = 1
        scenarios.append({
            "name": f"1_Bandgap_{f1}_{f2}kHz", 
            "mask": mask, "type": "bandgap", "targets": [(f1, f2)]
        })

    # 2) Single defect-band matching
    for f in [30, 35, 40]:
        mask = np.full(k_points, 3, dtype=np.int32)
        # Pad bandgap around defect
        idx1, idx2 = freq_to_idx(f - 5), freq_to_idx(f + 5)
        
        # 0-padding
        pad_start = max(0, idx1 - 20)
        pad_end = min(k_points, idx2 + 20)
        mask[pad_start:idx1] = 0
        mask[idx2:pad_end] = 0
        
        mask[idx1:idx2] = 1
        d1, d2 = freq_to_idx(f - 0.5), freq_to_idx(f + 0.5)
        mask[d1:d2] = 2
        scenarios.append({
            "name": f"2_SingleDefect_{f}kHz", 
            "mask": mask, "type": "defect", "targets": [f]
        })

    # 3) Double defect-band matching
    for f1, f2 in [(25, 35), (26, 34), (27, 33)]:
        mask = np.full(k_points, 3, dtype=np.int32)
        # Pad bandgap around defects
        idx1, idx2 = freq_to_idx(f1 - 4), freq_to_idx(f2 + 4)
        
        # 0-padding
        pad_start = max(0, idx1 - 20)
        pad_end = min(k_points, idx2 + 20)
        mask[pad_start:idx1] = 0
        mask[idx2:pad_end] = 0
        mask[idx1:idx2] = 1
        
        d1_s, d1_e = freq_to_idx(f1 - 0.5), freq_to_idx(f1 + 0.5)
        d2_s, d2_e = freq_to_idx(f2 - 0.5), freq_to_idx(f2 + 0.5)
        mask[d1_s:d1_e] = 2
        mask[d2_s:d2_e] = 2
        scenarios.append({
            "name": f"3_DoubleDefect_{f1}_{f2}kHz", 
            "mask": mask, "type": "defect", "targets": [f1, f2]
        })

    return scenarios


def check_success(pred_mask: np.ndarray, scenario: dict, freqs: np.ndarray,
                  iou_thr: float = 0.50, tol_kHz: float = 1.0) -> bool:
    """
    Interval-based success check (consistent with eval.py metrics).
    - Bandgap : greedy IoU matching between predicted and target intervals, threshold iou_thr
    - Defect  : greedy center-tolerance matching, tolerance tol_kHz
    Don't Care (class 3) regions are zeroed out before interval extraction.
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
            return False
        tp, fp, fn, matched_ious = _greedy_match_intervals(pis, gis, iou_thr=iou_thr)
        # Success: all GT intervals matched (recall = 1, no FN)
        return fn == 0 and tp > 0
    else:
        # Defect: center-tolerance matching
        target_val = 2
        pis = _mask_to_intervals_1d(freqs, pred_clean, target_val)
        gis = _mask_to_intervals_1d(freqs, target_mask, target_val)
        if len(gis) == 0:
            return False
        tp, fp, fn, _ = _match_intervals_tolerance(pis, gis, tol=tol_kHz)
        return fn == 0 and tp > 0


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

    # 3. Load Surrogate Solver
    surrogate_solver = SurrogateBandSolver(cfg, device)
    base_dir = getattr(cfg, "surrogate_model_dir", "./surrogate/best_model")
    udr_model = PnCFormer(
        x_input_dim=6, f_input_dim=1,
        d_model=128, nhead=4,
        num_encoder_layers=4, num_decoder_layers=4, dropout=0.0
    ).to(device)
    udr_model.load_state_dict(torch.load(os.path.join(base_dir, "Transformer_UDR_v1.4.pth"), map_location=device, weights_only=True))
    udr_model.eval()

    tr_model = PnCFormer(
        x_input_dim=6, f_input_dim=1,
        d_model=128, nhead=4,
        num_encoder_layers=4, num_decoder_layers=4, dropout=0.0
    ).to(device)
    tr_model.load_state_dict(torch.load(os.path.join(base_dir, "Transformer_TR_v1.4.pth"), map_location=device, weights_only=True))
    tr_model.eval()
    
    # Disable nested tensor to prevent Segfault in long loops
    for m in [udr_model, tr_model]:
        if hasattr(m, 'encoder'):
            m.encoder.enable_nested_tensor = False

    scenarios = build_scenario_masks()
    materials = ["CA", "SA", "TA", "AC", "AS", "AT"]
    cell_counts = [8, 10, 12, 14]
    
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

        for mat in materials:
            if found_solution: break
            
            # Obtain precise material conditions from cache test dataset to ensure consistency
            cache_path = os.path.join(cfg.cache_dir, f"{mat}_test_dispersion.npz")
            if not os.path.exists(cache_path): continue
            ds = TestDataset(cache_path, cfg.max_cells)
            try:
                mat_cond = ds[0]["material_conds"].unsqueeze(0).to(device) # [1, 4]
            except Exception as e:
                print(f"  [WARN] Failed to load material conds for {mat}: {e}")
                continue

            for nc in cell_counts:
                if found_solution: break
                
                nc_tensor = torch.tensor([nc], dtype=torch.long, device=device)
                print(f"  Attempting [{mat}] with {nc} cells...")
                
                for attempt in range(10):  # Retry 10 times
                    B = 5000
                    
                    with torch.no_grad():
                        # Use mini-batch to avoid VRAM OOM with B=5000
                        CHUNK = 500
                        all_lengths = []
                        all_masks   = []
                        
                        for c_start in range(0, B, CHUNK):
                            c_end = min(c_start + CHUNK, B)
                            c_size = c_end - c_start
                            
                            with torch.amp.autocast('cuda', enabled=torch.cuda.is_available()):
                                z_c = ddpm.sample_ddim(
                                    B=c_size,
                                    band_mask=target_mask_tensor.expand(c_size, -1),
                                    material_conds=mat_cond.expand(c_size, -1),
                                    n_cells=nc_tensor.expand(c_size),
                                    z_dim=cfg.latent_dim,
                                    device=device,
                                    w=5.0,
                                    ddim_steps=50,
                                    eta=0.0
                                )
                                z_c = z_c / getattr(cfg, "latent_scale_factor", 1.0)
                                lengths_c_scaled = vae_decoder(z_c, mat_cond.expand(c_size, -1), nc_tensor.expand(c_size))
                                lengths_c = unscale_from_tanh(lengths_c_scaled)
                            
                            # Surrogate in fp32 (outside AMP)
                            mask_c = surrogate_solver(
                                lengths_c.float().detach(),
                                mat_cond.expand(c_size, -1).float(),
                                nc_tensor.expand(c_size),
                                freqs_tensor.expand(c_size, -1)
                            ) # [c_size, 500] on CPU
                            
                            all_lengths.append(lengths_c.detach().cpu())
                            all_masks.append(mask_c)
                            
                            del z_c, lengths_c_scaled, lengths_c, mask_c
                            torch.cuda.synchronize()
                            torch.cuda.empty_cache()
                        
                        lengths_pred_unscaled = torch.cat(all_lengths, dim=0).to(device)
                        band_mask_pred = torch.cat(all_masks, dim=0)  # CPU
                        
                        # Evaluate constraints
                        pred_np = band_mask_pred.cpu().numpy()
                        success_indices = []
                        for i in range(B):
                            if check_success(pred_np[i], scenario, freqs):
                                success_indices.append(i)
                                
                    if len(success_indices) > 0:
                        best_idx = success_indices[0]
                        print(f"    ✅ [SUCCESS] Found {len(success_indices)} matching structural designs on retry {attempt+1}!")
                        
                        # Fetch best features for Visualization
                        best_lengths = lengths_pred_unscaled[best_idx:best_idx+1]
                        
                        # Get UDR/TR via surrogate_solver's internal models for best design
                        freqs_input = freqs_tensor.expand(1, -1)  # [1, 500]
                        nc_best = torch.tensor([nc], dtype=torch.long, device=device)
                        with torch.no_grad():
                            udr_pred_t = surrogate_solver.model_udr(
                                surrogate_solver._build_layers_and_features(best_lengths.unsqueeze(-1) if best_lengths.dim()==2 else best_lengths, mat_cond, nc_best)[0].to(device),
                                freqs_input
                            ).cpu().numpy()[0]  # [K]
                            tr_pred_t = surrogate_solver.model_trans(
                                surrogate_solver._build_layers_and_features(best_lengths.unsqueeze(-1) if best_lengths.dim()==2 else best_lengths, mat_cond, nc_best)[0].to(device),
                                freqs_input
                            ).cpu().numpy()[0]
                        udr_pred = udr_pred_t
                        tr_pred  = tr_pred_t
                        len_np = best_lengths.cpu().numpy().squeeze()
                        
                        # Prepare mask for plotting
                        mask_np = pred_np[best_idx]
                        mask_np[scenario["mask"] == 3] = 3 # Apply don't care to predicted mask like input mask
                        disp_colorized = np.where(mask_np == 0, udr_pred, np.nan)
                        
                        # Save Data
                        np.savez(f"{scenario_dir}/{mat}_{nc}cells_data.npz", 
                                 lengths=len_np, target_mask=scenario["mask"], 
                                 pred_mask=mask_np, dispersion=udr_pred, tr=tr_pred)
                                 
                        # ─── Visualization (3-panel, eval.py style) ───────
                        from matplotlib.patches import Rectangle
                        from matplotlib.ticker import MaxNLocator

                        freqs_np_vis = freqs  # already [500] numpy

                        # SDR + UDR for visualization (correct dims + padding_mask)
                        bl_3d = best_lengths.unsqueeze(-1) if best_lengths.dim() == 2 else best_lengths  # [1,L,1]
                        X_feat_vis, valid_mask_vis = surrogate_solver._build_layers_and_features(
                            bl_3d.to(device).float(), mat_cond.float(), nc_best)
                        padding_mask_vis = ~valid_mask_vis  # True=pad
                        f_vis = freqs_input  # [1, 500]
                        with torch.no_grad():
                            sdr_pred_t = surrogate_solver.model_sdr(
                                X_feat_vis, f_vis, src_key_padding_mask=padding_mask_vis
                            ).cpu().numpy()[0]

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
                        axLen.plot(idx_gen, L_gen_plot, color='b', linestyle='-', marker='o', markersize=4)
                        axLen.set_xlim(0, 15); axLen.set_xticks([0, 5, 10, 15])
                        axLen.set_ylim(0.0, 0.10)
                        axLen.tick_params(direction='in', which='both', top=False, right=False,
                                          labelbottom=False, labelleft=False)

                        # ── Panel 2: Band Mask (target top, predicted bottom) ──
                        COLOR = {0: 'white', 1: (1.0, 0.8, 0.6), 2: None, 3: 'lightgray'}

                        def _draw_mask_strip(ax, mask_vals, y_bottom, height=0.9):
                            for kk in range(K):
                                c = int(mask_vals[kk])
                                if c == 2: continue
                                col = COLOR.get(c, 'white')
                                w = edges[kk+1] - edges[kk]
                                ax.add_patch(Rectangle((edges[kk], y_bottom), w, height,
                                                       facecolor=col, edgecolor='none', zorder=1))
                            for kk in range(K):
                                if int(mask_vals[kk]) == 2:
                                    xc = (edges[kk] + edges[kk+1]) / 2.0
                                    ax.plot([xc, xc], [y_bottom, y_bottom+height],
                                            color=(0,0.5,0), linewidth=1.5, zorder=3)
                            ax.plot([edges[0], edges[-1], edges[-1], edges[0], edges[0]],
                                    [y_bottom, y_bottom, y_bottom+height, y_bottom+height, y_bottom],
                                    color='k', linewidth=1.0)

                        _draw_mask_strip(axM, mask_np,             0.0, 0.9)  # bottom: predicted
                        _draw_mask_strip(axM, scenario["mask"], 1.0, 0.9)  # top: target
                        axM.set_xlim(edges[0], edges[-1])
                        axM.set_ylim(0.0, 1.9)
                        axM.set_yticks([])
                        axM.tick_params(axis='x', direction='in', top=False, labelbottom=False)
                        axM.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
                        for sp in axM.spines.values(): sp.set_visible(True)

                        # ── Panel 3: SDR Dispersion Relation (predicted only) ──
                        for kk in range(K):
                            if scenario["mask"][kk] == 1:
                                axD.axhspan(edges[kk], edges[kk+1], color=(1.0, 0.8, 0.6), alpha=0.5, lw=0, zorder=-1)
                        axD.plot(sdr_pred_snap, freqs_np_vis, color='b', linestyle='-', linewidth=1.5)
                        axD.set_xlim(0, np.pi)
                        axD.set_ylim(freqs_np_vis.min(), freqs_np_vis.max())
                        axD.tick_params(direction='in', which='both', top=False, right=False,
                                        labelbottom=False, labelleft=False)
                        axD.set_xticks([0, np.pi/2, np.pi])
                        for sp in axD.spines.values(): sp.set_visible(True)
                        axD.yaxis.set_major_locator(MaxNLocator(nbins=5))

                        # Bottom caption
                        L_str = '[' + ', '.join(f'{x:.3f}' for x in L_gen_plot) + ']'
                        fig.text(0.5, 0.02, f"Mat: {mat}  Cells: {nc}  Lengths: {L_str}",
                                 ha='center', va='bottom', fontsize=8, family='monospace')
                        fig.subplots_adjust(wspace=0.35, hspace=0, left=0.05, right=0.95, top=0.95, bottom=0.20)
                        fig.savefig(f"{scenario_dir}/{mat}_{nc}cells_plot.png", dpi=300)
                        plt.close(fig)

                        found_solution = True
                        break
                    
        if not found_solution:
            print(f"    ❌ [FAILED] No configuration met the threshold for {scenario['name']}.")

if __name__ == "__main__":
    cfg = CFG()
    run_inverse_design(cfg, mode="adaln-zero")
