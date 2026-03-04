import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

from config import CFG
from vae import VAE_Decoder
from diffusion import DDPM, DiffusionTransformer, UNet1D
from eval import SurrogateBandSolver, unscale_from_tanh, TestDataset
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


def check_success(pred_mask, scenario, freqs):
    target_mask = scenario["mask"]
    
    if scenario["type"] == "bandgap":
        intersection = np.logical_and(pred_mask == 1, target_mask == 1).sum()
        union = np.logical_or(pred_mask == 1, target_mask == 1).sum()
        if union == 0: return False
        iou = intersection / union
        return iou >= 0.70
    else:
        # Defect points existence within 0.5 kHz
        for f_tgt in scenario["targets"]:
            pred_defects = freqs[pred_mask == 2]
            if len(pred_defects) == 0:
                return False
            if np.min(np.abs(pred_defects - f_tgt)) > 1.0:
                # Target unfulfilled
                return False
        return True


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
    # All models use depth=4
    cfg.transformer_width = 128
    cfg.transformer_depth = 4
        
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
                # Also extract unnormalized materials for visualizer
                mat_A = ds[0]["material_A"].unsqueeze(0).to(device)
                mat_B = ds[0]["material_B"].unsqueeze(0).to(device)
            except: continue

            for nc in cell_counts:
                if found_solution: break
                
                nc_tensor = torch.tensor([nc], dtype=torch.long, device=device)
                print(f"  Attempting [{mat}] with {nc} cells...")
                
                for attempt in range(10):  # Retry 10 times
                    B = 5000
                    
                    with torch.no_grad():
                        z_sampled = ddpm.sample_ddim(
                            B=B,
                            band_mask=target_mask_tensor.expand(B, -1),
                            material_conds=mat_cond.expand(B, -1),
                            n_cells=nc_tensor.expand(B),
                            z_dim=cfg.latent_dim,
                            device=device,
                            w=5.0, # CFG
                            ddim_steps=50,
                            eta=0.0
                        )
                        z_sampled = z_sampled / getattr(cfg, "latent_scale_factor", 1.0)
                        
                        lengths_pred_scaled = vae_decoder(z_sampled, mat_cond.expand(B, -1), nc_tensor.expand(B))
                        lengths_pred_unscaled = unscale_from_tanh(lengths_pred_scaled)
                        
                        band_mask_pred = surrogate_solver(
                            lengths_pred_unscaled,
                            mat_cond.expand(B, -1),
                            nc_tensor.expand(B),
                            freqs_tensor.expand(B, -1)
                        ) # [B, 500]
                        
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
                        
                        # Get full Surrogate Predictions (Dispersion and Transmittance)
                        # Re-calculate to extract intermediate variables
                        freqs_input_reshaped = freqs_tensor.expand(1, -1).unsqueeze(-1)
                        
                        # Format UDR input
                        l_padded = torch.zeros((1, cfg.max_cells, 1), device=device)
                        l_padded[:, :nc, :] = best_lengths
                        lengths_padded = l_padded.squeeze(-1)

                        mat_A_exp = mat_A.expand(1, -1)
                        mat_B_exp = mat_B.expand(1, -1)
                        x_inputs = []
                        for i in range(cfg.max_cells):
                            if i % 2 == 0: x_inputs.append(torch.cat([lengths_padded[:, i:i+1], mat_A_exp], dim=-1))
                            else:          x_inputs.append(torch.cat([lengths_padded[:, i:i+1], mat_B_exp], dim=-1))
                        x_input = torch.stack(x_inputs, dim=1)
                        
                        udr_pred = udr_model(x_input, freqs_input_reshaped).squeeze(-1).cpu().numpy()[0]
                        tr_pred = tr_model(x_input, freqs_input_reshaped).squeeze(-1).cpu().numpy()[0]
                        len_np = best_lengths.cpu().numpy().squeeze()
                        
                        # Prepare mask for plotting
                        mask_np = pred_np[best_idx]
                        disp_colorized = np.where(mask_np == 0, udr_pred, np.nan)
                        
                        # Save Data
                        np.savez(f"{scenario_dir}/{mat}_{nc}cells_data.npz", 
                                 lengths=len_np, target_mask=scenario["mask"], 
                                 pred_mask=mask_np, dispersion=udr_pred, tr=tr_pred)
                                 
                        # Visualize Final Plot
                        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
                        
                        # Left: Target vs Predicted Band Mask
                        ax = axes[0]
                        ax.fill_between(freqs, 0, 1, where=(scenario["mask"]==1), color="gray", alpha=0.3, label="Target Bandgap")
                        for tgt in scenario["targets"]:
                            if scenario["type"] == "defect":
                                ax.axvline(tgt, color='blue', linestyle='--', label=f"Target Defect ({tgt}kHz)")
                        
                        ax.scatter(udr_pred, freqs, c=mask_np, cmap="coolwarm", s=5, label="Dispersion")
                        ax.set_ylabel("Frequency (kHz)")
                        ax.set_xlabel("Dispersion (rad)")
                        ax.set_title(f"Target vs Predicted Mask\n{mat} - {nc} Cells")
                        
                        # Right: Transmittance
                        ax2 = axes[1]
                        ax2.plot(tr_pred, freqs, color="purple", linewidth=1.5, label="Predicted TR")
                        ax2.fill_between(freqs, 0, 1, where=(scenario["mask"]==1), color="gray", alpha=0.3)
                        ax2.set_ylabel("Frequency (kHz)")
                        ax2.set_xlabel("Transmittance")
                        ax2.set_xlim(0, 1)
                        ax2.set_title("Surrogate Transmittance")
                        
                        plt.tight_layout()
                        plt.savefig(f"{scenario_dir}/{mat}_{nc}cells_plot.png")
                        plt.close()

                        found_solution = True
                        break
                    
        if not found_solution:
            print(f"    ❌ [FAILED] No configuration met the threshold for {scenario['name']}.")

if __name__ == "__main__":
    cfg = CFG()
    run_inverse_design(cfg, mode="adaln-zero")
