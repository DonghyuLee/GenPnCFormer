
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import CFG
from surrogate.models.pncformer import PnCFormer
from surrogate.train_surrogate_regressor import SurrogateDataset
from data_utils import layers_to_six_features_torch

def load_surrogate_model(model_type, checkpoint_path, cfg, device):
    """
    Load a trained surrogate model from checkpoint.
    """
    model = PnCFormer(
        x_input_dim=6,
        f_input_dim=1,
        d_model=cfg.d_model,
        nhead=cfg.n_heads,
        num_encoder_layers=cfg.n_layers,
        num_decoder_layers=cfg.n_layers,
        dropout=cfg.dropout
    ).to(device)
    
    if not os.path.exists(checkpoint_path):
        print(f"[Error] Checkpoint not found: {checkpoint_path}")
        return None
        
    print(f"Loading {model_type} from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False) # weights_only=False for convenience with old saves
    
    # Check if checkpoint is full dict or just state_dict
    if "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    return model

def evaluate_and_visualize(model_type, target_key, cfg, device, num_vis=5):
    print(f"\n=== Evaluating {model_type} (Target: {target_key}) ===")
    
    # 1. Load Test Data
    if target_key in ["U", "D"]:
        suffix = "test_dispersion.npz"
    elif target_key == "T":
        suffix = "test_transmittance.npz"
    else:
        raise ValueError("Unknown target")
        
    target_folders = getattr(cfg, "target_folders", ["CA", "SA", "TA", "AC", "AS", "AT"])
    test_paths = [os.path.join(cfg.cache_dir, f"{f}_{suffix}") for f in target_folders]
    
    datasets = [SurrogateDataset(p, target_key) for p in test_paths if os.path.exists(p)]
    if not datasets:
        print("No test datasets found.")
        return
        
    test_ds = ConcatDataset(datasets)
    test_dl = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0)
    
    # 2. Load Model
    save_dir = getattr(cfg, "surrogate_model_dir", "./surrogate/best_model")
    ckpt_path = os.path.join(save_dir, f"Transformer_{model_type}_v1.4.pth")
    model = load_surrogate_model(model_type, ckpt_path, cfg, device)
    
    if model is None:
        return

    # 3. Quantitative Evaluation (MSE, MAE)
    criterion = nn.MSELoss()
    total_mse = 0
    total_mae = 0
    count = 0
    
    all_preds = []
    all_targets = []
    
    # For visualization selection
    vis_samples = []
    
    with torch.no_grad():
        for batch in tqdm(test_dl, desc=f"Evaluating {model_type}"):
            x = batch["x_seq"].to(device)
            f = batch["freqs"].to(device)
            y = batch["target"].to(device)
            n_cells = batch["n_cells"].to(device).clamp(min=1)
            
            # Feature Transform
            x = layers_to_six_features_torch(x)
            
            # Mask
            bool_mask = (torch.arange(x.size(1), device=device).unsqueeze(0) >= n_cells.unsqueeze(1))
            src_key_padding_mask = bool_mask # True=Pad
            
            # Forward
            pred = model(x, f, src_key_padding_mask=src_key_padding_mask)
            
            loss = criterion(pred, y)
            mae = torch.abs(pred - y).mean()
            
            curr_bs = x.size(0)
            total_mse += loss.item() * curr_bs
            total_mae += mae.item() * curr_bs
            count += curr_bs
            
            # Collect for visualization (first batch only usually, or random)
            if len(vis_samples) < num_vis:
                 # Take samples from this batch
                 needed = num_vis - len(vis_samples)
                 take = min(needed, curr_bs)
                 for i in range(take):
                     vis_samples.append({
                         "x": x[i].cpu(),
                         "f": f[i].cpu(),
                         "y": y[i].cpu(),
                         "pred": pred[i].cpu()
                     })

    avg_mse = total_mse / count
    avg_mae = total_mae / count
    print(f"[{model_type}] Test MSE: {avg_mse:.6f} | MAE: {avg_mae:.6f}")
    
    # 4. Visualization
    if not vis_samples:
        return

    fig, axes = plt.subplots(1, len(vis_samples), figsize=(4 * len(vis_samples), 4), constrained_layout=True)
    if len(vis_samples) == 1: axes = [axes]
    
    for i, sample in enumerate(vis_samples):
        ax = axes[i]
        freqs = sample["f"].numpy()
        gt = sample["y"].numpy()
        pred = sample["pred"].numpy()
        
        # Plot
        ax.plot(freqs, gt, 'k-', label='Ground Truth', linewidth=1.5, alpha=0.7)
        ax.plot(freqs, pred, 'r--', label='Prediction', linewidth=1.5)
        
        ax.set_title(f"Sample {i+1}")
        ax.set_xlabel("Frequency (k-point for U/D)")
        if i == 0:
            ax.set_ylabel(target_key)
            ax.legend()
            
    save_fig_path = os.path.join(save_dir, f"eval_{model_type}.png")
    plt.savefig(save_fig_path, dpi=300)
    print(f"Visualization saved to {save_fig_path}")
    plt.close()

def main():
    cfg = CFG()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Evaluate All 3 Models
    evaluate_and_visualize("UDR", "U", cfg, device)
    evaluate_and_visualize("SDR", "D", cfg, device)
    evaluate_and_visualize("TR", "T", cfg, device)
    
if __name__ == "__main__":
    main()
