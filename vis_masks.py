import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from config import CFG
from vae import VAEDataset
from data_utils import load_or_build_cache_multimat

def visualize_samples(num_samples=5, save_path="mask_samples.png"):
    cfg = CFG()
    print("Loading Dataset Paths...")
    train_paths, _, _ = load_or_build_cache_multimat(cfg)
    
    target_path = train_paths[0]
    
    print(f"Loading VAEDataset from {target_path}...")
    dataset = VAEDataset(target_path, cfg.max_cells)
    
    # We want to find samples that actually have a bandgap (class 1)
    # to show off the padding and don't care tokens.
    selected_indices = []
    
    np.random.seed(42) # For reproducibility if needed
    indices = np.random.permutation(len(dataset))
    
    for idx in indices:
        if len(selected_indices) >= num_samples:
            break
            
        data = dataset[idx]
        mask = data["band_mask"]
        mask_np = mask.numpy()
        
        # Check if there is a bandgap (class 1) or defect (class 2)
        if 1 in mask_np or 2 in mask_np:
            selected_indices.append(idx)
            
    if len(selected_indices) < num_samples:
        print("Warning: Not enough interesting samples found.")
        
    fig, axes = plt.subplots(len(selected_indices), 1, figsize=(10, 2.5 * len(selected_indices)))
    if len(selected_indices) == 1:
        axes = [axes]
        
    freqs = np.linspace(0, 50, 500)
    
    for i, idx in enumerate(selected_indices):
        data = dataset[idx]
        n_cells = data["N_cells"].item()
        attrs = data["material_conds"]
        freqs_dummy = data["freqs"]
        mask = data["band_mask"]
        mask_np = mask.numpy()
        
        ax = axes[i]
        
        # Color mapping for different classes
        colors = {0: 'tab:blue', 1: 'tab:red', 2: 'tab:green', 3: 'lightgray'}
        
        # Use step plot for the mask
        ax.step(freqs, mask_np, where='mid', color='black', alpha=0.5, linewidth=1)
        
        # Fill regions corresponding to classes
        for cls_val in range(4):
            ax.fill_between(freqs, -0.5, 3.5, where=(mask_np == cls_val), 
                            color=colors[cls_val], alpha=0.3, step='mid', 
                            label=f'Class {cls_val}' if i == 0 else "")
            
        ax.set_yticks([0, 1, 2, 3])
        ax.set_yticklabels(['Pass (0)', 'Bandgap (1)', 'Defect (2)', "Don't Care (3)"])
        ax.set_ylim(-0.5, 3.5)
        ax.set_xlim(0, 50)
        
        ax.set_title(f"Sample Index {idx} | N_cells: {n_cells} | Mat: {attrs[:2].numpy().round(2)}")
        ax.set_xlabel("Frequency (kHz)")
        ax.grid(axis='x', linestyle='--', alpha=0.5)
        
        if i == 0:
            ax.legend(loc='upper right', fontsize='small')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"Visualization saved to {save_path}")

if __name__ == "__main__":
    visualize_samples()
