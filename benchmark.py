import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from config import CFG, device
from diffusion import DiffusionTransformer, UNet1D
try:
    from thop import profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False
    print("Warning: 'thop' not installed. FLOPs calculation will be skipped.")

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_efficiency(cfg, mode, device):
    # 1. Setup Model
    # Helper to enforce mode in cfg temporarily
    original_mode = getattr(cfg, "cond_mode", "adaln-zero")
    original_width = cfg.transformer_width
    original_depth = cfg.transformer_depth
    
    cfg.cond_mode = mode
    
    # ---------------------------------------------------------
    # 💡 Param Matching Logic
    # ---------------------------------------------------------
    # All models use parameters defined in config.py
    
    print(f"[{mode.upper()}] Instantiating Model (W={cfg.transformer_width}, D={cfg.transformer_depth})...")
    
    # Initialize Model for Measurement
    # Using Transformer backbone
    model = DiffusionTransformer(
        latent_dim=cfg.latent_dim,
        width=cfg.transformer_width,
        depth=cfg.transformer_depth,
        heads=cfg.transformer_heads,
        dropout=cfg.dropout,
        cfg=cfg
    ).to(device)
    model.eval()
    
    # 2. Dummy Inputs
    B = 1
    z = torch.randn(B, cfg.latent_dim).to(device)
    t = torch.randint(0, cfg.timesteps, (B,)).to(device)
    # Material: [B, 4]
    mat = torch.randn(B, 4).to(device)
    # N_cells: [B]
    n_cells = torch.randint(1, cfg.max_cells, (B,)).to(device)
    # Band Mask: [B, K=500]
    mask = torch.randint(0, cfg.n_classes, (B, cfg.k_points)).to(device)
    
    # 3. Parameters
    params = count_parameters(model)
    
    # 4. FLOPs (using thop)
    flops = 0.0
    if THOP_AVAILABLE:
        try:
            # thop.profile returns (flops, params)
            # inputs must be a tuple of arguments for forward
            inputs = (z, t, mask, mat, n_cells)
            flops, _ = profile(model, inputs=inputs, verbose=False)
        except Exception as e:
            print(f"[{mode.upper()}] FLOPs calc failed: {e}")
    
    # 5. Memory & Time
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    start_time = time.time()
    with torch.no_grad():
        # Warmup
        for _ in range(10):
            _ = model(z, t, mask, mat, n_cells)
        
        # Measurement
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100):
            _ = model(z, t, mask, mat, n_cells)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.time()
        
    avg_latency = (t1 - t0) / 100.0
    
    if torch.cuda.is_available():
        max_mem = torch.cuda.max_memory_allocated() / (1024 ** 2) # MB
    else:
        max_mem = 0.0 # CPU memory tracking is harder, skip for now
    
    print(f"[{mode.upper()}] Params: {params/1e6:.2f}M | FLOPs: {flops/1e9:.2f}G | Latency: {avg_latency*1000:.2f}ms | Mem: {max_mem:.2f}MB")
    
    # Restore config
    cfg.cond_mode = original_mode
    cfg.transformer_width = original_width
    cfg.transformer_depth = original_depth
    
    return {
        "mode": mode,
        "params": params,
        "flops": flops,
        "latency": avg_latency,
        "memory": max_mem
    }

from eval import run_inference_and_evaluation

def measure_quality(cfg, mode, device):
    print(f"\n--- Measuring Quality for {mode.upper()} ---")
    original_mode = cfg.cond_mode
    original_save_dir = cfg.save_dir
    original_width = cfg.transformer_width
    original_depth = cfg.transformer_depth
    
    cfg.cond_mode = mode
    
    # ---------------------------------------------------------
    # 💡 Param Matching Logic (Must match train_comparison.py)
    # ---------------------------------------------------------
    print(f"[{mode.upper()}] Evaluation Config: W={cfg.transformer_width}, D={cfg.transformer_depth}")

    # Set save_dir logic matches train_comparison.py
    base_save_dir = "./checkpoints/v2.1.0" # Hardcoded base for now, should match config default
    # Only output TSNE for unconditional / standard variants if needed
    # (Previously skipping for 'hybrid', now can just run or skip conditionally if desired. Running by default)
    cfg.save_dir = f"{base_save_dir}_{mode}"
        
    print(f"[{mode.upper()}] Loading from: {cfg.save_dir}")
    
    ckpt_path = os.path.join(cfg.save_dir, "ddpm_best.pt")
    if not os.path.exists(ckpt_path):
        print(f"Skipping Quality Eval for {mode}: Checkpoint not found at {ckpt_path}")
        cfg.cond_mode = original_mode
        cfg.save_dir = original_save_dir
        return None
        
    # Run Eval
    # We use a small subset or full test set? 
    # run_inference_and_evaluation uses cfg.test_paths or default cache.
    # It prints metrics. We might want to capture them if possible, but run_inference_and_evaluation doesn't return them as dict yet.
    # It prints them. We can just let it print for now.
    
    try:
        run_inference_and_evaluation(cfg, device, min_width=0.0, w_cfg=5.0, ddim_steps=50)
    except Exception as e:
        import traceback
        print(f"Quality Eval Failed for {mode}: {e}")
        traceback.print_exc()
        
    cfg.cond_mode = original_mode
    cfg.save_dir = original_save_dir
    # Restore dimensions
    cfg.transformer_width = original_width
    cfg.transformer_depth = original_depth

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="benchmark_results.png")
    parser.add_argument("--skip_quality", action="store_true", help="Skip generation quality evaluation")
    args = parser.parse_args()
    
    cfg = CFG()
    modes = ["adaln", "adaln-zero", "mhca"]
    # modes = ["adaln", "mhca"]
    # modes = ["mhca"]

    results = []
    
    print("--- Starting Benchmark (Efficiency) ---")
    for m in modes:
        res = measure_efficiency(cfg, m, device)
        results.append(res)
        
    # Visualization
    modes_labels = [r["mode"].upper() for r in results]
    params = [r["params"]/1e6 for r in results]
    flops = [r["flops"]/1e9 for r in results] # GFLOPs
    latency = [r["latency"]*1000 for r in results]
    memory = [r["memory"] for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Params
    ax = axes[0,0]
    bars = ax.bar(modes_labels, params, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    ax.set_title("Parameters (M)")
    ax.bar_label(bars, fmt='%.2f')
    
    # 2. FLOPs
    ax = axes[0,1]
    if any(f > 0 for f in flops):
        bars = ax.bar(modes_labels, flops, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
        ax.set_title("FLOPs (G) per Step")
        ax.bar_label(bars, fmt='%.2f')
    else:
        ax.text(0.5, 0.5, "FLOPs N/A", ha='center', va='center')
    
    # 3. Latency
    ax = axes[1,0]
    bars = ax.bar(modes_labels, latency, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    ax.set_title("Inference Latency (ms)")
    ax.bar_label(bars, fmt='%.1f')
    
    # 4. Memory
    ax = axes[1,1]
    bars = ax.bar(modes_labels, memory, color=['tab:blue', 'tab:orange', 'tab:green', 'tab:red'])
    ax.set_title("Peak Memory (MB)")
    ax.bar_label(bars, fmt='%.1f')
    
    plt.tight_layout()
    plt.savefig(args.save_path)
    print(f"Benchmark efficiency plot saved to {args.save_path}")
    
    if not args.skip_quality:
        print("\n--- Starting Benchmark (Quality) ---")
        for m in modes:
            measure_quality(cfg, m, device)

if __name__ == "__main__":
    main()
