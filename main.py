import os
import torch
import numpy as np
from config import CFG, device
from data_utils import load_or_build_cache_multimat
from vae import train_vae, build_vae_latent_cache, ConditionalVAE
from diffusion import train_latent_diffusion
from eval import run_inference_and_evaluation, visualize_dispersion_comparison, compare_condition_vs_surrogate_truth
from benchmark import measure_efficiency
import matplotlib.pyplot as plt

if __name__ == "__main__":
    cfg = CFG()
    os.makedirs(cfg.save_dir, exist_ok=True)

    # 1. 데이터 캐시 준비 (Returns lists of paths)
    train_paths, valid_paths, test_paths = load_or_build_cache_multimat(cfg)
    
    # 2) VAE 학습 및 로딩 (Architecture Mismatch 처리)
    vae_ckpt_path = os.path.join(cfg.save_dir, "vae_model_best.pt")
    vae = ConditionalVAE(cfg).to(device)
    
    # Checkpoint 로딩 시도 (Architecture Mismatch 체크)
    if os.path.exists(vae_ckpt_path):
        print(f"[VAE] Found existing checkpoint: {vae_ckpt_path}")
        try:
            vae_state = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
            vae.load_state_dict(vae_state)
            print("[VAE] Successfully loaded existing checkpoint.")
            skip_vae_train = True
        except RuntimeError as e:
            print(f"\n[VAE] !! Architecture Mismatch or Corrupted Checkpoint !!")
            print(f"Error details: {e}")
            print(f"[VAE] The existing checkpoint is incompatible with the current code.")
            print(f"[VAE] Removing old checkpoint and restarting training...")
            os.remove(vae_ckpt_path)
            skip_vae_train = False
    else:
        skip_vae_train = False

    if not skip_vae_train:
        print("[VAE] No valid checkpoint found. Start training...")
        vae_ckpt_path = train_vae(cfg, device, train_paths=train_paths, valid_paths=valid_paths)
        print(f"[VAE] Training finished. checkpoint: {vae_ckpt_path}")
        # Reload best model
        vae_state = torch.load(vae_ckpt_path, map_location=device, weights_only=True)
        vae.load_state_dict(vae_state)

    # 3) VAE latent 캐시 생성
    for split_name, path_list in [("train", train_paths), ("valid", valid_paths), ("test", test_paths)]:
        print(f"[LatentCache] Checking/Building latent cache for '{split_name}'...")
        # build_vae_latent_cache now accepts a list of paths
        success = build_vae_latent_cache(split_name, cfg, vae, device, path_list)
        if success:
            print(f"[LatentCache] Processed latent cache for '{split_name}'.")
        else:
            print(f"[LatentCache] Failed/Skipped latent cache for '{split_name}'.")

    print("[VAE] Latent cache ready.")
    
    # 💡 [Data Reporting] Print dataset sizes
    print("\n--- Dataset Summary ---")
    for split_name, path_list in [("train", train_paths), ("valid", valid_paths), ("test", test_paths)]:
        total_samples = 0
        for p in path_list:
            final_p = p.replace("_with_mask.npz", "_vae_latent.npz")
            try:
                if os.path.exists(final_p):
                    # Just peek at Z size
                    # To be faster, we might rely on naming or trust build process.
                    # But verifying size is good.
                    data = np.load(final_p)
                    total_samples += data['Z'].shape[0]
            except:
                pass
        print(f"[{split_name.upper()}] Samples: {total_samples}")
    print("-----------------------\n")

    # 4) DDPM 학습 및 평가 파이프라인 (Multi-Mode 지원)
    modes = ["film", "mhca", "hybrid"]
    base_save_dir = cfg.save_dir

    for mode in modes:
        print(f"\n=========================================")
        print(f"       Starting Flow for Mode: {mode.upper()}")
        print(f"=========================================")
        
        # Override config for the current mode
        cfg.cond_mode = mode
        if mode == "hybrid":
            cfg.transformer_width = 128
            cfg.transformer_depth = 4
        elif mode == "film":
            cfg.transformer_width = 128
            cfg.transformer_depth = 5 
        elif mode == "mhca":
            cfg.transformer_width = 128
            cfg.transformer_depth = 6 

        cfg.save_dir = f"{base_save_dir}_{mode}"
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        # DDPM 학습
        out_name = f"ddpm_{getattr(cfg, 'diffusion_backbone', 'transformer')}_best.pt"
        ddpm_ckpt_path = os.path.join(cfg.save_dir, out_name)

        if not skip_vae_train and os.path.exists(ddpm_ckpt_path):
            print(f"[DDPM {mode.upper()}] VAE was retrained. Removing old DDPM checkpoint: {ddpm_ckpt_path}")
            os.remove(ddpm_ckpt_path)

        if os.path.exists(ddpm_ckpt_path):
            print(f"[DDPM {mode.upper()}] Found existing checkpoint: {ddpm_ckpt_path}")
            print(f"[DDPM {mode.upper()}] Skipping training...")
        else:
            print(f"[DDPM {mode.upper()}] Start training...")
            vae.eval()
            tr_latent_paths = [p.replace("_dispersion.npz", "_vae_latent.npz") for p in train_paths]
            va_latent_paths = [p.replace("_dispersion.npz", "_vae_latent.npz") for p in valid_paths]
            
            ddpm_ckpt_path = train_latent_diffusion(
                cfg, device,
                vae_decoder=vae.decoder,
                train_paths=tr_latent_paths,
                valid_paths=va_latent_paths,
                resume_path=None
            )
            print(f"[DDPM {mode.upper()}] Training finished. checkpoint: {ddpm_ckpt_path}")

        # 5) 평가
        run_inference_and_evaluation(
            cfg, device, min_width=0.0, w_cfg=5.0, ddim_steps=50, eta=0.0,
            test_paths=test_paths, diffusion_path=ddpm_ckpt_path, vae_path=vae_ckpt_path
        )

        # 6) 결과 시각화 (hybrid만 시각화)
        if mode == "hybrid":
            vis_dir = f"vis_results_{mode}"
            visualize_dispersion_comparison(cfg, device, save_dir=vis_dir, w_cfg=5.0)

    # 7) 전체 모드 Benchmarking (효율성 통합 비교)

    print("\n--- Starting Benchmark (Efficiency) ---")
    results = []
    for m in modes:
        res = measure_efficiency(cfg, m, device)
        if res: results.append(res)
        
    if results:
        modes_labels = [r["mode"].upper() for r in results]
        params = [r["params"]/1e6 for r in results]
        flops = [r["flops"]/1e9 for r in results]
        latency = [r["latency"]*1000 for r in results]
        memory = [r["memory"] for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0,0].bar(modes_labels, params, color=['tab:blue', 'tab:orange', 'tab:green'])
        axes[0,0].set_title("Parameters (M)")
        axes[0,0].bar_label(axes[0,0].containers[0], fmt='%.2f')
        
        if any(f > 0 for f in flops):
            axes[0,1].bar(modes_labels, flops, color=['tab:blue', 'tab:orange', 'tab:green'])
            axes[0,1].set_title("FLOPs (G) per Step")
            axes[0,1].bar_label(axes[0,1].containers[0], fmt='%.2f')
        else:
            axes[0,1].text(0.5, 0.5, "FLOPs N/A", ha='center', va='center')
        
        axes[1,0].bar(modes_labels, latency, color=['tab:blue', 'tab:orange', 'tab:green'])
        axes[1,0].set_title("Inference Latency (ms)")
        axes[1,0].bar_label(axes[1,0].containers[0], fmt='%.1f')
        
        axes[1,1].bar(modes_labels, memory, color=['tab:blue', 'tab:orange', 'tab:green'])
        axes[1,1].set_title("Peak Memory (MB)")
        axes[1,1].bar_label(axes[1,1].containers[0], fmt='%.1f')
        
        plt.tight_layout()
        plt.savefig("benchmark_results.png")
        print(f"Benchmark efficiency plot saved to benchmark_results.png")