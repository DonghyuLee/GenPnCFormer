import os
# 💡 [Fix] Expandable CUDA memory segments (PyTorch 2.1+): prevents heap fragmentation
#    during long inference loops (2600+ batches × 50 DDIM steps each).
#    MUST be set before `import torch` to take effect.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch

import numpy as np
import faulthandler
faulthandler.enable()

# Completely disable tqdm monitor
import tqdm
try:
    tqdm.monitor_interval = 0
except:
    pass
from config import CFG, device
from data_utils import load_or_build_cache_multimat
from vae import train_vae, build_vae_latent_cache, ConditionalVAE
from diffusion import train_latent_diffusion, DiffusionTransformer, UNet1D, DDPM
from eval import run_inference_and_evaluation, visualize_dispersion_comparison
from benchmark import measure_efficiency
import matplotlib.pyplot as plt
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GenPnCFormer Main Script")
    setup_group = parser.add_argument_group("Setup")
    parser.add_argument("--use_tmm", action="store_true", help="Use exact physics TMM solver instead of PnCFormer surrogate for evaluation")
    parser.add_argument("--test_only", action="store_true", help="Skip training and only run evaluation")
    args = parser.parse_args()

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

    if args.test_only:
        skip_vae_train = True

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
            final_p = p.replace("_dispersion.npz", "_vae_latent.npz")
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
    modes = ['adaln-zero']
    base_save_dir = cfg.save_dir

    for mode in modes:
        print(f"\n=========================================")
        print(f"       Starting Flow for Mode: {mode.upper()}")
        print(f"=========================================")
        
        # Override config for the current mode
        cfg.cond_mode = mode

        cfg.save_dir = f"{base_save_dir}_{mode}"
        os.makedirs(cfg.save_dir, exist_ok=True)
        
        # DDPM 학습
        out_name = f"ddpm_{getattr(cfg, 'diffusion_backbone', 'transformer')}_best.pt"
        ddpm_ckpt_path = os.path.join(cfg.save_dir, out_name)

        skip_ddpm_train = False
        if not skip_vae_train and os.path.exists(ddpm_ckpt_path):
            print(f"[DDPM {mode.upper()}] VAE was retrained. Renaming old DDPM checkpoint to add .bak")
            os.rename(ddpm_ckpt_path, ddpm_ckpt_path + ".bak")

        resume_ckpt_path = None
        if os.path.exists(ddpm_ckpt_path):
            print(f"[DDPM {mode.upper()}] Found existing checkpoint: {ddpm_ckpt_path}")
            try:
                backbone_type = getattr(cfg, "diffusion_backbone", "transformer")
                if backbone_type == "transformer":
                    _test_model = DiffusionTransformer(
                        latent_dim=cfg.latent_dim, width=cfg.transformer_width,
                        depth=cfg.transformer_depth, heads=cfg.transformer_heads,
                        dropout=cfg.dropout, cfg=cfg
                    )
                else:
                    _test_model = UNet1D(cfg.latent_dim, cfg.unet_width, cfg.unet_depth, cfg.dropout, cfg)
                
                _test_ddpm = DDPM(_test_model, cfg.timesteps, cfg.beta_start, cfg.beta_end)
                _test_state = torch.load(ddpm_ckpt_path, map_location='cpu', weights_only=True)
                
                if "ddpm" in _test_state:
                    _test_ddpm.load_state_dict(_test_state["ddpm"])
                else:
                    _test_ddpm.load_state_dict(_test_state)
                    
                print(f"[DDPM {mode.upper()}] Successfully loaded existing checkpoint (architecture matched).")
                
                # Interactive prompt for the user
                user_input = input(f"[DDPM {mode.upper()}] Do you want to skip training and proceed to inference? [Y/n]: ").strip().lower()
                if user_input == 'n':
                    skip_ddpm_train = False
                    resume_ckpt_path = ddpm_ckpt_path
                    print(f"[DDPM {mode.upper()}] Will resume training from {ddpm_ckpt_path}...")
                else:
                    skip_ddpm_train = True
                    resume_ckpt_path = None
            except Exception as e:
                print(f"\n[DDPM {mode.upper()}] !! Architecture Mismatch or Corrupted Checkpoint !!")
                print(f"Error details: {e}")
                print(f"Renaming old DDPM checkpoint to .bak and restarting training...")
                os.rename(ddpm_ckpt_path, ddpm_ckpt_path + ".bak")
                resume_ckpt_path = None

        if args.test_only:
            skip_ddpm_train = True

        if skip_ddpm_train:
            print(f"[DDPM {mode.upper()}] Skipping training...")
        else:
            if resume_ckpt_path:
                print(f"[DDPM {mode.upper()}] Resuming training...")
            else:
                print(f"[DDPM {mode.upper()}] Start NEW training...")
                
            vae.eval()
            tr_latent_paths = [p.replace("_dispersion.npz", "_vae_latent.npz") for p in train_paths]
            va_latent_paths = [p.replace("_dispersion.npz", "_vae_latent.npz") for p in valid_paths]
            
            ddpm_ckpt_path = train_latent_diffusion(
                cfg, device,
                vae_decoder=vae.decoder,
                train_paths=tr_latent_paths,
                valid_paths=va_latent_paths,
                resume_path=resume_ckpt_path
            )
            print(f"[DDPM {mode.upper()}] Training finished. checkpoint: {ddpm_ckpt_path}")

        # 5) 평가 
        run_inference_and_evaluation(
            cfg, device, min_width=0.0, w_cfg=5.0, ddim_steps=50, eta=0.0,
            test_paths=test_paths, diffusion_path=ddpm_ckpt_path, vae_path=vae_ckpt_path,
            use_tmm=args.use_tmm
        )

        # 6) 결과 시각화 
        # vis_dir = f"vis_results_{mode}"
        # if args.use_tmm: vis_dir += "_tmm"
        
        # visualize_dispersion_comparison(
        #     cfg, device, save_dir=vis_dir, w_cfg=5.0,
        #     diffusion_path=ddpm_ckpt_path, vae_path=vae_ckpt_path,
        #     use_tmm=args.use_tmm
        # )