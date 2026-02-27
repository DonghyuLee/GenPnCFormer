from __future__ import annotations
import argparse
from typing import Iterable, List
import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from surrogate.models.pncformer import PnCFormer
from surrogate.data_loader.dataset import (
    scan_and_load, split_datasets, PnCDataset
)

@torch.no_grad()
def visualize_test_results_by_indices(model: torch.nn.Module,
                                      dataset: PnCDataset,
                                      indices: Iterable[int],
                                      *,
                                      clamp_pi: bool | None = None,
                                      save_dir: str | None = None,
                                      title_prefix: str = ""):
    """
    Plot predictions vs ground-truth for given sample indices from a dataset.

    Args:
      model: Trained PnCFormer.
      dataset: PnCDataset returning (x, f, y).
      indices: Sample indices to visualize.
      clamp_pi: If True, clamp outputs to [0, pi]. If None, auto-enable for dispersion targets.
      save_dir: If provided, save plots as PNG files under this directory.
      title_prefix: Title prefix for figures.
    """
    device = next(model.parameters()).device
    model.eval()

    # Auto-detect clamp for dispersion-like targets if not specified
    if clamp_pi is None:
        clamp_pi = True  # safe default for dispersion; harmless for others

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    for idx in indices:
        x, f, y = dataset[idx]       # x: (L,6), f: (F,), y: (F,)
        x = x.unsqueeze(0).to(device)
        f = f.unsqueeze(0).to(device)
        attn_mask = torch.ones(x.size(0), x.size(1), dtype=torch.bool, device=device)

        pred = model(x, f, src_key_padding_mask=attn_mask)[0].detach().cpu().numpy()
        true = y.numpy()
        freq_khz = f[0].detach().cpu().numpy()

        if clamp_pi:
            pred = np.clip(pred, 0.0, np.pi)
            true = np.clip(true, 0.0, np.pi)

        print("[layers length(m)] =", x[0, :, 2].detach().cpu().numpy())

        plt.figure(figsize=(7.5, 5))
        # x-axis: wavenumber-like value; y-axis: frequency (kHz)
        plt.plot(true, freq_khz, label="Ground Truth", linewidth=2.0)
        plt.plot(pred, freq_khz, label="Prediction", linestyle="--", linewidth=2.0)
        plt.xlabel("Response")
        plt.ylabel("Frequency [kHz]")
        plt.grid(True)
        title = f"{title_prefix} sample #{idx}"
        plt.title(title)
        plt.legend()

        if save_dir:
            out = os.path.join(save_dir, f"viz_{idx:06d}.png")
            plt.savefig(out, bbox_inches="tight", dpi=150)
            print(f"  saved: {out}")
            plt.close()
        else:
            plt.show()


def _parse_indices(spec: str, n: int) -> List[int]:
    """
    Parse indices specification:
      - "10"          → [10]
      - "0,5,12"      → [0,5,12]
      - "100:110"     → [100..109]
      - "rand10"      → 10 random unique indices in [0, n)
    """
    spec = spec.strip()
    if spec.startswith("rand"):
        k = int(spec[4:])
        rng = np.random.default_rng(42)
        return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    if ":" in spec:
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = int(b) if b else n
        return list(range(max(0, a), min(b, n)))
    if "," in spec:
        return [int(s) for s in spec.split(",") if s.strip() != ""]
    return [int(spec)]


def main():
    ap = argparse.ArgumentParser(description="Visualize PnCFormer predictions vs ground-truth.")
    ap.add_argument("--ckpt", type=str, required=True, help="Checkpoint path (e.g., surrogate/ckpt/PnCFormer_DR_v1.pth)")
    ap.add_argument("--data_dir", type=str, default="data", help="Root data directory")
    ap.add_argument("--pairs", type=str, default="CA", help='Comma-separated subfolders, e.g. "TA,CA,SA"')
    ap.add_argument("--sample_size", type=int, default=20000, help="Per-file sample cap")
    ap.add_argument("--freq_size", type=int, default=500, help="Frequency points cap")
    ap.add_argument("--target", type=str, default="dispersion_relation_supercell",
                    choices=["dispersion_relation_supercell", "dispersion_relation_unitcell", "transmittance"])
    ap.add_argument("--max_seq_len", type=int, default=14, help="(Unused here; only for loaders with padding)")
    ap.add_argument("--indices", type=str, default="0:5", help='Indices spec: "0:5", "0,10,20", "123", "rand10"')
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--enc_layers", type=int, default=4)
    ap.add_argument("--dec_layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--save_dir", type=str, default=None, help="If set, save figs to this dir instead of plt.show()")
    ap.add_argument("--no_clamp_pi", action="store_true", help="Disable clamping outputs to [0, pi]")
    args = ap.parse_args()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data (no DataLoader needed for per-sample viz)
    folders = [s.strip() for s in args.pairs.split(",") if s.strip()]
    X_list, Y_list, F_list = scan_and_load(args.data_dir, folders,
                                           sample_size=args.sample_size,
                                           freq_size=args.freq_size,
                                           target_key=args.target,
                                           verbose=1, progress=True)
    X_tr, Y_tr, F_tr, X_va, Y_va, F_va, X_te, Y_te, F_te = split_datasets(X_list, Y_list, F_list)

    # Use test split for visualization
    ds_te = PnCDataset(X_te, F_te, Y_te)
    N = len(ds_te)
    indices = _parse_indices(args.indices, N)

    # Build model & load ckpt
    model = PnCFormer(
        x_input_dim=6, f_input_dim=1,
        d_model=args.d_model, nhead=args.nhead,
        num_encoder_layers=args.enc_layers, num_decoder_layers=args.dec_layers,
        dropout=args.dropout, output_seq_len=args.freq_size
    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device))
    model.eval()

    # Auto clamp: True for dispersion by default (disable with --no_clamp_pi)
    clamp = (not args.no_clamp_pi) and ("dispersion" in args.target)

    visualize_test_results_by_indices(
        model, ds_te, indices,
        clamp_pi=clamp,
        save_dir=args.save_dir,
        title_prefix=f"{args.target}"
    )


if __name__ == "__main__":
    main()