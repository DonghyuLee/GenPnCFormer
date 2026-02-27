# data_generation/visualize.py
from __future__ import annotations
import argparse
import os
from typing import List, Tuple, Dict, Any, Iterable
import json
import h5py
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ----------------------------- I/O helpers -----------------------------

def _jsonish(x):
    """Decode JSON-like attributes stored as bytes/strings, otherwise return as-is."""
    if isinstance(x, (bytes, bytearray)):
        x = x.decode("utf-8", "ignore")
    try:
        return json.loads(x)
    except Exception:
        return x


def list_h5_files(data_dir: str, folders: List[str], pattern: str | None = None) -> List[Tuple[str, str]]:
    """
    Return list of (folder_name, absolute_file_path) for .h5 files under selected folders.
    If pattern is provided, only keep files whose names contain the pattern (simple substring filter).
    """
    entries: List[Tuple[str, str]] = []
    for folder in folders:
        root = os.path.join(data_dir, folder)
        if not os.path.isdir(root):
            print(f"[WARN] Missing folder: {root}")
            continue
        for fn in sorted(os.listdir(root)):
            if not fn.endswith(".h5"):
                continue
            if pattern and pattern not in fn:
                continue
            entries.append((folder, os.path.join(root, fn)))
    return entries


# ----------------------------- Validation -----------------------------

def check_identical_rows(arr: np.ndarray, atol=1e-12) -> bool:
    """Return True if every row equals the first row (within atol)."""
    if arr.ndim != 2:
        return False
    if arr.shape[0] <= 1:
        return True
    return np.allclose(arr, arr[0:1, :], atol=atol, rtol=0.0)


def maybe_to_khz(freq: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Try to infer unit: if max freq > 5e3 assume Hz and convert to kHz, else keep.
    Returns (freq_khz, unit_label).
    """
    mx = float(np.nanmax(freq))
    if mx > 5_000.0:
        return freq / 1_000.0, "kHz"
    return freq, "kHz"  # most datasets here already store kHz


def validate_file(fp: str, sample_cap: int, freq_cap: int) -> Dict[str, Any]:
    """
    Open one H5 file and run integrity checks.
    Returns a dict of summary and flags.
    """
    out: Dict[str, Any] = {"file": fp, "ok": True, "messages": []}
    with h5py.File(fp, "r") as h:
        keys = set(h.keys())
        required = {
            "design_variable",
            "dispersion_relation_unitcell",
            "dispersion_relation_supercell",
            "transmittance",
            "frequencies",
        }
        missing = sorted(list(required - keys))
        if missing:
            out["ok"] = False
            out["messages"].append(f"[ERROR] Missing datasets: {missing}")
            return out

        dv = h["design_variable"][:sample_cap]  # (N, L, 3) expected
        dr_uc = h["dispersion_relation_unitcell"][:sample_cap, :freq_cap]
        dr_sc = h["dispersion_relation_supercell"][:sample_cap, :freq_cap]
        tr    = h["transmittance"][:sample_cap, :freq_cap]
        fq    = h["frequencies"][:sample_cap, :freq_cap]

        # Basic shapes
        if not (dv.ndim == 3 and dv.shape[-1] == 3):
            out["ok"] = False
            out["messages"].append(f"[ERROR] design_variable must be (N,L,3), got {dv.shape}")
        if not (dr_uc.shape == dr_sc.shape == tr.shape == fq.shape):
            out["ok"] = False
            out["messages"].append(f"[ERROR] core arrays shape mismatch: "
                                   f"uc={dr_uc.shape}, sc={dr_sc.shape}, tr={tr.shape}, fq={fq.shape}")

        # Frequency rows should be identical
        if not check_identical_rows(fq):
            out["ok"] = False
            out["messages"].append("[ERROR] frequencies rows are not identical")

        # Attributes (optional but helpful)
        attrs = {k: _jsonish(v) for k, v in h.attrs.items()}
        out["attrs"] = attrs

        # If cells attr exists, check L == 2*cells
        if "cells" in attrs:
            cells = int(attrs["cells"])
            L = int(dv.shape[1]) if dv.ndim == 3 else -1
            if L != 2 * cells:
                out["ok"] = False
                out["messages"].append(f"[ERROR] layer count L={L} != 2*cells={2*cells}")

        # Physical ranges sanity
        # dispersion ~ [0, pi], transmittance >= 0 (often <=1 but may exceed with specific definitions)
        bad_uc = np.logical_or(dr_uc < -1e-6, dr_uc > np.pi + 1e-6).sum()
        bad_sc = np.logical_or(dr_sc < -1e-6, dr_sc > np.pi + 1e-6).sum()
        if bad_uc > 0:
            out["ok"] = False
            out["messages"].append(f"[ERROR] unitcell dispersion out-of-range count={bad_uc}")
        if bad_sc > 0:
            out["ok"] = False
            out["messages"].append(f"[ERROR] supercell dispersion out-of-range count={bad_sc}")

        if (tr < 0.0).any():
            out["ok"] = False
            out["messages"].append("[ERROR] transmittance has negative values")

        if (tr > 1.0 + 1e-3).any():
            # Not strictly an error in all models, but highlight
            out["messages"].append("[WARN] transmittance exceeds 1.0 (check normalization)")

        out["shapes"] = {
            "design_variable": tuple(dv.shape),
            "dispersion_relation_unitcell": tuple(dr_uc.shape),
            "dispersion_relation_supercell": tuple(dr_sc.shape),
            "transmittance": tuple(tr.shape),
            "frequencies": tuple(fq.shape),
        }

    return out


# ----------------------------- Plotting -----------------------------

def plot_two_panels(dr_uc: np.ndarray,
                    dr_sc: np.ndarray,
                    tr: np.ndarray,
                    freq_row: np.ndarray,
                    sample_idx: int,
                    unit_label: str = "kHz",
                    suptitle: str | None = None,
                    show: bool = True,
                    save_path: str | None = None):
    """
    Side-by-side plots:
      Left  : overlay of unitcell vs supercell dispersion (x: wavenumber, y: frequency)
      Right : transmittance (x: frequency, y: transmittance)
    """
    # Ensure 1-D arrays
    dr_uc = np.asarray(dr_uc).ravel()
    dr_sc = np.asarray(dr_sc).ravel()
    tr    = np.asarray(tr).ravel()
    freq  = np.asarray(freq_row).ravel()

    # Try to clamp dispersion to [0, pi] for visualization sanity
    dr_uc_vis = np.clip(dr_uc, 0.0, np.pi)
    dr_sc_vis = np.clip(dr_sc, 0.0, np.pi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    ax0, ax1 = axes

    # Left: dispersion overlay (x = wavenumber, y = frequency)
    ax0.plot(dr_uc_vis, freq, label="Unitcell", linewidth=2.0)
    ax0.plot(dr_sc_vis, freq, label="Supercell", linestyle="--", linewidth=2.0)
    ax0.set_title(f"Dispersion (sample #{sample_idx})")
    ax0.set_xlabel("Wavenumber [rad]")
    ax0.set_ylabel(f"Frequency [{unit_label}]")
    ax0.grid(True)
    ax0.legend()

    # Right: transmittance (x = frequency, y = tr)
    ax1.plot(freq, tr, linewidth=2.0)
    ax1.set_title(f"Transmittance (sample #{sample_idx})")
    ax1.set_xlabel(f"Frequency [{unit_label}]")
    ax1.set_ylabel("Transmittance")
    ax1.grid(True)

    if suptitle:
        fig.suptitle(suptitle)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
    elif show:
        plt.show()
    else:
        plt.close(fig)


def _parse_indices(spec: str, n: int) -> List[int]:
    """
    Parse indices specification into a list of indices:
      - "10"          -> [10]
      - "0,5,12"      -> [0, 5, 12]
      - "100:110"     -> [100..109]
      - "rand10"      -> 10 random unique indices in [0, n)
    """
    spec = (spec or "").strip()
    if not spec:
        return [0]
    if spec.startswith("rand"):
        try:
            k = int(spec[4:])
        except Exception:
            k = 5
        rng = np.random.default_rng(42)
        return sorted(rng.choice(n, size=min(k, n), replace=False).tolist())
    if ":" in spec:
        a, b = spec.split(":")
        a = int(a) if a else 0
        b = int(b) if b else n
        a = max(0, a); b = min(b, n)
        return list(range(a, b))
    if "," in spec:
        return [max(0, min(n - 1, int(s))) for s in spec.split(",") if s.strip() != ""]
    return [max(0, min(n - 1, int(spec)))]


# ----------------------------- Main CLI -----------------------------

def main():
    ap = argparse.ArgumentParser(description="Verify generated HDF5 datasets and visualize sample plots.")
    ap.add_argument("--data_dir", type=str, default="data", help="Root directory containing data subfolders (e.g., data/CA)")
    ap.add_argument("--pairs", type=str, default="CA", help='Comma-separated subfolders to include, e.g. "TA,CA,SA"')
    ap.add_argument("--pattern", type=str, default=None, help="Optional substring filter to match file names (e.g., 'AC')")
    ap.add_argument("--sample_size", type=int, default=20000, help="Max samples per file to read")
    ap.add_argument("--freq_size", type=int, default=1000, help="Max frequency points to read per sample")
    ap.add_argument("--indices", type=str, default="0:2", help='Indices to plot per file: "0:2", "0,10,20", "123", "rand5"')
    ap.add_argument("--save_dir", type=str, default=None, help="If set, save figures here instead of showing")
    ap.add_argument("--no_show", action="store_true", help="Disable plt.show(); use with --save_dir for batch export")
    ap.add_argument("--verbose", type=int, default=1, help="0=silent, 1=summary(default), 2=per-file details")
    args = ap.parse_args()

    folders = [s.strip() for s in args.pairs.split(",") if s.strip()]
    files = list_h5_files(args.data_dir, folders, pattern=args.pattern)

    if args.verbose >= 1:
        print(f"[scan] folders={folders} files_found={len(files)} sample_cap={args.sample_size} freq_cap={args.freq_size}")

    ok_count = 0
    fail_count = 0

    it = tqdm(files, desc="Verifying", unit="file") if args.verbose >= 1 else files
    for folder, fp in it:
        v = validate_file(fp, args.sample_size, args.freq_size)
        if v["ok"]:
            ok_count += 1
        else:
            fail_count += 1

        if args.verbose >= 2:
            print(f"\n[{folder}/{os.path.basename(fp)}]")
            for k, shp in v.get("shapes", {}).items():
                print(f"  {k:35s}: {shp}")
            if "attrs" in v:
                cells = v["attrs"].get("cells", None)
                defects = v["attrs"].get("defects", None)
                print(f"  attrs.cells={cells}, attrs.defects={defects}")
            for msg in v["messages"]:
                print(" ", msg)

        # Skip plotting on invalid files
        if not v["ok"]:
            continue

        # Load data again for plotting (avoid keeping large arrays around)
        with h5py.File(fp, "r") as h:
            # Slice with caps
            dr_uc = h["dispersion_relation_unitcell"][:args.sample_size, :args.freq_size]
            dr_sc = h["dispersion_relation_supercell"][:args.sample_size, :args.freq_size]
            tr    = h["transmittance"][:args.sample_size, :args.freq_size]
            fq    = h["frequencies"][:args.sample_size, :args.freq_size]
            # Choose any row (all identical)
            freq_row = fq[0]
            freq_row, unit_label = maybe_to_khz(freq_row)

            N = dr_uc.shape[0]
            sel = _parse_indices(args.indices, N)

            for idx in sel:
                suptitle = f"{folder}/{os.path.basename(fp)}"
                save_path = None
                if args.save_dir:
                    base = os.path.splitext(os.path.basename(fp))[0]
                    out_name = f"{base}_idx{idx:05d}.png"
                    save_path = os.path.join(args.save_dir, folder, out_name)

                plot_two_panels(
                    dr_uc[idx], dr_sc[idx], tr[idx], freq_row,
                    sample_idx=idx, unit_label=unit_label,
                    suptitle=suptitle,
                    show=(not args.no_show and not args.save_dir),
                    save_path=save_path
                )

    if args.verbose >= 1:
        print(f"\n[summary] ok_files={ok_count} failed_files={fail_count} total={len(files)}")


if __name__ == "__main__":
    main()
