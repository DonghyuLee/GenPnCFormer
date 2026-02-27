from __future__ import annotations
from typing import List, Tuple, Dict, Any
import os
import numpy as np
from collections import defaultdict
from tqdm.auto import tqdm
import torch
from torch.utils.data import Dataset
from .features import load_h5_to_lists


def scan_and_load(data_dir: str,
                  selected_folders: list[str],
                  sample_size: int,
                  freq_size: int, 
                  target_key: str = "dispersion_relation_supercell",
                  *,
                  verbose: int = 1,
                  progress: bool = True,
                  return_summary: bool = False):
    """
    Scan folders under `data_dir`, load .h5 files, and build X/Y/F lists.

    Assumptions:
      - `design_variable` is stored as (N, L, 3).
      - `target_key` is one of: "dispersion_relation_supercell",
        "dispersion_relation_unitcell", or "transmittance".
      - Frequencies are (N, F); rows are identical.

    Args:
      data_dir: Root directory containing data subfolders.
      selected_folders: Subfolder names to include (e.g., ["CA", "SA"]).
      sample_size: Max number of samples to read per file.
      freq_size: Max frequency points to read per sample.
      target_key: Dataset key to use as target.
      verbose:
        0 = silent
        1 = one-line final summary (default)
        2 = summary + per-folder counts
        3 = verbose per-file logging
      progress: Show a tqdm progress bar while loading.
      return_summary: If True, returns a summary dict as the 4th value.

    Returns:
      (X_list, Y_list, F_list) or (X_list, Y_list, F_list, summary_dict) if return_summary=True.
    """
    # 1) Collect all files first
    file_entries: List[Tuple[str, str]] = []  # (folder, filepath)
    missing_folders: List[str] = []

    for folder in selected_folders:
        folder_path = os.path.join(data_dir, folder)
        if not os.path.isdir(folder_path):
            missing_folders.append(folder_path)
            continue
        for fn in sorted(os.listdir(folder_path)):
            if fn.endswith(".h5"):
                file_entries.append((folder, os.path.join(folder_path, fn)))

    if verbose >= 1 and missing_folders:
        for m in missing_folders:
            print(f"[WARN] Missing folder: {m}")

    # 2) Load with optional progress bar
    iterator = file_entries
    if progress:
        iterator = tqdm(file_entries, desc=f"Loading {target_key}", unit="file", leave=False)

    X_list: List = []
    Y_list: List = []
    F_list: List = []

    # Aggregates for summary reporting
    per_folder_files: Dict[str, int] = defaultdict(int)
    per_folder_samples: Dict[str, int] = defaultdict(int)
    total_samples = 0

    for folder, filepath in iterator:
        X, Y, F = load_h5_to_lists(filepath, sample_size=sample_size, freq_size=freq_size, target_key=target_key)

        if verbose >= 3:
            print(f"{folder}/{os.path.basename(filepath)} loaded: X={X.shape}, Y={Y.shape}, F={F.shape}")

        X_list.append(X)
        Y_list.append(Y)
        F_list.append(F)

        n = int(X.shape[0])
        per_folder_files[folder] += 1
        per_folder_samples[folder] += n
        total_samples += n

        if progress and hasattr(iterator, "set_postfix"):
            iterator.set_postfix(last=f"{folder}/{os.path.basename(filepath)}", samples=total_samples)

    # 3) Final summary
    if verbose >= 1:
        print(f"[load] folders={len(selected_folders)} "
              f"files={len(file_entries)} "
              f"samples={total_samples:,} "
              f"(per-file cap={sample_size}) F={freq_size} target={target_key}")
    if verbose >= 2:
        for folder in selected_folders:
            if per_folder_files[folder] > 0:
                print(f"  - {folder}: files={per_folder_files[folder]} samples={per_folder_samples[folder]:,}")

    summary: Dict[str, Any] = {
        "folders_requested": selected_folders,
        "files_found": len(file_entries),
        "total_samples": total_samples,
        "per_file_cap": sample_size,
        "freq_size": freq_size,
        "target_key": target_key,
        "per_folder_files": dict(per_folder_files),
        "per_folder_samples": dict(per_folder_samples),
        "missing_folders": missing_folders,
    }

    return (X_list, Y_list, F_list, summary) if return_summary else (X_list, Y_list, F_list)

def split_datasets(X_list: List[np.ndarray],
                   Y_list: List[np.ndarray],
                   F_list: List[np.ndarray],
                   train_ratio: float = 0.9,
                   valid_ratio: float = 0.05,
                   test_ratio: float | None = None):
    """
    Split per-file, then concat Y/F; keep X as a flat list of sequences.
    """
    if test_ratio is None:
        test_ratio = 1.0 - train_ratio - valid_ratio

    X_tr_l, Y_tr_l, F_tr_l = [], [], []
    X_va_l, Y_va_l, F_va_l = [], [], []
    X_te_l, Y_te_l, F_te_l = [], [], []

    for X, Y, F in zip(X_list, Y_list, F_list):
        n = X.shape[0]
        n_tr = int(n * train_ratio)
        n_va = int(n * valid_ratio)
        # train
        X_tr_l.append(X[:n_tr])
        Y_tr_l.append(Y[:n_tr])
        F_tr_l.append(F[:n_tr])
        # val
        X_va_l.append(X[n_tr:n_tr + n_va])
        Y_va_l.append(Y[n_tr:n_tr + n_va])
        F_va_l.append(F[n_tr:n_tr + n_va])
        # test
        X_te_l.append(X[n_tr + n_va:])
        Y_te_l.append(Y[n_tr + n_va:])
        F_te_l.append(F[n_tr + n_va:])

    Y_train = np.concatenate(Y_tr_l, axis=0)
    Y_valid = np.concatenate(Y_va_l, axis=0)
    Y_test  = np.concatenate(Y_te_l, axis=0)

    F_train = np.concatenate(F_tr_l, axis=0)
    F_valid = np.concatenate(F_va_l, axis=0)
    F_test  = np.concatenate(F_te_l, axis=0)

    def _flatten_X_list(L):
        out = []
        for arr in L:
            for i in range(arr.shape[0]):
                out.append(arr[i])
        return out

    X_train = _flatten_X_list(X_tr_l)
    X_valid = _flatten_X_list(X_va_l)
    X_test  = _flatten_X_list(X_te_l)

    return X_train, Y_train, F_train, X_valid, Y_valid, F_valid, X_test, Y_test, F_test


class PnCDataset(Dataset):
    """
    Holds variable-length sequences X (L, feat_dim), and targets Y (F,) with decoder queries Fgrid (F,).
    """
    def __init__(self, X_list: List[np.ndarray], F: np.ndarray, Y: np.ndarray):
        assert len(X_list) == Y.shape[0] == F.shape[0], "Mismatched lengths."
        self.X_list = [torch.tensor(x, dtype=torch.float32) for x in X_list]
        self.F = torch.tensor(F, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)

    def __len__(self): return len(self.X_list)

    def __getitem__(self, idx):
        return self.X_list[idx], self.F[idx], self.Y[idx]


def collate_fn(batch, max_seq_len: int):
    """
    Pad sequences to max_seq_len and build attention mask.
    Returns: padded_X (B, max_seq_len, feat), F (B, F), Y (B, F), attn_mask (B, max_seq_len)
    """
    X_list, F_list, Y_list = zip(*batch)
    padded_X, attn_mask = [], []
    for x in X_list:
        cur_len = x.size(0)
        if cur_len < max_seq_len:
            pad_len = max_seq_len - cur_len
            # pad: (last dim unchanged, pad rows at end)
            x_pad = torch.nn.functional.pad(x, (0, 0, 0, pad_len), mode="constant", value=0.0)
        else:
            x_pad = x[:max_seq_len]
            cur_len = max_seq_len
        padded_X.append(x_pad)
        mask = torch.ones(max_seq_len, dtype=torch.bool)
        mask[:cur_len] = False
        attn_mask.append(mask)

    padded_X  = torch.stack(padded_X, dim=0)
    attn_mask = torch.stack(attn_mask, dim=0)
    F_tensor  = torch.stack(F_list, dim=0)
    Y_tensor  = torch.stack(Y_list, dim=0)
    return padded_X, F_tensor, Y_tensor, attn_mask


