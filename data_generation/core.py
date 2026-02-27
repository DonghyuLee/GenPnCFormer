from __future__ import annotations
from typing import List, Tuple, Sequence
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm

from .data_types import Material, Case, SimulationConfig
from .io_utils import filename_for_case, write_h5

import torch
from .tmm_torch import TorchTMM

def create_sample_chunk(mat_A: Material, mat_B: Material, lengths: np.ndarray, case: Case) -> np.ndarray:
    """
    Vectorized version of create_sample.
    lengths shape: (N, 4) where N is chunk_size
    Returns: (N, num_layers, 3) 
    """
    N = lengths.shape[0]
    total_cells = case.cells
    defect_cells = set(case.defects)

    layers = np.zeros((N, total_cells * 2, 3), dtype=np.float64)

    length_A = lengths[:, 0]
    length_B = lengths[:, 1]
    length_C = lengths[:, 2]
    length_D = lengths[:, 3]

    idx = 0
    defect_count = 0
    for cell in range(1, total_cells + 1):
        # A layer
        layers[:, idx, 0] = mat_A.modulus_GPa
        layers[:, idx, 1] = mat_A.density
        layers[:, idx, 2] = length_A
        idx += 1

        # B layer
        layers[:, idx, 0] = mat_B.modulus_GPa
        layers[:, idx, 1] = mat_B.density
        if cell in defect_cells:
            defect_count += 1
            if len(defect_cells) == 2:
                layers[:, idx, 2] = length_C if defect_count == 1 else length_D
            else:
                layers[:, idx, 2] = length_C
        else:
            layers[:, idx, 2] = length_B
        idx += 1

    return layers


def run_case_torch(case: Case,
                   config: SimulationConfig,
                   variables: np.ndarray,
                   freq_grid_khz: np.ndarray,
                   prefix: str = "CA") -> None:
    
    N = variables.shape[0]
    F = freq_grid_khz.shape[0]
    
    # Preallocate CPU arrays for the final results (saving memory by not keeping everything on GPU)
    design_vars_arr = np.zeros((N, case.cells * 2, 3), dtype=np.float64)
    dr_uc_arr       = np.zeros((N, F), dtype=np.float64)
    dr_sc_arr       = np.zeros((N, F), dtype=np.float64)
    trans_arr       = np.zeros((N, F), dtype=np.float64)
    freqs_arr       = np.repeat(freq_grid_khz[None, :], repeats=N, axis=0)

    # Use chunk size based on tmm_batch_size in config
    chunk_size = config.tmm_batch_size
    
    # We will instantiate TorchTMM per chunk because batch_size is fixed in TorchTMM __init__
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for start_idx in tqdm(range(0, N, chunk_size), desc=f"Chunks (Size {chunk_size})", leave=False):
        end_idx = min(start_idx + chunk_size, N)
        current_batch_size = end_idx - start_idx
        
        var_chunk = variables[start_idx:end_idx] # (B, 4)
        
        # 1. Input Sample (Vectorized via CPU Numpy)
        ds_chunk = create_sample_chunk(config.mat_A, config.mat_B, var_chunk, case)
        design_vars_arr[start_idx:end_idx] = ds_chunk
        
        # 2. Setup GPU TMM analyzer for this specific batch size
        tmm = TorchTMM(modulus_A=config.mat_A.modulus_GPa * 1e9,
                       density_A=config.mat_A.density,
                       modulus_B=config.mat_B.modulus_GPa * 1e9,
                       density_B=config.mat_B.density,
                       batch_size=current_batch_size,
                       freq_start_hz=config.f_start_hz,
                       freq_stop_hz=config.f_stop_hz,
                       freq_step_hz=config.f_step_hz,
                       device=device)
        
        # Send inputs to GPU, dimension needs to be (B, 1) for broadcasting in TorchTMM
        lA = torch.tensor(var_chunk[:, 0:1], dtype=torch.float64, device=device)
        lB = torch.tensor(var_chunk[:, 1:2], dtype=torch.float64, device=device)
        lC = torch.tensor(var_chunk[:, 2:3], dtype=torch.float64, device=device)
        lD = torch.tensor(var_chunk[:, 3:4], dtype=torch.float64, device=device)
        
        case_dict = {"cells": case.cells, "defects": list(case.defects)}
        
        # Calculate
        dr_uc = tmm.get_dispersion_relation_unitcell(lA, lB)
        dr_sc = tmm.get_dispersion_relation_supercell(lA, lB, lC, lD, case=case_dict)
        trans = tmm.get_transmittance(lA, lB, lC, lD, case=case_dict)
        
        # Move back to CPU and store
        dr_uc_arr[start_idx:end_idx] = dr_uc.cpu().numpy()
        dr_sc_arr[start_idx:end_idx] = dr_sc.cpu().numpy()
        trans_arr[start_idx:end_idx] = trans.cpu().numpy()

    # Save to H5
    filename = filename_for_case(prefix=prefix, case=case)
    out_path = f"{config.data_dir}/{filename}"

    meta = {
        "cells": case.cells,
        "defects": list(case.defects),
        "mat_A": {"name": config.mat_A.name, "E_GPa": config.mat_A.modulus_GPa, "rho": config.mat_A.density},
        "mat_B": {"name": config.mat_B.name, "E_GPa": config.mat_B.modulus_GPa, "rho": config.mat_B.density},
        "n_samples": config.n_samples,
        "var_range_m": [config.var_low, config.var_high],
        "freq_hz": {"start": config.f_start_hz, "stop": config.f_stop_hz, "step": config.f_step_hz},
        "tmm": {"batch_size": chunk_size},
        "random_seed": config.random_seed
    }

    write_h5(out_path, design_vars_arr, dr_uc_arr, dr_sc_arr, trans_arr, freqs_arr, meta)