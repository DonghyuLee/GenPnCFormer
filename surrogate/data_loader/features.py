from __future__ import annotations
import numpy as np
import h5py

def calculate_features(layers: np.ndarray) -> np.ndarray:
    """
    Input
      layers: (N, L, 3) with columns [modulus(GPa), density(kg/m^3), length(m)]
    Output
      X: (N, L, 6)
    """
    assert layers.ndim == 3 and layers.shape[-1] == 3, f"Expect (N,L,3), got {layers.shape}"
    modulus = layers[:, :, 0]
    density = layers[:, :, 1]
    length  = layers[:, :, 2]

    X1 = modulus / 1000.0                       # scaled modulus
    X2 = density / 1000.0                       # scaled density
    X3 = length                                 # length
    X4 = np.sqrt(modulus * density) / 1000.0    # mech. impedance
    X5 = np.sqrt(modulus / density)             # wave speed
    X6 = length / np.sqrt(modulus / density)    # time delay

    return np.stack([X1, X2, X3, X4, X5, X6], axis=2)  # (N, L, 6)


def load_h5_to_lists(file_path: str,
                     sample_size: int,
                     freq_size: int,
                     target_key: str = "dispersion_relation_supercell"
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load one HDF5 (with design_variable fixed as (N, L, 3)) and return (X, Y, F).
      X: (N, L, 6) features derived from layer stack
      Y: (N, F)    target (choose via target_key)
                   - dispersion_relation_unitcell
                   - dispersion_relation_supercell
                   - transmittance
      F: (N, F)    frequency grid (rows identical)
    """
    with h5py.File(file_path, "r") as hf:
        dv = hf["design_variable"][:sample_size]                         # (N, L, 3)
        assert dv.ndim == 3 and dv.shape[-1] == 3, f"design_variable must be (N,L,3), got {dv.shape}"

        if target_key not in hf:
            raise KeyError(f"'{target_key}' not found in '{file_path}'. "
                           f"Available: {list(hf.keys())}")
        Y  = hf[target_key][:sample_size, :freq_size]                    # (N, F)
        F  = hf["frequencies"][:sample_size, :freq_size]                 # (N, F)

    X = calculate_features(dv)  # (N, L, 6)
    return X, Y, F
