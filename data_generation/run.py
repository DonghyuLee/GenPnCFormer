from __future__ import annotations
import numpy as np
import os

from . import Material, SimulationConfig, generate_cases
from .core import run_case_torch

def main():
    # --- defined materials ---
    mat_copper = Material(name="Copper", modulus_GPa=110.0, density=8960.0)
    mat_aluminum = Material(name="Aluminum", modulus_GPa=70.0, density=2700.0)
    mat_steel = Material(name="Steel", modulus_GPa=200.0, density=7850.0)
    mat_titanium = Material(name="Titanium", modulus_GPa=116.0, density=4500.0)

    # --- configurations ---
    configs_to_run = [
        ("TA", mat_titanium, mat_aluminum),
        ("CA", mat_copper, mat_aluminum),
        ("SA", mat_steel, mat_aluminum),
        ("AT", mat_aluminum, mat_titanium),
        ("AC", mat_aluminum, mat_copper),
        ("AS", mat_aluminum, mat_steel),
    ]

    # Shared parameters
    n_samples = 20000
    var_low = 0.005
    var_high = 0.100
    random_seed = 7
    f_start_hz = 100.0
    f_stop_hz = 50000.0
    f_step_hz = 100.0
    tmm_batch_size = 5000
    n_jobs = 8
    min_cells = 4
    max_cells = 7
    include_double_defects = True

    # RNG & grids
    np.random.seed(random_seed)
    variables = np.random.uniform(low=var_low, high=var_high, size=(n_samples, 4))
    freq_grid_khz = np.arange(f_start_hz, f_stop_hz + f_step_hz, f_step_hz) / 1000.0

    # Generate cases
    cases = generate_cases(min_cells, max_cells, include_double=include_double_defects)

    for prefix, mat_a, mat_b in configs_to_run:
        print(f"\n========================================")
        print(f"Starting Generation for: {prefix} ({mat_a.name} / {mat_b.name})")
        print(f"========================================")
        
        data_dir = f"data/{prefix}"
        os.makedirs(data_dir, exist_ok=True)
        
        config = SimulationConfig(
            mat_A=mat_a,
            mat_B=mat_b,
            n_samples=n_samples,
            var_low=var_low, var_high=var_high,
            random_seed=random_seed,
            f_start_hz=f_start_hz, f_stop_hz=f_stop_hz, f_step_hz=f_step_hz,
            tmm_batch_size=tmm_batch_size,
            n_jobs=n_jobs,
            data_dir=data_dir,
            min_cells=min_cells, max_cells=max_cells,
            include_double_defects=include_double_defects
        )
        
        # Run all cases (Torch/GPU accelerated)
        for case in cases:
            run_case_torch(case, config, variables, freq_grid_khz, prefix=prefix)

if __name__ == "__main__":
    main()