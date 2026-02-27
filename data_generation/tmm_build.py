from __future__ import annotations
from .data_types import SimulationConfig

from physics.analysis import TMM

def build_tmm(config: SimulationConfig) -> "TMM":
    return TMM(
        modulus_A=config.mat_A.modulus_GPa * 1e9,
        density_A=config.mat_A.density,
        modulus_B=config.mat_B.modulus_GPa * 1e9,
        density_B=config.mat_B.density,
        batch_size=config.tmm_batch_size,
        freq_start_hz=config.f_start_hz,
        freq_stop_hz=config.f_stop_hz,
        freq_step_hz=config.f_step_hz,
    )