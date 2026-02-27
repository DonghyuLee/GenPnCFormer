from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class Material:
    name: str
    modulus_GPa: float
    density: float

@dataclass(frozen=True)
class Case:
    cells: int
    defects: Tuple[int, ...]  # 1-based indices (e.g., (2,), (2,4))

@dataclass
class SimulationConfig:
    # Materials
    mat_A: Material
    mat_B: Material

    # Design variable sampling (A,B,C,D lengths) in meters
    n_samples: int = 1000
    var_low: float = 0.005
    var_high: float = 0.100
    random_seed: int = 7

    # Frequency grid (Hz)
    f_start_hz: float = 100.0
    f_stop_hz: float = 100000.0
    f_step_hz: float = 100.0

    # TMM params
    tmm_batch_size: int = 1
    tmm_data_points: int = 1000

    # Compute
    n_jobs: int = -1

    # IO
    data_dir: str = r"data\CA"

    # Case generation
    min_cells: int = 4
    max_cells: int = 7
    include_double_defects: bool = True