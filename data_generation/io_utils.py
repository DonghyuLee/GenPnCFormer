from __future__ import annotations
import os, json, time
from typing import Any, Dict
import numpy as np
import h5py
from .data_types import Case

def filename_for_case(prefix: str, case: Case) -> str:
    if len(case.defects) == 0:
        defect_str = "00"
    elif len(case.defects) == 1:
        defect_str = f"{case.defects[0]}0"
    else:
        defect_str = "".join(map(str, case.defects))
    return f"{prefix}{case.cells}{defect_str}.h5"

def write_h5(path: str,
             design_vars: np.ndarray,
             dr_uc: np.ndarray,
             dr_sc: np.ndarray,
             trans: np.ndarray,
             freqs: np.ndarray,
             meta: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with h5py.File(path, "w") as hf:
        hf.create_dataset("design_variable", data=design_vars)
        hf.create_dataset("dispersion_relation_unitcell", data=dr_uc)
        hf.create_dataset("dispersion_relation_supercell", data=dr_sc)
        hf.create_dataset("transmittance", data=trans)
        hf.create_dataset("frequencies", data=freqs)

        # metadata for reproducibility
        meta = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), **meta}
        for k, v in meta.items():
            try:
                if isinstance(v, (dict, list, tuple)):
                    hf.attrs[k] = json.dumps(v)
                elif v is None:
                    hf.attrs[k] = "None"
                else:
                    hf.attrs[k] = v
            except Exception:
                hf.attrs[k] = str(v)