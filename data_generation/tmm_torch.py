import torch
import numpy as np

def batch_matrix_chain_multiply_torch(cell_TMs: torch.Tensor, T_init: torch.Tensor) -> torch.Tensor:
    """
    Vectorized 2x2 chain multiply over (cells, B, DP) using PyTorch.

    Parameters
    ----------
    cell_TMs : torch.Tensor
        Shape (C, B, DP, 2, 2)
    T_init : torch.Tensor
        Shape (B, DP, 2, 2).
    """
    C = cell_TMs.shape[0]
    T = T_init.clone()

    for c in range(C):
        M = cell_TMs[c]  # (B, DP, 2, 2)
        # T = T @ M
        # Using explicit multiplication for performance, or torch.matmul
        T = torch.matmul(T, M)

    return T


class TorchTMM:
    def __init__(self,
                 modulus_A: float, density_A: float,
                 modulus_B: float, density_B: float,
                 batch_size: int,
                 freq_start_hz: float,
                 freq_stop_hz: float,
                 freq_step_hz: float,
                 device: str = 'cuda'):
        self.modulus_A = float(modulus_A)
        self.density_A = float(density_A)
        self.modulus_B = float(modulus_B)
        self.density_B = float(density_B)
        self.batch_size = int(batch_size)
        self.device = torch.device(device)

        fs, fe, d = float(freq_start_hz), float(freq_stop_hz), float(freq_step_hz)
        DP = int(round((fe - fs) / d)) + 1
        frow = fs + d * torch.arange(DP, dtype=torch.float64, device=self.device)
        self.freq_start_hz = fs
        self.freq_step_hz = d

        self.data_points = DP
        # (B, DP)
        self.f = frow.unsqueeze(0).expand(self.batch_size, -1)

    def KL(self, modulus: float, density: float, length: torch.Tensor) -> torch.Tensor:
        """
        freq: (B,DP), length: (B, 1) -> broadcast to (B,DP)
        """
        velocity = np.sqrt(modulus / density)
        wavenumber = 2.0 * np.pi * self.f / velocity
        return wavenumber * length

    def Impedance(self, modulus: float, density: float) -> float:
        return float(np.sqrt(density * modulus))

    def TM(self, modulus: float, density: float, length: torch.Tensor) -> torch.Tensor:
        """
        length shape: (B, 1) -> Returns (B, DP, 2, 2) complex128 tensor
        """
        KL_val = self.KL(modulus, density, length)  # (B, DP)
        cos_KL = torch.cos(KL_val)
        sin_KL = torch.sin(KL_val)
        Im = self.Impedance(modulus, density)

        # Allocate
        TM_real = torch.zeros((self.batch_size, self.data_points, 2, 2), dtype=torch.float64, device=self.device)
        TM_real[..., 0, 0] = cos_KL
        TM_real[..., 1, 1] = cos_KL

        TM_imag = torch.zeros((self.batch_size, self.data_points, 2, 2), dtype=torch.float64, device=self.device)
        TM_imag[..., 0, 1] = sin_KL / Im
        TM_imag[..., 1, 0] = Im * sin_KL

        return torch.complex(TM_real, TM_imag)

    def PM(self, modulus: float, density: float) -> torch.Tensor:
        """
        Returns (B, DP, 2, 2) complex propagation matrix
        """
        omega = 2.0 * np.pi * self.f
        Z = self.Impedance(modulus, density)

        PM_real = torch.zeros((self.batch_size, self.data_points, 2, 2), dtype=torch.float64, device=self.device)
        PM_imag = torch.zeros((self.batch_size, self.data_points, 2, 2), dtype=torch.float64, device=self.device)

        PM_imag[..., 0, 0] = omega
        PM_imag[..., 0, 1] = omega
        PM_imag[..., 1, 0] = -omega * Z
        PM_imag[..., 1, 1] = omega * Z

        return torch.complex(PM_real, PM_imag)

    def _build_cell_TMs(self,
                        length_A: torch.Tensor,
                        length_B: torch.Tensor,
                        length_C: torch.Tensor,
                        length_D: torch.Tensor,
                        case: dict) -> torch.Tensor:
        """
        Build per-cell transfer matrices (C, B, DP, 2, 2)
        """
        TM_A = self.TM(self.modulus_A, self.density_A, length_A)
        TM_B = self.TM(self.modulus_B, self.density_B, length_B)

        BA = torch.matmul(TM_B, TM_A)  # (B,DP,2,2)
        defects = list(case.get("defects", []))
        C = int(case.get("cells", 0))

        # Fill all cells, stack to (C, B, DP, 2, 2)
        cell_TMs = BA.unsqueeze(0).expand(C, -1, -1, -1, -1).clone()

        if len(defects) == 2:
            D1A = torch.matmul(self.TM(self.modulus_B, self.density_B, length_C), TM_A)
            D2A = torch.matmul(self.TM(self.modulus_B, self.density_B, length_D), TM_A)
            i1, i2 = defects[0] - 1, defects[1] - 1
            cell_TMs[i1] = D1A
            cell_TMs[i2] = D2A
        elif len(defects) == 1:
            DA = torch.matmul(self.TM(self.modulus_B, self.density_B, length_C), TM_A)
            i = defects[0] - 1
            cell_TMs[i] = DA

        return cell_TMs

    def get_dispersion_relation_unitcell(self,
                                         length_A: torch.Tensor,
                                         length_B: torch.Tensor) -> torch.Tensor:
        TM_A = self.TM(self.modulus_A, self.density_A, length_A)
        TM_B = self.TM(self.modulus_B, self.density_B, length_B)
        T = torch.matmul(TM_B, TM_A)
        trace_T = T[..., 0, 0] + T[..., 1, 1]
        x = torch.clamp(torch.real(trace_T) / 2.0, -1.0, 1.0)
        return torch.acos(x)

    def get_dispersion_relation_supercell(self,
                                          length_A: torch.Tensor,
                                          length_B: torch.Tensor,
                                          length_C: torch.Tensor,
                                          length_D: torch.Tensor,
                                          case: dict) -> torch.Tensor:
        cell_TMs = self._build_cell_TMs(length_A, length_B, length_C, length_D, case)

        B, DP = self.batch_size, self.data_points
        T_init = torch.zeros((B, DP, 2, 2), dtype=torch.complex128, device=self.device)
        T_init[..., 0, 0] = 1.0
        T_init[..., 1, 1] = 1.0

        # Note: TMM code loops cell_TMs[::-1]
        T = batch_matrix_chain_multiply_torch(torch.flip(cell_TMs, [0]), T_init)
        trace_T = T[..., 0, 0] + T[..., 1, 1]
        x = torch.clamp(torch.real(trace_T) / 2.0, -1.0, 1.0)
        return torch.acos(x)

    def get_transmittance(self,
                          length_A: torch.Tensor,
                          length_B: torch.Tensor,
                          length_C: torch.Tensor,
                          length_D: torch.Tensor,
                          case: dict) -> torch.Tensor:
        cell_TMs = self._build_cell_TMs(length_A, length_B, length_C, length_D, case)

        B, DP = self.batch_size, self.data_points
        T_init = torch.zeros((B, DP, 2, 2), dtype=torch.complex128, device=self.device)
        T_init[..., 0, 0] = 1.0
        T_init[..., 1, 1] = 1.0

        T = batch_matrix_chain_multiply_torch(torch.flip(cell_TMs, [0]), T_init)

        PM = self.PM(self.modulus_B, self.density_B)
        # 2x2 batched inverse exists in pytorch: torch.linalg.inv
        PM_inv = torch.linalg.inv(PM)
        SM = torch.matmul(torch.matmul(PM_inv, T), PM)

        det_SM = torch.linalg.det(SM)
        SM_11 = SM[..., 1, 1]

        eps = 1e-30
        tr = (torch.abs(det_SM) / torch.clamp_min(torch.abs(SM_11), eps)) ** 2
        return tr

