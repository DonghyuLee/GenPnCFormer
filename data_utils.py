import os
import h5py
import numpy as np
from scipy.signal import find_peaks
import torch
import torch.nn.functional as F

from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import matplotlib.pyplot as plt

# ==============================================================================

def calculate_features(X):
    modulus = X[:, :, 0]
    density = X[:, :, 1]
    length  = X[:, :, 2]
    X1 = modulus / 1000
    X2 = density / 1000
    X3 = length 
    return np.stack([X1, X2, X3], axis=2)

def pad_or_truncate_X(X, max_cells):
    n, L, C = X.shape
    out = np.zeros((n, max_cells, C), dtype=X.dtype)
    use = min(L, max_cells)
    out[:, :use, :] = X[:, :use, :]
    return out

def _runs_bool(b):
    """bool 1D 배열 True 연속구간을 (start,end) (end포함) 리스트로"""
    F = b.shape[0]
    runs, i = [], 0
    while i < F:
        if b[i]:
            s = i
            while i+1 < F and b[i+1]:
                i += 1
            runs.append((s, i))
        i += 1
    return runs

def _smooth_1d(y, k=3):
    """간단 이동평균(홀수 k)"""
    if k <= 1: return y
    # k를 홀수로 보장
    k = max(1, int(k) | 1)
    pad = k//2
    ypad = np.pad(y, (pad, pad), mode='edge')
    ker = np.ones(k)/k
    return np.convolve(ypad, ker, mode='valid')

# --- 통합된 메인 마스크 함수 (D와 T 모두 사용) ---

def build_band_mask(
    udr, sdr, trans=None,
    tol_zero=0.01 * np.pi, tol_pi=0.01 * np.pi,
    bandgap_margin_bins=0,
    peak_height=0.1, peak_prominence=0.05, peak_distance=5,
    smooth_bins=1,
    out_mask=None
):
    """
    분산 관계(UDR)와 Supercell Dispersion(SDR)을 사용하여 단일 샘플의 밴드 마스크를 생성합니다.
    
    Args:
        udr (np.ndarray): 단일 샘플의 비접힘 분산 관계 (UDR). (F,) 형태.
        sdr (np.ndarray): 단일 샘플의 슈퍼셀 분산 관계 (SDR). (F,) 형태.
        trans (np.ndarray): 단일 샘플의 전달률 (Optional).
        tol_zero (float): UDR=0 주변에서 밴드갭으로 간주할 허용 오차.
        tol_pi (float): UDR=pi 주변에서 밴드갭으로 간주할 허용 오차.
        bandgap_margin_bins (int): 밴드갭 경계에서 마진을 줄 빈 개수.
        
    Returns:
        np.ndarray: 밴드 마스크 배열 (0: 전달 밴드, 1: 밴드갭, 2: 디펙트 밴드). (F,) 형태.
    """
    u = np.nan_to_num(udr, nan=0.0)
    
    F = u.shape[0]

    # 1) UC gap mask 정의 (UDR 사용)
    # 1a. 0과 pi 근처 영역을 각각 정의
    is_near_zero = (np.abs(u - 0.0) <= tol_zero)
    is_near_pi = (np.abs(u - np.pi) <= tol_pi)

    # 1b. 저주파 예외 처리 (0에서 시작하는 Acoustic branch)
    is_not_near_zero = ~is_near_zero
    first_pass_end = 0
    if F > 0 and is_near_zero[0]:
        if is_not_near_zero.any():
            first_pass_end = np.where(is_not_near_zero)[0][0]
        else:
            first_pass_end = F

    # 1c. 최종 밴드갭 마스크
    uc_gap = is_near_zero | is_near_pi
    uc_gap[0 : first_pass_end] = False

    # 2) 기본 마스크 설정: gap=1, pass=0
    if out_mask is None:
        mask = np.zeros_like(u, dtype=np.uint8)
    else:
        mask = out_mask
        mask.fill(0)
        
    mask[uc_gap] = 1

    # 3) Defect Detection using Supercell Dispersion (SDR)
    # Logic: Gap 구간 내부에서 SDR이 0에서 pi로 가로지르는 밴드를 모두 찾음 (Double Defect 등 겹침 허용).
    
    s = np.nan_to_num(sdr, nan=0.0)
    gap_runs = _runs_bool(uc_gap)
    
    for (g_start, g_end) in gap_runs:
        gap_length = g_end - g_start + 1
        if gap_length < 3: continue
        
        # 앞뒤로 여유(2포인트)를 두어 끝점 터치 확인 보완
        check_s = max(0, g_start - 2)
        check_e = min(F - 1, g_end + 2)
        
        band_vals = s[check_s : check_e + 1]
        
        # Double Defect같이 밴드가 여러 번 꺾이는 부분(Extrema) 찾기
        peaks, _ = find_peaks(band_vals, prominence=1.0)
        valleys, _ = find_peaks(-band_vals, prominence=1.0)
        
        turn_pts = [0] + sorted(list(peaks) + list(valleys)) + [len(band_vals) - 1]
        
        for i in range(len(turn_pts) - 1):
            seg_s = turn_pts[i]
            seg_e = turn_pts[i+1]
            if seg_e - seg_s < 1: continue
            
            check_vals = band_vals[seg_s : seg_e + 1]
            b_range = np.max(check_vals) - np.min(check_vals)
            
            # K-point resolution 한계를 고려하여 b_range > 2.5 이면 0~pi를 횡단한 것으로 인정
            if b_range >= 2.5:
                 # 밴드가 가장 평평한(Dispersion X가 가장 급격하게 변하는) 위치를 찾음
                 diffs = np.abs(np.diff(check_vals))
                 max_diff_idx = np.argmax(diffs)
                 
                 # 변동이 일어나는 두 포인트 중 뒤쪽(혹은 앞쪽)을 중심점으로 잡음
                 center_idx = check_s + seg_s + max_diff_idx
                 
                 # 만약 해당 인덱스가 밴드갭을 벗어났다면 갭 안쪽으로 조정
                 if center_idx < F and not uc_gap[center_idx]:
                     if center_idx + 1 < F and uc_gap[center_idx + 1]:
                         center_idx += 1
                     elif center_idx - 1 >= 0 and uc_gap[center_idx - 1]:
                         center_idx -= 1
                 
                 if 0 <= center_idx < F and uc_gap[center_idx]:
                     # 밴드갭 끝단(위/아래 모서리)에 너무 바싹 붙어있는 밴드는 결함으로 보지 않음 (기본 마진 5칸)
                     margin = max(5, bandgap_margin_bins)
                     if (center_idx - g_start >= margin) and (g_end - center_idx >= margin):
                         mask[center_idx] = 2

    return mask

def split_datasets(X_list, D_list, F_list, M_list, cfg):
    """
    List of Arrays -> Concat -> Shuffle -> Split (8:1:1) -> Return Arrays
    """
    X_tr_l, D_tr_l, F_tr_l, M_tr_l = [], [], [], []
    X_va_l, D_va_l, F_va_l, M_va_l = [], [], [], []
    X_te_l, D_te_l, F_te_l, M_te_l = [], [], [], []

    for X, D, F, M in zip(X_list, D_list, F_list, M_list):
        total = X.shape[0]
        tr = int(total * 0.8)
        va = int(total * 0.1)
        # te = rest
        
        # 2. 데이터 섞기 (Shuffle)
        indices = np.random.permutation(total)
        X, D, F, M = X[indices], D[indices], F[indices], M[indices]

        # 데이터 분할
        X_tr_l.append(X[:tr]);               D_tr_l.append(D[:tr]);               F_tr_l.append(F[:tr]);               M_tr_l.append(M[:tr])
        X_va_l.append(X[tr:tr+va]);          D_va_l.append(D[tr:tr+va]);          F_va_l.append(F[tr:tr+va]);          M_va_l.append(M[tr:tr+va])
        X_te_l.append(X[tr+va:]);            D_te_l.append(D[tr+va:]);            F_te_l.append(F[tr+va:]);            M_te_l.append(M[tr+va:])

    # 3. np.concatenate로 모든 데이터 합치기
    X_train = np.concatenate(X_tr_l, axis=0) if X_tr_l else np.zeros((0, cfg.max_cells, 6))
    D_train = np.concatenate(D_tr_l, axis=0); F_train = np.concatenate(F_tr_l, axis=0); M_train = np.concatenate(M_tr_l, axis=0)
    
    X_valid = np.concatenate(X_va_l, axis=0) if X_va_l else np.zeros((0, cfg.max_cells, 6))
    D_valid = np.concatenate(D_va_l, axis=0); F_valid = np.concatenate(F_va_l, axis=0); M_valid = np.concatenate(M_va_l, axis=0)
    
    X_test  = np.concatenate(X_te_l, axis=0) if X_te_l else np.zeros((0, cfg.max_cells, 6))
    D_test  = np.concatenate(D_te_l, axis=0); F_test  = np.concatenate(F_te_l, axis=0); M_test  = np.concatenate(M_te_l, axis=0)

    return X_train, D_train, F_train, M_train, X_valid, D_valid, F_valid, M_valid, X_test, D_test, F_test, M_test

def visualize_sample_paper(
    frequencies,
    udr_row,
    sdr_row,
    mask_row,
    frf_row=None,
    figsize=(6, 3),
    colors=dict(
        bandgap="#FFF2CC",      # 연노랑
        pass_band="#E6F2FF",    # 연한 파랑 (Pass Band)
        dont_care="#F2F2F2",    # 연한 회색 (Don't Care)
        uc_line="#004080",      # 진파랑
        sc_line="#8080FF",      # 연파랑
        frf_line="#666666",     # 회색
        defect_line="#CC0000",  # 빨강
    ),
    lw_uc=1.5, lw_sc=1.2, lw_frf=1.0, lw_defect=1.2,
    alpha_gap=0.4, alpha_bg=0.3,
    title=None,
    show_legend=True,
    save_path=None, 
    dpi=400,
    font_size=10,
):
    """Publication-ready visualization"""
    plt.rcParams.update({
        "font.family": "serif", 
        "font.size": font_size,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "savefig.transparent": True
    })

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    def _regions(mask_row, value):
        F = len(mask_row); out=[]; on=False; s=0
        for i in range(F):
            if (mask_row[i]==value) and not on:
                on=True; s=i
            if on and (mask_row[i]!=value or i == F-1):
                end_idx = i-1 if mask_row[i]!=value else i
                on=False; out.append((s, end_idx))
        return out
        
    def force_edges(S_new, thresh=0.15):
        S = S_new.copy()
        for i in range(1, len(S)-1):
            if S[i] >= S[i-1] and S[i] >= S[i+1] and (np.pi - S[i]) < thresh:
                S[i] = np.pi
            elif S[i] <= S[i-1] and S[i] <= S[i+1] and S[i] < thresh:
                S[i] = 0.0
        if S[0] < thresh: S[0] = 0.0
        elif (np.pi - S[0]) < thresh: S[0] = np.pi
        if S[-1] < thresh: S[-1] = 0.0
        elif (np.pi - S[-1]) < thresh: S[-1] = np.pi
        return S

    sdr_row = force_edges(sdr_row)
    udr_row = force_edges(udr_row)

    for s, e in _regions(mask_row, 1):
        ax.axhspan(frequencies[s], frequencies[e], color=colors["bandgap"], alpha=alpha_gap, lw=0, zorder=-1)
    for s, e in _regions(mask_row, 0):
        ax.axhspan(frequencies[s], frequencies[e], color=colors["pass_band"], alpha=alpha_bg, lw=0, zorder=-1)
    for s, e in _regions(mask_row, 3):
        ax.axhspan(frequencies[s], frequencies[e], color=colors["dont_care"], alpha=alpha_bg, lw=0, zorder=-1)

    ax.plot(udr_row, frequencies, color=colors["uc_line"], lw=lw_uc, label="Unit cell", zorder=1)
    ax.plot(sdr_row, frequencies, color=colors["sc_line"], lw=lw_sc, ls="--", label="Supercell", zorder=1)

    if frf_row is not None:
        ax2 = ax.twiny()
        ax2.plot(frf_row, frequencies, color=colors["frf_line"], lw=lw_frf, alpha=0.8, label="Transmittance", zorder=0)
        ax2.set_xlabel("Transmittance", color=colors["frf_line"], fontsize=font_size)
        ax2.tick_params(axis='x', labelcolor=colors["frf_line"], width=0.6, size=2.5)
        ax2.set_xlim(0, 1.05)
        ax2.spines['top'].set_color(colors["frf_line"])
        ax2.spines['top'].set_linewidth(0.8)
    ax.set_ylabel("Frequency", fontsize=font_size)
    ax.set_xlabel("Dispersion [rad]", fontsize=font_size)
    ax.set_ylim(frequencies[0], frequencies[-1]) 
    ax.set_xlim(0, np.pi) 
    ax.set_xticks([0, np.pi/2, np.pi])
    ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
    ax.grid(False)
    ax.tick_params(axis='y', width=0.8, size=3)
    
    if title:
        ax.set_title(title, fontsize=font_size + 1, pad=4)

    if show_legend:
        legend_elements = [
            Line2D([0], [0], color=colors["uc_line"], lw=lw_uc, label="Unit cell"),
            Line2D([0], [0], color=colors["sc_line"], lw=lw_sc, ls="--", label="Supercell"),
            Line2D([0], [0], color=colors["defect_line"], lw=lw_defect, label="Defect band"),
            Patch(facecolor=colors["bandgap"], alpha=alpha_gap, label="Bandgap"),
        ]
        if frf_row is not None:
            legend_elements.append(Line2D([0], [0], color=colors["frf_line"], lw=lw_frf, label="Transmittance"))
            
        ax.legend(handles=legend_elements, loc="upper right", frameon=False, fontsize=font_size - 1)

    plt.tight_layout(pad=0.3)
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", transparent=True, dpi=dpi)
        print(f"Visualization saved to {save_path}")
    
    plt.close()

def split_datasets(X_list, D_list, F_list, M_list, U_list, T_list, cfg):
    """
    List of Arrays -> Concat -> Shuffle -> Split (8:1:1) -> Return Arrays
    """
    X_tr_l, D_tr_l, F_tr_l, M_tr_l, U_tr_l, T_tr_l = [], [], [], [], [], []
    X_va_l, D_va_l, F_va_l, M_va_l, U_va_l, T_va_l = [], [], [], [], [], []
    X_te_l, D_te_l, F_te_l, M_te_l, U_te_l, T_te_l = [], [], [], [], [], []

    for X, D, F, M, U, T in zip(X_list, D_list, F_list, M_list, U_list, T_list):
        total = X.shape[0]
        tr = int(total * 0.8)
        va = int(total * 0.1)
        # te = rest
        
        # 2. 데이터 섞기 (Shuffle)
        indices = np.random.permutation(total)
        idx_tr = indices[:tr]
        idx_va = indices[tr:tr+va]
        idx_te = indices[tr+va:]

        # 데이터 분할 (Memory efficient slicing without full copy permutation first)
        X_tr_l.append(X[idx_tr]); D_tr_l.append(D[idx_tr]); F_tr_l.append(F[idx_tr]); M_tr_l.append(M[idx_tr]); U_tr_l.append(U[idx_tr])
        X_va_l.append(X[idx_va]); D_va_l.append(D[idx_va]); F_va_l.append(F[idx_va]); M_va_l.append(M[idx_va]); U_va_l.append(U[idx_va])
        X_te_l.append(X[idx_te]); D_te_l.append(D[idx_te]); F_te_l.append(F[idx_te]); M_te_l.append(M[idx_te]); U_te_l.append(U[idx_te])
        
        if T.shape[0] > 0:
            T_tr_l.append(T[idx_tr]); T_va_l.append(T[idx_va]); T_te_l.append(T[idx_te])
        else:
            T_tr_l.append(np.zeros((0, cfg.k_points))); T_va_l.append(np.zeros((0, cfg.k_points))); T_te_l.append(np.zeros((0, cfg.k_points)))

    # 3. np.concatenate로 모든 데이터 합치기
    # Helper to concat or empty
    # This helper is not used in the provided code, but the logic for empty lists is applied below.
    # def _concat(lst, feat_dim, is_1d=False):
    #     if not lst:
    #         if is_1d: return np.zeros((0,), dtype=np.float32)
    #         else: return np.zeros((0, feat_dim), dtype=np.float32)
    #     return np.concatenate(lst, axis=0)

    X_train = np.concatenate(X_tr_l, axis=0) if X_tr_l else np.zeros((0, cfg.max_cells, 6))
    D_train = np.concatenate(D_tr_l, axis=0) if D_tr_l else np.zeros((0, cfg.k_points))
    F_train = np.concatenate(F_tr_l, axis=0) if F_tr_l else np.zeros((0, cfg.k_points))
    M_train = np.concatenate(M_tr_l, axis=0) if M_tr_l else np.zeros((0, cfg.k_points), dtype=np.uint8)
    U_train = np.concatenate(U_tr_l, axis=0) if U_tr_l else np.zeros((0, cfg.k_points))
    T_train = np.concatenate(T_tr_l, axis=0) if T_tr_l else np.zeros((0, cfg.k_points))
    
    X_valid = np.concatenate(X_va_l, axis=0) if X_va_l else np.zeros((0, cfg.max_cells, 6))
    D_valid = np.concatenate(D_va_l, axis=0) if D_va_l else np.zeros((0, cfg.k_points))
    F_valid = np.concatenate(F_va_l, axis=0) if F_va_l else np.zeros((0, cfg.k_points))
    M_valid = np.concatenate(M_va_l, axis=0) if M_va_l else np.zeros((0, cfg.k_points), dtype=np.uint8)
    U_valid = np.concatenate(U_va_l, axis=0) if U_va_l else np.zeros((0, cfg.k_points))
    T_valid = np.concatenate(T_va_l, axis=0) if T_va_l else np.zeros((0, cfg.k_points))
    
    X_test  = np.concatenate(X_te_l, axis=0) if X_te_l else np.zeros((0, cfg.max_cells, 6))
    D_test  = np.concatenate(D_te_l, axis=0) if D_te_l else np.zeros((0, cfg.k_points))
    F_test  = np.concatenate(F_te_l, axis=0) if F_te_l else np.zeros((0, cfg.k_points))
    M_test  = np.concatenate(M_te_l, axis=0) if M_te_l else np.zeros((0, cfg.k_points), dtype=np.uint8)
    U_test  = np.concatenate(U_te_l, axis=0) if U_te_l else np.zeros((0, cfg.k_points))
    T_test  = np.concatenate(T_te_l, axis=0) if T_te_l else np.zeros((0, cfg.k_points))

    return (X_train, D_train, F_train, M_train, U_train, T_train,
            X_valid, D_valid, F_valid, M_valid, U_valid, T_valid,
            X_test,  D_test,  F_test,  M_test,  U_test,  T_test)


def build_cache_multimat(cfg):
    target_folders = getattr(cfg, "target_folders", ["CA", "SA", "TA"])
    print(f"[Cache] Building Multi-Material Cache from: {target_folders}")
    
    # Local lists for splitting logic input
    # But wait, logic below iterates folders.
    
    def safe_load_hdf5(dataset, limit, chunk_size=5000):
        # Determine actual size to read
        total_len = dataset.shape[0]
        read_len = min(limit, total_len)
        
        # If small enough, read directly
        if read_len <= chunk_size:
            return dataset[:read_len]
            
        # Chunked read
        chunks = []
        for i in range(0, read_len, chunk_size):
            end = min(read_len, i + chunk_size)
            chunks.append(dataset[i:end])
        return np.concatenate(chunks, axis=0)

    for folder_name in target_folders:
        folder_path = os.path.join(cfg.data_dir, folder_name)
        if not os.path.isdir(folder_path):
            print(f"Folder not found: {folder_path}, skipping.")
            continue

        files = sorted([f for f in os.listdir(folder_path) if f.endswith(".h5")])
        if not files:
            print(f"No .h5 files in {folder_name}, skipping.")
            continue
            
        print(f"--> Processing {folder_name} ({len(files)} files)...")
        
        # Local lists for this folder only
        X_local, D_local, F_local, M_local, U_local, T_local = [], [], [], [], [], []

        for filename in files:
            path = os.path.join(folder_path, filename)
            try:
                with h5py.File(path, "r") as hf:
                    limit = int(cfg.sample_size)
                    k_lim = int(cfg.k_points)
                    
                    design_variable = safe_load_hdf5(hf["design_variable"], limit)
                    udr_full = safe_load_hdf5(hf["dispersion_relation_unitcell"], limit)
                    udr_relation = udr_full[:, :k_lim]
                    
                    sdr_full = safe_load_hdf5(hf["dispersion_relation_supercell"], limit)
                    sdr_relation = sdr_full[:, :k_lim]
                    
                    freq_full = safe_load_hdf5(hf["frequencies"], limit)
                    frequencies = freq_full[:, :k_lim]
                    
                    # Transmittance is no longer needed
                    T_data = np.zeros((0, k_lim))
            
                    X6 = calculate_features(design_variable)
                    X6 = pad_or_truncate_X(X6, cfg.max_cells)
                    
                    # Pass SDR (udi, sdi) to build_band_mask with pre-allocated memory
                    M = np.zeros((udr_relation.shape[0], k_lim), dtype=np.int8)
                    for i in range(udr_relation.shape[0]):
                        build_band_mask(udr_relation[i], sdr_relation[i], out_mask=M[i])
                        
                        # Apply 'Don't Care' (Class 3) Masking for Inverse Design
                        # 0: Pass, 1: Gap, 2: Defect -> Keep only one contiguous Gap+Defect region
                        curr_mask = M[i]
                        gap_defect_mask = (curr_mask == 1) | (curr_mask == 2)
                        
                        # Find contiguous regions of gaps/defects
                        runs = _runs_bool(gap_defect_mask)
                        
                        # Filter runs: only keep those >= 5kHz (50 bins at 0.1kHz resolution)
                        valid_runs = []
                        for (start_idx, end_idx) in runs:
                            if (end_idx - start_idx + 1) >= 50:
                                valid_runs.append((start_idx, end_idx))
                        
                        # If there is at least one valid wide bandgap
                        if len(valid_runs) > 0:
                            p_single = getattr(cfg, "multi_bandgap_p_single", 0.5)
                            
                            # valid_runs >= 2이고 확률 분기에서 multi-gap이 선택된 경우
                            use_multi = (len(valid_runs) >= 2) and (np.random.rand() >= p_single)
                            
                            # Create a new mask filled with 3 (Don't Care)
                            new_mask = np.full_like(curr_mask, 3, dtype=np.int8)
                            
                            if use_multi:
                                # --- 2-gap 선택 ---
                                chosen = np.random.choice(len(valid_runs), size=2, replace=False)
                                # 주파수 순서로 정렬 (낮은 쪽 먼저)
                                chosen_runs = sorted([valid_runs[chosen[0]], valid_runs[chosen[1]]], key=lambda r: r[0])
                                
                                for (keep_start, keep_end) in chosen_runs:
                                    # Pass-band boundary constraint (앞뒤 20 bins 패딩)
                                    pad_start = max(0, keep_start - 20)
                                    pad_end   = min(k_lim, keep_end + 1 + 20)
                                    new_mask[pad_start:keep_start]  = 0
                                    new_mask[keep_end + 1:pad_end]  = 0
                                    # Gap 구간 원래 레이블 복원 (덮어쓰기로 gap 내부 보존)
                                    new_mask[keep_start:keep_end + 1] = curr_mask[keep_start:keep_end + 1]
                                
                                # 두 gap 사이 거리가 40 bins 미만이면 사이 구간도 Pass-band로
                                between_start = chosen_runs[0][1] + 1
                                between_end   = chosen_runs[1][0]
                                if between_end - between_start < 40:
                                    new_mask[between_start:between_end] = 0
                            else:
                                # --- 1-gap 선택 (기존 방식) ---
                                target_run_idx = np.random.randint(len(valid_runs))
                                keep_start, keep_end = valid_runs[target_run_idx]
                                
                                # Add 0-padding (Pass-band boundary constraint) 20 bins (2kHz) around the gap
                                pad_start = max(0, keep_start - 20)
                                pad_end = min(k_lim, keep_end + 1 + 20)
                                
                                # First, apply 0 to the padded region
                                new_mask[pad_start:keep_start] = 0
                                new_mask[keep_end+1:pad_end] = 0
                                
                                # Finally, restore the targeted gap and its defects
                                new_mask[keep_start:keep_end+1] = curr_mask[keep_start:keep_end+1]
                            
                            M[i] = new_mask
                        else:
                            # If no gaps >= 5kHz exist, everything is Don't Care
                            M[i] = np.full_like(curr_mask, 3, dtype=np.int8)

                    X_local.append(X6)
                    D_local.append(sdr_relation)
                    F_local.append(frequencies)
                    M_local.append(M)
                    U_local.append(udr_relation)
                    T_local.append(T_data)
                    
            except Exception as e:
                print(f"Error loading {path}: {e}")
                continue

        if not X_local:
            print(f"[Cache] No valid data in {folder_name}.")
            continue
            
        # Split & Save PER FOLDER
        print(f"--> Saving cache for {folder_name}...")
        (X_tr, D_tr, F_tr, M_tr, U_tr, T_tr,
         X_va, D_va, F_va, M_va, U_va, T_va,
         X_te, D_te, F_te, M_te, U_te, T_te) = split_datasets(X_local, D_local, F_local, M_local, U_local, T_local, cfg=cfg)

        os.makedirs(cfg.cache_dir, exist_ok=True)
        
        # 💡 [Memory Optimization] Downcast massive float matrices to uncompressed np.float16 to avoid zipfile deadlock
        # Save Dispersion Cache (X, U, D, M, F)
        np.savez(os.path.join(cfg.cache_dir, f"{folder_name}_train_dispersion.npz"), 
                 X=X_tr.astype(np.float16), U=U_tr.astype(np.float16), D=D_tr.astype(np.float16), M=M_tr, F=F_tr.astype(np.float16))
        
        np.savez(os.path.join(cfg.cache_dir, f"{folder_name}_valid_dispersion.npz"), 
                 X=X_va.astype(np.float16), U=U_va.astype(np.float16), D=D_va.astype(np.float16), M=M_va, F=F_va.astype(np.float16))
                 
        np.savez(os.path.join(cfg.cache_dir, f"{folder_name}_test_dispersion.npz"),  
                 X=X_te.astype(np.float16), U=U_te.astype(np.float16), D=D_te.astype(np.float16), M=M_te, F=F_te.astype(np.float16))

        # Not saving Transmittance Cache
        # np.savez_compressed(os.path.join(cfg.cache_dir, f"{folder_name}_train_transmittance.npz"), 
        #                     X=X_tr, T=T_tr, M=M_tr, F=F_tr)
        # np.savez_compressed(os.path.join(cfg.cache_dir, f"{folder_name}_valid_transmittance.npz"), 
        #                     X=X_va, T=T_va, M=M_va, F=F_va)
        # np.savez_compressed(os.path.join(cfg.cache_dir, f"{folder_name}_test_transmittance.npz"),  
        #                     X=X_te, T=T_te, M=M_te, F=F_te)
        
        # Clear memory
        del X_local, D_local, F_local, M_local, U_local, T_local
        import gc; gc.collect()

    print(f"Saved Split Material caches to {cfg.cache_dir}")
    return True


def _load_npz_6(path):
    """npz에서 가능한 키를 모두 안전하게 읽어오기"""
    try:
        with np.load(path) as data:
            keys = data.files
            X = data["X"] if "X" in keys else None
            D = data["D"] if "D" in keys else None
            F = data["F"] if "F" in keys else None
            M = data["M"] if "M" in keys else None
            U = data["U"] if "U" in keys else None
            T = data["T"] if "T" in keys else None
            return X, D, F, M, U, T 
    except Exception as e:
        print(f"[cache] Failed to load {path}: {e}")
        return None, None, None, None, None, None

def load_or_build_cache_multimat(cfg, strict=True):
    """
    Returns lists of paths.
    Now follows split file naming: _dispersion.npz and _transmittance.npz
    Primary path list returned is DISPERSION (for VAE/Diffusion).
    """
    os.makedirs(cfg.cache_dir, exist_ok=True)
    target_folders = getattr(cfg, "target_folders", ["CA", "SA", "TA"])
    
    # 1) 빌드 확인 (dispersion 파일 기준)
    missing = False
    for folder in target_folders:
        t_p = os.path.join(cfg.cache_dir, f"{folder}_train_dispersion.npz")
        if not os.path.isfile(t_p): 
            missing = True; break
            
    if missing:
        print("[Cache] Some caches missing. Building individual caches...")
        ok = build_cache_multimat(cfg)
        if not ok: raise RuntimeError("Cache build failed.")

    # 2) 경로 리스트 수집 (Return Dispersion paths primarily)
    train_paths, valid_paths, test_paths = [], [], []
    
    for folder in target_folders:
        train_paths.append(os.path.join(cfg.cache_dir, f"{folder}_train_dispersion.npz"))
        valid_paths.append(os.path.join(cfg.cache_dir, f"{folder}_valid_dispersion.npz"))
        test_paths.append(os.path.join(cfg.cache_dir, f"{folder}_test_dispersion.npz"))
        
    print(f"[Cache] Ready. {len(train_paths)} materials found (Dispersion).")
    return train_paths, valid_paths, test_paths

# ==============================================================================
# Helper functions for Feature construction (Differentiable)
# ==============================================================================

def zero_pad_layers(layers: torch.Tensor, valid_mask_1D: torch.Tensor) -> torch.Tensor:
    """layers [B,L,3] * valid_mask_1D [B,L] -> pad구간 0."""
    return layers * valid_mask_1D.unsqueeze(-1).to(layers.dtype)

def layers_to_six_features_torch(layers: torch.Tensor) -> torch.Tensor:
    """
    layers: [B,L,3] with columns [E_scaled, rho_scaled, length].
    Zero-padding rule: invalid cells produce zero features (no eps tricks).
    """
    modulus = layers[..., 0]
    density = layers[..., 1]
    length  = layers[..., 2]

    X1 = modulus
    X2 = density
    X3 = length

    mul_valid = (modulus > 0) & (density > 0)
    div_valid = mul_valid

    X4 = torch.zeros_like(modulus)
    X4[mul_valid] = torch.sqrt(modulus[mul_valid] * density[mul_valid])

    X5 = torch.zeros_like(modulus)
    X5[div_valid] = torch.sqrt(modulus[div_valid] / density[div_valid])

    X6 = torch.zeros_like(modulus)
    valid_X6 = div_valid & (X5 > 0)
    X6[valid_X6] = length[valid_X6] / X5[valid_X6]

    return torch.stack([X1, X2, X3, X4, X5, X6], dim=-1)  # [B,L,6]
