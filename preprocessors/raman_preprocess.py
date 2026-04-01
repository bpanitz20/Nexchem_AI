#Preprocessors

import numpy as np
import os
import matplotlib.pyplot as plt
import ramanspy
from ramanspy.preprocessing.misc import Cropper
from scipy.signal import savgol_filter
from ramanspy import Spectrum
import pandas as pd
from preprocessors.transforms import GlobalMeanCenterStep, SNVStep
from config import (DEFAULT_CROP_REGION, DEFAULT_SAVGOL_WINDOW,
                    DEFAULT_SAVGOL_POLYORDER, DEFAULT_DERIV_ORDER,
                    DEFAULT_ASLS_LAMBDA, DEFAULT_ASLS_P)


def snv_normalization(spectral_data):
    """
    Apply Standard Normal Variate (SNV) normalization to the given spectral data.
    SNV = (X - mean(X)) / std(X)
    
    Parameters:
        spectral_data (numpy.ndarray): The spectral data array to normalize.

    Returns:
        numpy.ndarray: The SNV-normalized spectral data.
    """
    # Center and scale each spectrum
    mean_spectrum = np.mean(spectral_data, axis=1).reshape(-1, 1)  # Calculate mean for each spectrum
    std_spectrum = np.std(spectral_data, axis=1).reshape(-1, 1)    # Calculate std for each spectrum
    
    # Apply SNV normalization
    snv_spectra = (spectral_data - mean_spectrum) / std_spectrum
    return snv_spectra


def mean_center(X):
    """
    Mean-center a 2D array across samples (rows).
    X: shape (n_samples, n_features)
    """
    mean_spectrum = np.mean(X, axis=0, keepdims=True)
    return X - mean_spectrum


def _als_baseline(y, lam=1e5, p=0.001, niter=10):
    """Asymmetric Least Squares baseline correction (Eilers & Boelens 2005).

    Parameters
    ----------
    y    : 1-D array — spectrum
    lam  : float     — smoothness (log₁₀ scale typical: 5.8 → lam≈6.3e5)
    p    : float     — asymmetry weight (0 < p < 1; small p → baseline below peaks)
    niter: int       — number of iterations
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * (D @ D.T)
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.diags(w, 0)
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def _compute_preprocess_savgol_snv_mc(sample_spectra, crop_region, derivative_order, use_state):
    """Pure computation: Crop → SavGol-deriv → SNVStep → GlobalMeanCenterStep. No I/O."""
    import copy
    from scipy.signal import savgol_filter

    spectra_copy = copy.deepcopy(sample_spectra)
    row_bank = []
    axis_ref = None

    cropper = Cropper(region=crop_region)
    snv_step = SNVStep()
    for sample_name, spectra in spectra_copy.items():
        processed_list = []
        for i, spectrum in enumerate(spectra):
            cropped = cropper.apply(spectrum)
            axis_ref = cropped.spectral_axis if axis_ref is None else axis_ref
            y = cropped.spectral_data
            y_diff = savgol_filter(y, polyorder=2, window_length=13, deriv=derivative_order)
            y_snv = snv_step.transform(y_diff[np.newaxis, :]).squeeze(0)
            row_bank.append(y_snv)
            processed_list.append(ramanspy.Spectrum(y_snv, cropped.spectral_axis))
        spectra_copy[sample_name] = processed_list

    X = np.vstack(row_bank)
    mc_step = GlobalMeanCenterStep()
    if use_state is not None and "mean_spectrum" in use_state:
        mc_step.mean_ = np.asarray(use_state["mean_spectrum"])
        X_mc = mc_step.transform(X)
    else:
        X_mc = mc_step.fit_transform(X)

    k = 0
    for sample_name, spectra in spectra_copy.items():
        for j in range(len(spectra)):
            spectra_copy[sample_name][j] = ramanspy.Spectrum(
                X_mc[k], spectra_copy[sample_name][j].spectral_axis)
            k += 1

    return spectra_copy, axis_ref, mc_step


def plot_preprocess_results_savgol_snv_mc(spectra_copy, spectra_dir):
    """Save per-sample and overlay diagnostic plots for preprocess_savgol_snv_mc."""
    os.makedirs(spectra_dir, exist_ok=True)

    for sample_name, spectra in spectra_copy.items():
        plt.figure(figsize=(8, 5))
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.7)
        plt.title(f"Processed: {sample_name}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.savefig(os.path.join(spectra_dir, f"{sample_name}_processed.png"), dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    for sample_name, spectra in spectra_copy.items():
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.6)
    plt.title("Overlay of All Preprocessed Spectra (Mean-Centered)")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.savefig(os.path.join(spectra_dir, "Overlay_Preprocessed.png"), dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_savgol_snv_mc(sample_spectra, spectra_dir, crop_region=DEFAULT_CROP_REGION,
                              derivative_order=DEFAULT_DERIV_ORDER,
                              return_state=False, use_state=None):
    """
    Preprocess Raman spectra with: Crop → SavGol-deriv → SNV → Global Mean-Center.
    Returns:
        spectra_copy (dict): same structure as input but with processed ramanspy.Spectrum objects
        cropped_axis (array-like): spectral axis after cropping
    """
    spectra_copy, cropped_axis, mc_step = _compute_preprocess_savgol_snv_mc(
        sample_spectra, crop_region, derivative_order, use_state)
    plot_preprocess_results_savgol_snv_mc(spectra_copy, spectra_dir)
    if return_state:
        return spectra_copy, cropped_axis, {"mean_spectrum": mc_step.mean_}
    return spectra_copy, cropped_axis
    


def _poly_baseline_lieber(spectrum, axis, order=4, niter=100):
    """Iterative polynomial baseline (Lieber & Mahadevan-Jansen 2005).

    Iteratively fits a polynomial constrained to remain <= the spectrum,
    yielding a conservative fluorescence / scatter baseline estimate.
    """
    x = 2.0 * (axis - axis.min()) / (axis.max() - axis.min()) - 1.0  # normalise to [-1, 1]
    y = spectrum.copy().astype(float)
    for _ in range(niter):
        coeffs = np.polyfit(x, y, order)
        poly = np.polyval(coeffs, x)
        y = np.minimum(spectrum, poly)  # constrain below original spectrum
    coeffs = np.polyfit(x, y, order)
    return np.polyval(coeffs, x)


def _emsc_correct_ref(spectra_matrix, reference, axis, p_order=6, interferents=None):
    """Extended Multiplicative Scatter Correction with an explicit reference spectrum.

    For each spectrum *s*, solves the linear system:
        s ≈ a·r + c₁·q₁ + … + cₖ·qₖ + b₀ + b₁·x + … + bₚ·xᵖ
    where qᵢ are optional interferent vectors (Liland 2016).
    Corrected spectrum: (s − interferent_part − polynomial_part) / a

    Parameters
    ----------
    spectra_matrix : ndarray, shape (n, m)
    reference      : ndarray, shape (m,)        — target reference spectrum
    axis           : ndarray, shape (m,)        — spectral axis
    p_order        : int                        — polynomial extension order
    interferents   : ndarray, shape (k, m) or None
                     Interferent loading vectors to orthogonalise out of each spectrum.

    Returns
    -------
    corrected : ndarray, shape (n, m)
    """
    axis = np.asarray(axis, dtype=float)
    x = 2.0 * (axis - axis.min()) / (axis.max() - axis.min()) - 1.0
    poly_terms = np.column_stack([x ** k for k in range(p_order + 1)])  # (m, p_order+1)

    if interferents is not None and len(interferents):
        # D = [reference | q₁ | … | qₖ | poly_terms]
        D = np.column_stack([reference, interferents.T, poly_terms])
    else:
        D = np.column_stack([reference, poly_terms])

    corrected = np.zeros_like(spectra_matrix, dtype=float)
    for i, s in enumerate(spectra_matrix):
        coeffs, _, _, _ = np.linalg.lstsq(D, s, rcond=None)
        a = coeffs[0]
        # Remove everything except a·reference
        non_ref_part = D[:, 1:] @ coeffs[1:]
        corrected[i] = (s - non_ref_part) / a if a != 0.0 else s - non_ref_part
    return corrected


def _compute_ref_baseline(spectrum, axis, method="lieber",
                           poly_ref_order=4, als_lam=6.31e5, als_p=0.01):
    """Apply baseline correction to a single reference spectrum.

    method : "lieber" — iterative polynomial (Lieber & Mahadevan-Jansen 2005)
             "als"    — Asymmetric Least Squares (Eilers & Boelens 2005)
    Returns the baseline-corrected spectrum (spectrum − baseline).
    """
    if method == "als":
        bl = _als_baseline(spectrum, lam=als_lam, p=als_p)
    else:
        bl = _poly_baseline_lieber(spectrum, axis, order=poly_ref_order)
    return spectrum - bl


def _build_reference_map(spectral_matrix, group_ids_bank, sample_groups, axis,
                          poly_ref_order, reference_mode, class_lookup, use_state,
                          ref_baseline_method="lieber", als_lam=6.31e5, als_p=0.01):
    """Return ({gid: reference_array}, state_extras_dict) for EMSC correction.

    For sample mode the reference is always computed fresh (training and prediction).
    For dataset/class modes, use_state is consulted first (prediction path).
    state_extras_dict contains what should be saved into the preprocessing state.
    """
    def _bl(spectrum):
        return _compute_ref_baseline(spectrum, axis, method=ref_baseline_method,
                                     poly_ref_order=poly_ref_order,
                                     als_lam=als_lam, als_p=als_p)

    # --- Sample mode: always fresh, nothing extra to store ---
    if reference_mode == "sample":
        group_spectra = {}
        for arr, gid in zip(spectral_matrix, group_ids_bank):
            group_spectra.setdefault(gid, []).append(arr)
        ref_map = {}
        for gid, arrs in group_spectra.items():
            ref_map[gid] = _bl(np.mean(arrs, axis=0))
        return ref_map, {}

    # --- Dataset / class modes: check use_state first (prediction path) ---
    if use_state is not None:
        if reference_mode == "dataset" and "emsc_reference" in use_state:
            ref = np.asarray(use_state["emsc_reference"], dtype=float)
            return {gid: ref for gid in sample_groups}, {}
        if reference_mode == "class" and "class_references" in use_state:
            stored = {k: np.asarray(v, dtype=float)
                      for k, v in use_state["class_references"].items()}
            fallback = next(iter(stored.values()))
            ref_map = {}
            for gid in sample_groups:
                cls = (class_lookup or {}).get(str(gid))
                ref_map[gid] = stored.get(cls, fallback)
            return ref_map, {}

    # --- Training path: compute fresh ---
    if reference_mode == "dataset":
        ref = _bl(np.mean(spectral_matrix, axis=0))
        return {gid: ref for gid in sample_groups}, {"emsc_reference": ref}

    if reference_mode == "class":
        class_spectra = {}
        for arr, gid in zip(spectral_matrix, group_ids_bank):
            cls = (class_lookup or {}).get(str(gid), "__unknown__")
            class_spectra.setdefault(cls, []).append(arr)
        class_refs = {}
        for cls, arrs in class_spectra.items():
            class_refs[cls] = _bl(np.mean(arrs, axis=0))
        fallback = next(iter(class_refs.values()))
        ref_map = {}
        for gid in sample_groups:
            cls = (class_lookup or {}).get(str(gid), "__unknown__")
            ref_map[gid] = class_refs.get(cls, fallback)
        return ref_map, {"class_references": class_refs}

    raise ValueError(f"Unknown reference_mode: {reference_mode!r}")


def _compute_group_preprocess_ref_emsc_mc(sample_spectra, sample_groups, crop_region,
                                           poly_ref_order, emsc_p_order, use_state,
                                           reference_mode="dataset", class_lookup=None,
                                           ref_baseline_method="lieber",
                                           als_lam=6.31e5, als_p=0.01,
                                           apply_sg_smooth=False, sg_window=9, sg_polyorder=2,
                                           apply_mean_center=True, n_interferents=0):
    """Crop → (optional SG smooth) → Reference-EMSC (2-pass if n_interferents > 0) →
    (optional GlobalMeanCenter) → Group Average.  No I/O."""
    row_bank = []
    group_ids_bank = []
    cropped_axis = None

    cropper = Cropper(region=crop_region)

    for group_id, sample_ids in sample_groups.items():
        for sid in sorted(sample_ids):
            spectra = sample_spectra.get(sid)
            if not spectra:
                print(f"⚠️ Sample '{sid}' missing or empty. Skipping.")
                continue
            for spectrum in spectra:
                cropped = cropper.apply(spectrum)
                if cropped_axis is None:
                    cropped_axis = cropped.spectral_axis
                y = cropped.spectral_data.copy()
                # Optional SavGol smoothing (deriv=0) before EMSC
                if apply_sg_smooth:
                    wl = sg_window if sg_window % 2 == 1 else sg_window + 1
                    wl = max(wl, sg_polyorder + 1)
                    y = savgol_filter(y, window_length=wl, polyorder=sg_polyorder, deriv=0)
                row_bank.append(y)
                group_ids_bank.append(group_id)

    if not row_bank:
        return {}, cropped_axis, {}, None, {}

    spectral_matrix = np.array(row_bank, dtype=float)
    axis = np.asarray(cropped_axis, dtype=float)

    # Build per-group reference map
    reference_map, state_extras = _build_reference_map(
        spectral_matrix, group_ids_bank, sample_groups, axis,
        poly_ref_order, reference_mode, class_lookup, use_state,
        ref_baseline_method=ref_baseline_method, als_lam=als_lam, als_p=als_p)

    # Pass 1 EMSC — reference + polynomial only
    emsc_pass1 = np.zeros_like(spectral_matrix, dtype=float)
    for i, (s, gid) in enumerate(zip(spectral_matrix, group_ids_bank)):
        emsc_pass1[i] = _emsc_correct_ref(
            s[np.newaxis], reference_map[gid], axis, p_order=emsc_p_order)[0]

    # Interferent correction — PCA on pass-1 residuals (Liland 2016)
    interferent_vecs = None
    if n_interferents > 0:
        if use_state is not None and "emsc_interferents" in use_state:
            # Prediction path: reuse training interferent vectors
            interferent_vecs = np.asarray(use_state["emsc_interferents"], dtype=float)
        else:
            # Training path: compute PCA on mean-centred pass-1 spectra
            from sklearn.decomposition import PCA
            residuals = emsc_pass1 - np.mean(emsc_pass1, axis=0)
            n_comp = min(n_interferents, residuals.shape[0] - 1, residuals.shape[1])
            pca = PCA(n_components=n_comp)
            pca.fit(residuals)
            interferent_vecs = pca.components_          # (n_comp, n_wavenumbers)
            state_extras["emsc_interferents"] = interferent_vecs.tolist()

    # Pass 2 EMSC — with interferent columns if computed
    emsc_corrected = np.zeros_like(spectral_matrix, dtype=float)
    for i, (s, gid) in enumerate(zip(spectral_matrix, group_ids_bank)):
        emsc_corrected[i] = _emsc_correct_ref(
            s[np.newaxis], reference_map[gid], axis,
            p_order=emsc_p_order, interferents=interferent_vecs)[0]

    # Optional global mean-centering
    mc_step = GlobalMeanCenterStep() if apply_mean_center else None
    if apply_mean_center:
        if use_state is not None and "mean_spectrum" in use_state:
            mc_step.mean_ = np.asarray(use_state["mean_spectrum"], dtype=float)
            X_out = mc_step.transform(emsc_corrected)
        else:
            X_out = mc_step.fit_transform(emsc_corrected)
    else:
        X_out = emsc_corrected

    # Group average
    grouped_temp = {gid: [] for gid in sample_groups.keys()}
    for y_row, gid in zip(X_out, group_ids_bank):
        grouped_temp[gid].append(y_row)

    group_avg_spectra = {}
    for gid, arr_list in grouped_temp.items():
        if not arr_list:
            continue
        arr = np.vstack(arr_list)
        group_avg_spectra[gid] = Spectrum(np.mean(arr, axis=0), cropped_axis)

    return group_avg_spectra, cropped_axis, grouped_temp, mc_step, state_extras


def plot_preprocess_results_group_ref_emsc_mc(grouped_temp, group_avg_spectra,
                                               cropped_axis, spectra_dir):
    """Save per-group replicate + average diagnostic plots. Returns group_plot_dict."""
    os.makedirs(spectra_dir, exist_ok=True)
    group_plot_dict = {}

    for gid, arr_list in grouped_temp.items():
        if not arr_list:
            continue
        arr = np.vstack(arr_list)
        avg_spectrum = group_avg_spectra.get(gid)
        avg = avg_spectrum.spectral_data if avg_spectrum is not None else np.mean(arr, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, y_proc in enumerate(arr):
            ax.plot(cropped_axis, y_proc, alpha=0.3)
        ax.plot(cropped_axis, avg, color="black", linewidth=2, label=f"{gid} Avg")
        ax.set_title(f"{gid} – Reference-EMSC + Global Mean-Centered")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.legend()
        fig.savefig(os.path.join(spectra_dir, f"{gid}_ref_emsc_avg.png"),
                    dpi=300, bbox_inches="tight")
        group_plot_dict[gid] = fig
        plt.close(fig)

    return group_plot_dict


def group_preprocess_ref_emsc_mc(sample_spectra, sample_groups, spectra_dir,
                                  crop_region=DEFAULT_CROP_REGION,
                                  poly_ref_order=4, emsc_p_order=6,
                                  reference_mode="dataset", class_lookup=None,
                                  ref_baseline_method="lieber",
                                  als_lam=6.31e5, als_p=0.01,
                                  apply_sg_smooth=False, sg_window=9, sg_polyorder=2,
                                  apply_mean_center=True, n_interferents=0,
                                  return_state=False, use_state=None):
    """
    Group-level preprocessing: Crop → Reference-EMSC → Global Mean-Center → Group Average.

    The EMSC reference spectrum is computed according to ``reference_mode``:
      - ``"dataset"``  — single reference from the global mean of all replicates (default)
      - ``"class"``    — one reference per class label (requires ``class_lookup`` dict)
      - ``"sample"``   — one reference per sample group (computed fresh each time)

    The reference is the group mean with a Lieber polynomial baseline removed
    (order ``poly_ref_order``). EMSC is solved via OLS with a ``emsc_p_order``-degree
    polynomial extension (Liland, 2016).

    Supports:
      - return_state=True  → returns state dict with mean_spectrum + reference(s)
      - use_state={…}      → reuses training reference and mean for prediction sets
    """
    group_avg_spectra, cropped_axis, grouped_temp, mc_step, state_extras = \
        _compute_group_preprocess_ref_emsc_mc(
            sample_spectra, sample_groups, crop_region,
            poly_ref_order, emsc_p_order, use_state,
            reference_mode=reference_mode, class_lookup=class_lookup,
            ref_baseline_method=ref_baseline_method,
            als_lam=als_lam, als_p=als_p,
            apply_sg_smooth=apply_sg_smooth, sg_window=sg_window, sg_polyorder=sg_polyorder,
            apply_mean_center=apply_mean_center, n_interferents=n_interferents)

    if not group_avg_spectra:
        return {}, cropped_axis, {}

    group_plot_dict = plot_preprocess_results_group_ref_emsc_mc(
        grouped_temp, group_avg_spectra, cropped_axis, spectra_dir)

    if return_state:
        full_state = {"emsc_ref_mode": reference_mode}
        if mc_step is not None:
            full_state["mean_spectrum"] = mc_step.mean_
        full_state.update(state_extras)
        return group_avg_spectra, cropped_axis, group_plot_dict, full_state
    return group_avg_spectra, cropped_axis, group_plot_dict


def _compute_group_preprocess_savgol_snv_mc(sample_spectra, sample_groups, crop_region,
                                             derivative_order, use_state):
    """Pure computation: Crop → SavGol-deriv → SNVStep → GlobalMeanCenterStep → Group Average. No I/O."""
    row_bank = []
    group_ids_bank = []
    cropped_axis = None

    cropper = Cropper(region=crop_region)
    snv_step = SNVStep()

    for group_id, sample_ids in sample_groups.items():
        for sid in sorted(sample_ids):
            spectra = sample_spectra.get(sid)
            if not spectra:
                print(f"⚠️ Sample '{sid}' missing or empty. Skipping.")
                continue
            for spectrum in spectra:
                cropped = cropper.apply(spectrum)
                if cropped_axis is None:
                    cropped_axis = cropped.spectral_axis
                y = cropped.spectral_data
                y_deriv = savgol_filter(y, polyorder=2, window_length=13, deriv=derivative_order)
                y_snv = snv_step.transform(y_deriv.reshape(1, -1)).flatten()
                row_bank.append(y_snv)
                group_ids_bank.append(group_id)

    if not row_bank:
        return {}, cropped_axis, {}, None

    X = np.vstack(row_bank)
    mc_step = GlobalMeanCenterStep()
    if use_state is not None and "mean_spectrum" in use_state:
        mc_step.mean_ = np.asarray(use_state["mean_spectrum"])
        X_mc = mc_step.transform(X)
    else:
        X_mc = mc_step.fit_transform(X)

    grouped_temp = {gid: [] for gid in sample_groups.keys()}
    for y_row, gid in zip(X_mc, group_ids_bank):
        grouped_temp[gid].append(y_row)

    group_avg_spectra = {}
    for gid, arr_list in grouped_temp.items():
        if not arr_list:
            continue
        arr = np.vstack(arr_list)
        avg = np.mean(arr, axis=0)
        group_avg_spectra[gid] = Spectrum(avg, cropped_axis)

    return group_avg_spectra, cropped_axis, grouped_temp, mc_step


def plot_preprocess_results_group_savgol_snv_mc(grouped_temp, group_avg_spectra,
                                                 cropped_axis, spectra_dir):
    """Save per-group replicate + average diagnostic plots. Returns group_plot_dict."""
    os.makedirs(spectra_dir, exist_ok=True)
    group_plot_dict = {}

    for gid, arr_list in grouped_temp.items():
        if not arr_list:
            continue
        arr = np.vstack(arr_list)
        avg_spectrum = group_avg_spectra.get(gid)
        avg = avg_spectrum.spectral_data if avg_spectrum is not None else np.mean(arr, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, y_proc in enumerate(arr):
            ax.plot(cropped_axis, y_proc, alpha=0.3)
        ax.plot(cropped_axis, avg, color="black", linewidth=2, label=f"{gid} Avg")
        ax.set_title(f"{gid} – SavGol Deriv + SNV + Global Mean-Centered")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.legend()
        fig.savefig(os.path.join(spectra_dir, f"{gid}_group_avg_meancentered.png"),
                    dpi=300, bbox_inches="tight")
        group_plot_dict[gid] = fig
        plt.close(fig)

    return group_plot_dict


def group_preprocess_savgol_snv_mc(sample_spectra, sample_groups, spectra_dir="Group_Preprocessed_Averages",
                                   crop_region=DEFAULT_CROP_REGION,
                                   derivative_order=DEFAULT_DERIV_ORDER,
                                   return_state=False, use_state=None):
    """
    Group-level preprocessing: Crop → SavGol-deriv → SNV → Global Mean-Center → Group Average.

    Supports:
      - return_state=True → returns {"mean_spectrum": ...}
      - use_state={"mean_spectrum": ...} → reuses training-set mean
    """
    group_avg_spectra, cropped_axis, grouped_temp, mc_step = \
        _compute_group_preprocess_savgol_snv_mc(
            sample_spectra, sample_groups, crop_region, derivative_order, use_state)

    if mc_step is None:
        return {}, cropped_axis, {}

    group_plot_dict = plot_preprocess_results_group_savgol_snv_mc(
        grouped_temp, group_avg_spectra, cropped_axis, spectra_dir)

    if return_state:
        return group_avg_spectra, cropped_axis, group_plot_dict, {"mean_spectrum": mc_step.mean_}
    return group_avg_spectra, cropped_axis, group_plot_dict


def avg_y_block(gcms_df):
    """
    Average the Y block (GC-MS or metadata) across replicate samples by group ID.

    Parameters:
    -----------
    gcms_df : pd.DataFrame
        Original target dataframe with replicate sample IDs as index (e.g., '1-1', '1-2').

    Returns:
    --------
    pd.DataFrame
        Group-averaged GC-MS data (numeric columns averaged, categorical columns taken from first replicate),
        indexed by group ID (e.g., '1', '2').
    """
    df = gcms_df.copy()

    # Create 'Group' column from the part before dash
    df['Group'] = df.index.to_series().apply(lambda sid: sid.split("-")[0])

    # Separate numeric and non-numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    non_numeric_cols = [col for col in df.columns if col not in numeric_cols and col != 'Group']

    # Average numeric columns by group
    averaged_numeric = df.groupby("Group")[numeric_cols].mean()

    # Take the first non-numeric entry per group (e.g., for 'Class' or other labels)
    first_non_numeric = df.groupby("Group")[non_numeric_cols].first()

    # Combine them
    grouped_df = pd.concat([averaged_numeric, first_non_numeric], axis=1)

    return grouped_df

def _compute_preprocess_none(sample_spectra, crop_region):
    """Pure computation: Crop only. No I/O."""
    import copy

    spectra_copy = copy.deepcopy(sample_spectra)
    cropper = Cropper(region=crop_region)

    for sample_name, spectra in spectra_copy.items():
        if not isinstance(spectra, list):
            spectra = [spectra]
        cropped_spectra = []
        for spectrum in spectra:
            cropped_spectrum = cropper.apply(spectrum)
            cropped_spectra.append(
                ramanspy.Spectrum(cropped_spectrum.spectral_data, cropped_spectrum.spectral_axis)
            )
        spectra_copy[sample_name] = cropped_spectra

    first_sample = next(iter(spectra_copy.values()))
    cropped_axis = np.asarray(first_sample[0].spectral_axis)
    return spectra_copy, cropped_axis


def plot_preprocess_results_none(spectra_copy, spectra_dir):
    """Save per-sample and overlay diagnostic plots for preprocess_none."""
    os.makedirs(spectra_dir, exist_ok=True)

    for sample_name, spectra in spectra_copy.items():
        plt.figure(figsize=(8, 5))
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.7)
        plt.title(f"Raw (Cropped): {sample_name}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity (a.u.)")
        plt.tight_layout()
        plt.savefig(os.path.join(spectra_dir, f"{sample_name}_raw_cropped.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    for sample_name, spectra in spectra_copy.items():
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.6)
    plt.title("Overlay of All Raw (Cropped) Spectra")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity (a.u.)")
    plt.tight_layout()
    plt.savefig(os.path.join(spectra_dir, "Overlay_Raw_Cropped.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_none(sample_spectra, spectra_dir, crop_region=DEFAULT_CROP_REGION):
    """
    Crop-only preprocessor ("None"): keeps raw intensities, only trims the spectral axis.

    Parameters
    ----------
    sample_spectra : dict[str, list[ramanspy.Spectrum]]
        Mapping sample name -> list of Spectrum objects (replicates).
    spectra_dir : str
        Directory to save per-sample and overlay plots.
    crop_region : tuple(int, int)
        (min_cm1, max_cm1) cropping window.

    Returns
    -------
    spectra_copy : dict[str, list[ramanspy.Spectrum]]
        Cropped spectra in the same structure as input.
    cropped_axis : np.ndarray
        The cropped spectral axis (cm⁻¹).
    """
    spectra_copy, cropped_axis = _compute_preprocess_none(sample_spectra, crop_region)
    plot_preprocess_results_none(spectra_copy, spectra_dir)
    return spectra_copy, cropped_axis

def _compute_preprocess_asls_savgol_snv(sample_spectra, crop_region, asls_lambda,
                                         asls_p, sg_window, sg_polyorder):
    """Pure computation: Crop → AsLS-baseline → SavGol-smooth → SNVStep. No I/O."""
    import copy

    spectra_copy = copy.deepcopy(sample_spectra)
    cropper = Cropper(region=crop_region)
    snv_step = SNVStep()
    axis_ref = None

    for sample_name, spectra in spectra_copy.items():
        processed_list = []
        for spectrum in spectra:
            cropped = cropper.apply(spectrum)
            if axis_ref is None:
                axis_ref = cropped.spectral_axis
            y = cropped.spectral_data
            baseline = _als_baseline(y, lam=asls_lambda, p=asls_p)
            y_corr = y - baseline
            wl = min(sg_window, len(y_corr) if len(y_corr) % 2 == 1 else len(y_corr) - 1)
            if wl < 3:
                wl = 3
            if wl % 2 == 0:
                wl += 1
            y_smooth = savgol_filter(y_corr, window_length=wl, polyorder=sg_polyorder, deriv=0)
            y_snv = snv_step.transform(y_smooth[np.newaxis, :]).squeeze(0)
            processed_list.append(ramanspy.Spectrum(y_snv, cropped.spectral_axis))
        spectra_copy[sample_name] = processed_list

    return spectra_copy, axis_ref


def plot_preprocess_results_asls_savgol_snv(spectra_copy, spectra_dir):
    """Save per-sample and overlay diagnostic plots for preprocess_asls_savgol_snv."""
    os.makedirs(spectra_dir, exist_ok=True)

    for sample_name, spectra in spectra_copy.items():
        plt.figure(figsize=(8, 5))
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.7)
        plt.title(f"Processed (Crop + AsLS + SG + SNV): {sample_name}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.tight_layout()
        plt.savefig(os.path.join(spectra_dir, f"{sample_name}_processed_AsLS_SG_SNV.png"),
                    dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    for sample_name, spectra in spectra_copy.items():
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.6)
    plt.title("Overlay: Crop + AsLS + Savgol + SNV")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.tight_layout()
    plt.savefig(os.path.join(spectra_dir, "Overlay_Preprocessed_AsLS_SG_SNV.png"),
                dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_asls_savgol_snv(
        sample_spectra,
        spectra_dir,
        crop_region=DEFAULT_CROP_REGION,
        asls_lambda=DEFAULT_ASLS_LAMBDA,
        asls_p=DEFAULT_ASLS_P,
        sg_window=DEFAULT_SAVGOL_WINDOW,
        sg_polyorder=DEFAULT_SAVGOL_POLYORDER):
    """
    Preprocess Raman spectra with:
        1. Crop
        2. Asymmetric Least Squares (AsLS) baseline correction
        3. Savitzky-Golay smoothing
        4. SNV normalization

    NO MEAN-CENTERING.
    All steps are per-spectrum (no fitted global state).

    Parameters
    ----------
    sample_spectra : dict[str, list[ramanspy.Spectrum]]
    spectra_dir : str
        Directory to save diagnostic plots.
    crop_region : (float, float)
        Region for cropping the Raman shift axis.
    asls_lambda : float
        Smoothness parameter for AsLS.
    asls_p : float
        Asymmetry parameter for AsLS (0 < p < 1).
    sg_window : int
        Savitzky–Golay window length (must be odd, <= number of points).
    sg_polyorder : int
        Polynomial order for Savitzky–Golay filter.

    Returns
    -------
    spectra_copy : dict
        Same structure as input but with processed ramanspy.Spectrum objects.
    cropped_axis : np.ndarray
        Spectral axis after cropping.
    """
    spectra_copy, axis_ref = _compute_preprocess_asls_savgol_snv(
        sample_spectra, crop_region, asls_lambda, asls_p, sg_window, sg_polyorder)
    plot_preprocess_results_asls_savgol_snv(spectra_copy, spectra_dir)
    return spectra_copy, axis_ref


def _compute_group_preprocess_asls_savgol_snv(sample_spectra, sample_groups, crop_region,
                                               asls_lambda, asls_p, sg_window, sg_polyorder):
    """Pure computation: Crop → AsLS → SavGol-smooth → SNV → Group Average. No I/O."""
    row_bank = []
    group_ids_bank = []
    cropped_axis = None

    cropper = Cropper(region=crop_region)
    snv_step = SNVStep()

    for group_id, sample_ids in sample_groups.items():
        for sid in sorted(sample_ids):
            spectra = sample_spectra.get(sid)
            if not spectra:
                print(f"⚠️ Sample '{sid}' missing or empty. Skipping.")
                continue
            for spectrum in spectra:
                cropped = cropper.apply(spectrum)
                if cropped_axis is None:
                    cropped_axis = cropped.spectral_axis
                y = cropped.spectral_data
                baseline = _als_baseline(y, lam=asls_lambda, p=asls_p)
                y_corr = y - baseline
                wl = min(sg_window, len(y_corr) if len(y_corr) % 2 == 1 else len(y_corr) - 1)
                if wl < 3:
                    wl = 3
                if wl % 2 == 0:
                    wl += 1
                y_smooth = savgol_filter(y_corr, window_length=wl, polyorder=sg_polyorder, deriv=0)
                y_snv = snv_step.transform(y_smooth[np.newaxis, :]).squeeze(0)
                row_bank.append(y_snv)
                group_ids_bank.append(group_id)

    if not row_bank:
        return {}, cropped_axis, {}

    grouped_temp = {gid: [] for gid in sample_groups.keys()}
    for y_row, gid in zip(row_bank, group_ids_bank):
        grouped_temp[gid].append(y_row)

    group_avg_spectra = {}
    for gid, arr_list in grouped_temp.items():
        if not arr_list:
            continue
        avg = np.mean(np.vstack(arr_list), axis=0)
        group_avg_spectra[gid] = Spectrum(avg, cropped_axis)

    return group_avg_spectra, cropped_axis, grouped_temp


def plot_preprocess_results_group_asls_savgol_snv(grouped_temp, group_avg_spectra,
                                                   cropped_axis, spectra_dir):
    """Save per-group replicate + average diagnostic plots. Returns group_plot_dict."""
    os.makedirs(spectra_dir, exist_ok=True)
    group_plot_dict = {}

    for gid, arr_list in grouped_temp.items():
        if not arr_list:
            continue
        arr = np.vstack(arr_list)
        avg_spectrum = group_avg_spectra.get(gid)
        avg = avg_spectrum.spectral_data if avg_spectrum is not None else np.mean(arr, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for y_proc in arr:
            ax.plot(cropped_axis, y_proc, alpha=0.3)
        ax.plot(cropped_axis, avg, color="black", linewidth=2, label=f"{gid} Avg")
        ax.set_title(f"{gid} – AsLS + SavGol + SNV")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.legend()
        fig.savefig(os.path.join(spectra_dir, f"{gid}_asls_sg_snv_avg.png"),
                    dpi=300, bbox_inches="tight")
        group_plot_dict[gid] = fig
        plt.close(fig)

    return group_plot_dict


def group_preprocess_asls_savgol_snv(sample_spectra, sample_groups, spectra_dir,
                                      crop_region=DEFAULT_CROP_REGION,
                                      asls_lambda=DEFAULT_ASLS_LAMBDA,
                                      asls_p=DEFAULT_ASLS_P,
                                      sg_window=DEFAULT_SAVGOL_WINDOW,
                                      sg_polyorder=DEFAULT_SAVGOL_POLYORDER):
    """
    Group-level preprocessing: Crop → AsLS → SavGol-smooth → SNV → Group Average.

    Stateless — all steps are per-spectrum; no global mean-centering.

    Parameters
    ----------
    sample_spectra : dict[str, list[ramanspy.Spectrum]]
    sample_groups  : dict[group_id, list[sample_id]]
    spectra_dir    : str

    Returns
    -------
    group_avg_spectra : dict[group_id, ramanspy.Spectrum]
    cropped_axis      : np.ndarray
    group_plot_dict   : dict[group_id, matplotlib.Figure]
    """
    group_avg_spectra, cropped_axis, grouped_temp = _compute_group_preprocess_asls_savgol_snv(
        sample_spectra, sample_groups, crop_region,
        asls_lambda, asls_p, sg_window, sg_polyorder)

    if not group_avg_spectra:
        return {}, cropped_axis, {}

    group_plot_dict = plot_preprocess_results_group_asls_savgol_snv(
        grouped_temp, group_avg_spectra, cropped_axis, spectra_dir)

    return group_avg_spectra, cropped_axis, group_plot_dict


# ---------------------------------------------------------------------------
# Backward-compatibility aliases — old names are preserved; do not remove yet
# ---------------------------------------------------------------------------
preprocess_pipeline_2        = preprocess_savgol_snv_mc
group_preprocess_2           = group_preprocess_savgol_snv_mc
preprocess_pipeline_AsLS_SNV = preprocess_asls_savgol_snv
