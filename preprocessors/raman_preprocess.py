#Preprocessors

import numpy as np
import os
import matplotlib.pyplot as plt
from ramanspy.preprocessing.denoise import SavGol
from sklearn.pipeline import make_pipeline
import chemometrics as cm
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
    


def _compute_preprocess_savgol_emsc_mc(sample_spectra, crop_region, savgol_window,
                                        savgol_polyorder, emsc_p_order, emsc_asym_factor,
                                        deriv_order, use_state):
    """Pure computation: Crop → SavGol → EMSC → GlobalMeanCenterStep. No I/O."""
    import copy

    spectra_copy = copy.deepcopy(sample_spectra)
    row_bank = []
    axis_ref = None
    cropped_spectra_dict = {}   # {sample_name: [Spectrum, ...]} for diagnostic plots

    cropper = Cropper(region=crop_region)
    savgol = SavGol(window_length=savgol_window, polyorder=savgol_polyorder, deriv=deriv_order)
    emsc_pipeline = make_pipeline(
        cm.Emsc(p_order=emsc_p_order, normalize=False, algorithm='als', asym_factor=emsc_asym_factor)
    )

    for sample_name, spectra in spectra_copy.items():
        cropped_spectra = [cropper.apply(spectrum) for spectrum in spectra]
        if axis_ref is None:
            axis_ref = cropped_spectra[0].spectral_axis
        cropped_axis = axis_ref
        cropped_spectra_dict[sample_name] = cropped_spectra

        smoothed_spectra = savgol.apply(cropped_spectra)
        spectral_matrix = np.array([s.spectral_data for s in smoothed_spectra])
        emsc_corrected = emsc_pipeline.fit_transform(spectral_matrix)

        final_spectra = []
        for i, data in enumerate(emsc_corrected):
            final_spectra.append(ramanspy.Spectrum(data, cropped_axis))
            row_bank.append(data)
        spectra_copy[sample_name] = final_spectra

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
            spectra[j].spectral_data = X_mc[k]
            k += 1

    return spectra_copy, axis_ref, mc_step, cropped_spectra_dict


def plot_preprocess_results_savgol_emsc_mc(spectra_copy, cropped_spectra_dict, spectra_dir):
    """Save per-sample cropped, EMSC+MC, and overlay diagnostic plots."""
    os.makedirs(spectra_dir, exist_ok=True)

    for sample_name, cropped_spectra in cropped_spectra_dict.items():
        plt.figure(figsize=(8, 5))
        for spec in cropped_spectra:
            plt.plot(spec.spectral_axis, spec.spectral_data, color="blue", alpha=0.4)
        plt.title(f"Cropped Spectra: {sample_name}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.savefig(os.path.join(spectra_dir, f"{sample_name}_Cropped.png"), dpi=300, bbox_inches="tight")
        plt.close()

    for sample_name, spectra in spectra_copy.items():
        plt.figure(figsize=(8, 5))
        for spec in spectra:
            plt.plot(spec.spectral_axis, spec.spectral_data, color="red", alpha=0.8)
        plt.title(f"EMSC + Global Mean Centered: {sample_name}")
        plt.xlabel("Raman Shift (cm⁻¹)")
        plt.ylabel("Intensity")
        plt.savefig(os.path.join(spectra_dir, f"{sample_name}_EMSCpreprocessed.png"), dpi=300, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(8, 5))
    for sample_name, spectra in spectra_copy.items():
        for s in spectra:
            plt.plot(s.spectral_axis, s.spectral_data, alpha=0.6)
    plt.title("Overlay of All Preprocessed Spectra (Global Mean-Centered)")
    plt.xlabel("Raman Shift (cm⁻¹)")
    plt.ylabel("Intensity")
    plt.savefig(os.path.join(spectra_dir, "Overlay_Preprocessed.png"), dpi=300, bbox_inches="tight")
    plt.close()


def preprocess_savgol_emsc_mc(sample_spectra, spectra_dir, crop_region=(500, 1800),
                               savgol_window=13, savgol_polyorder=2, emsc_p_order=2,
                               emsc_asym_factor=0.1, deriv_order=1,
                               return_state=False, use_state=None):
    """
    Preprocess Raman spectra with: Crop → SavGol → EMSC → Global Mean-Center.
    Returns:
        spectra_copy (dict), cropped_axis
    """
    spectra_copy, axis_ref, mc_step, cropped_spectra_dict = _compute_preprocess_savgol_emsc_mc(
        sample_spectra, crop_region, savgol_window, savgol_polyorder,
        emsc_p_order, emsc_asym_factor, deriv_order, use_state)
    plot_preprocess_results_savgol_emsc_mc(spectra_copy, cropped_spectra_dict, spectra_dir)
    if return_state:
        return spectra_copy, axis_ref, {"mean_spectrum": mc_step.mean_}
    return spectra_copy, axis_ref




def _compute_group_preprocess_savgol_emsc_mc(sample_spectra, sample_groups, crop_region,
                                              emsc_p_order, emsc_asym_factor, deriv_order,
                                              use_state):
    """Pure computation: Crop → SavGol → EMSC → GlobalMeanCenterStep → Group Average. No I/O."""
    group_avg_spectra = {}
    group_replicates = {}
    cropped_axis = None
    row_bank = []
    idx_bank = []

    for group_id, sample_ids in sample_groups.items():
        for sid in sorted(sample_ids):
            spectra = sample_spectra.get(sid)
            if not spectra:
                print(f"⚠️ Sample '{sid}' missing or has no spectra. Skipping.")
                continue
            for spectrum in spectra:
                cropper = Cropper(region=crop_region)
                cropped = cropper.apply(spectrum)
                if cropped_axis is None:
                    cropped_axis = cropped.spectral_axis
                savgol = SavGol(window_length=9, polyorder=2, deriv=deriv_order)
                smoothed = savgol.apply([cropped])[0]
                row_bank.append(smoothed.spectral_data)
                idx_bank.append(group_id)

    if not row_bank:
        return {}, cropped_axis, {}, {}, None

    spectral_matrix = np.array(row_bank)
    pipeline = make_pipeline(
        cm.Emsc(p_order=emsc_p_order, background=None, normalize=False,
                algorithm='als', asym_factor=emsc_asym_factor)
    )
    emsc_corrected = pipeline.fit_transform(spectral_matrix)

    mc_step = GlobalMeanCenterStep()
    if use_state is not None and "mean_spectrum" in use_state:
        mc_step.mean_ = np.asarray(use_state["mean_spectrum"])
        emsc_centered = mc_step.transform(emsc_corrected)
    else:
        emsc_centered = mc_step.fit_transform(emsc_corrected)

    group_data_temp = {gid: [] for gid in sample_groups.keys()}
    for y, gid in zip(emsc_centered, idx_bank):
        group_data_temp[gid].append(y)

    for group_id, spectra_arrs in group_data_temp.items():
        if not spectra_arrs:
            continue
        group_array = np.array(spectra_arrs)
        group_avg = np.mean(group_array, axis=0)
        group_avg_spectra[group_id] = [Spectrum(group_avg, cropped_axis)]
        group_replicates[group_id] = [Spectrum(y, cropped_axis) for y in group_array]

    return group_avg_spectra, cropped_axis, group_replicates, group_data_temp, mc_step


def plot_preprocess_results_group_savgol_emsc_mc(group_data_temp, group_avg_spectra,
                                                  cropped_axis, spectra_dir):
    """Save per-group replicate + average diagnostic plots. Returns group_plot_dict."""
    os.makedirs(spectra_dir, exist_ok=True)
    group_plot_dict = {}

    for group_id, spectra_arrs in group_data_temp.items():
        if not spectra_arrs:
            continue
        group_array = np.array(spectra_arrs)
        avg_spectrum = group_avg_spectra.get(group_id)
        group_avg = avg_spectrum[0].spectral_data if avg_spectrum else np.mean(group_array, axis=0)

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, y in enumerate(group_array):
            ax.plot(cropped_axis, y, alpha=0.3, label=f"{group_id}-{i+1}")
        ax.plot(cropped_axis, group_avg, color="black", linewidth=2, label=f"{group_id} Avg")
        ax.set_title(f"Group {group_id} - SavGol Deriv + EMSC + Global Mean Centered")
        ax.set_xlabel("Raman Shift (cm⁻¹)")
        ax.set_ylabel("Intensity")
        ax.legend()
        fig.savefig(os.path.join(spectra_dir, f"{group_id}_emsc_avg.png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        group_plot_dict[group_id] = fig

    return group_plot_dict


def group_preprocess_savgol_emsc_mc(sample_spectra, sample_groups, spectra_dir,
                                    crop_region=(500, 1800),
                                    emsc_p_order=6, emsc_asym_factor=0.1,
                                    deriv_order=1,
                                    return_state=False, use_state=None):
    """
    Group-level preprocessing: Crop → SavGol-deriv → EMSC → Global Mean-Center → Group Average.
    """
    group_avg_spectra, cropped_axis, group_replicates, group_data_temp, mc_step = \
        _compute_group_preprocess_savgol_emsc_mc(
            sample_spectra, sample_groups, crop_region,
            emsc_p_order, emsc_asym_factor, deriv_order, use_state)

    if mc_step is None:
        return {}, cropped_axis, {}, {}

    group_plot_dict = plot_preprocess_results_group_savgol_emsc_mc(
        group_data_temp, group_avg_spectra, cropped_axis, spectra_dir)

    if return_state:
        return group_avg_spectra, cropped_axis, group_replicates, group_plot_dict, \
               {"mean_spectrum": mc_step.mean_}
    return group_avg_spectra, cropped_axis, group_replicates, group_plot_dict


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
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    def asls_baseline(y, lam=1e5, p=0.001, niter=10):
        """Asymmetric Least Squares baseline correction. Eilers & Boelens (2005)."""
        L = y.size
        D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L-2))
        D = lam * (D @ D.T)
        w = np.ones(L)
        for _ in range(niter):
            W = sparse.diags(w, 0)
            Z = W + D
            z = spsolve(Z, w * y)
            w = p * (y > z) + (1 - p) * (y < z)
        return z

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
            baseline = asls_baseline(y, lam=asls_lambda, p=asls_p)
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


# ---------------------------------------------------------------------------
# Backward-compatibility aliases — old names are preserved; do not remove yet
# ---------------------------------------------------------------------------
preprocess_pipeline_2        = preprocess_savgol_snv_mc
preprocess_pipeline_1        = preprocess_savgol_emsc_mc
group_preprocess             = group_preprocess_savgol_emsc_mc
group_preprocess_2           = group_preprocess_savgol_snv_mc
preprocess_pipeline_AsLS_SNV = preprocess_asls_savgol_snv
