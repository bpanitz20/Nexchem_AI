#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:39:25 2025

@author: bp
"""

import streamlit as st
import io
import pandas as pd
import numpy as np
import zipfile
import os
import tempfile
from loaders.raman_loader import load_raman
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import re
from plotting.plot_raw import plot_spectra_colored_by_analyte
from plotting.plot_regression import plot_pred_vs_actual_interactive, plot_pred_vs_actual_journal, _build_cv_figures, _build_pred_vs_actual_fig, plot_pred_vs_actual_paper, plot_analyte_correlation_map
from preprocessors.raman_preprocess import (
    preprocess_savgol_snv_mc,
    group_preprocess_savgol_snv_mc,
    group_preprocess_ref_emsc_mc,
    group_preprocess_asls_savgol_snv,
    avg_y_block,
    preprocess_none,
    preprocess_asls_savgol_snv,
)
from collections import defaultdict
from pathlib import Path
from config import DEFAULT_CROP_REGION
from utils.pdf_export import figures_to_pdf_bytes, image_paths_to_pdf_bytes, pdf_paths_to_pdf_bytes

# ---------------------------------------------------------------------------
# Preprocessing options — single definition shared by Preprocessing and
# Prediction tabs.  Add new methods here; both tabs pick them up automatically.
# ---------------------------------------------------------------------------
PREPROCESS_OPTIONS = {
    "1. Savgol-SNV-MeanCenter": preprocess_savgol_snv_mc,
    "2. Baseline-Smooth-SNV": preprocess_asls_savgol_snv,
    "3. Average Replicates: EMSC": group_preprocess_ref_emsc_mc,
    "4. None": preprocess_none,
}


def ensure_class_colors_from_y(y_df, class_col):
    """Return (class_by_id, color_by_class) and works with either sample IDs or group IDs as index."""
    classes = y_df[class_col].astype(str)
    class_by_id = classes.to_dict()  # {id -> class label}

    uniq = sorted(classes.unique())
    cmap = plt.cm.get_cmap('tab20', len(uniq))
    color_by_class = {cls: cmap(i) for i, cls in enumerate(uniq)}  # RGBA tuples OK for plt
    return class_by_id, color_by_class



st.set_page_config(page_title="NexChem App", layout="wide")


# === Sidebar Tabs ===
tab = st.sidebar.radio("Navigation", ["Home", "Data Loading", "Preprocessing", "Modeling", "Prediction", "PCA"], index=0)

# === TAB 0: Home / Instructions ===
if tab == "Home":
    st.title("🔬 NexChem Chemometric App")
    st.markdown("""
    #### Intelligent Chemometric Modeling Platform
    All-in-one app for developing, validating, and deploying predictive models from spectroscopy data.

    ---

    ### User Guide
    Follow the tabs on the left side of the page. Below are instructions for each step:

    **1️⃣ Load Calibration Data**
    - Choose an output directory to save spectra and results
    - Upload a `.zip` containing your Thermo **Raman spectra (.spc)** files.  
    - Upload an **Excel Y-block** containing target concentrations.  
      - Make sure samples use the correct naming convention (see bottom).  
      - Ensure sample names in the spectra and Excel **ID** column match exactly.  
      - Include a class column for visualization, cross validation, and prediction set hold out 
    - Visualize raw data to verify spectra were loaded correctly.
    - Choose samples to remove or hold out as a separate prediction set

    **2️⃣ Preprocess Spectra**
    - Choose one of the preprocessing pipelines (e.g., Savitzky-Golay, SNV, EMSC).  
    - Adjust parameters or leave defaults.  
    - Visualize preprocessed spectra.  
    
    **3️⃣ Modeling**
    - Choose classification or regression
    - Choose model type and parameters or leave blank to use automatic selection.  
    - Choose Cross Validation parameters or use defaults.
    - Some models allow variable selection  
    - Visualize predicted vs. actual plots and model performance metrics.

    **4️⃣ Prediction Tab**
    - Use with regression models
    - Upload new Raman spectra to generate predictions from your saved calibration model or use samples removed in step 1.  
    - If you upload a Y-block for the prediction set, external predicted vs. actual plots will be generated.

    **5️⃣ PCA Tab**
    - Generates a PCA plot using your spectral data or y block analytes
    - Select which principal components (PCs) to display and view loadings.  
    - Uses the **Class** column from the Y-block for grouping and coloring.
    - PCA-DA also available

    ---

    ### File Naming Convention
    Raman spectra should follow this format:
    ```
    SampleID-Replicate_.spc
    ```

    **Example:**
    ```
    SS188-1_450mw_10s.spc
    SS188-2_450mw_10s.spc
    SS188-3_450mw_10s.spc
    SS190-1_450mw_10s.spc
    SS190-2_450mw_10s.spc
    SS190-3_450mw_10s.spc
    ```

    - Anything can come after the underscore (`_`) — for example, acquisition parameters.  
    - NexChem_AI automatically detects the number after the dash for replicate grouping and visualization.  
    - If you are using a different experimental design (e.g., time series), you can still use this format to separate samples in CV and visualization.

    ---

    ### Y-Block Excel Format
    The Y-block must contain at least these columns:

    | ID    | Class | DHA | EPA | PUFA |
    |:------|:------|----:|----:|----:|
    | SS188-1 | 2022  | 10.5 | 5.1 | 15.6 |
    | SS188-2 | 2022  | 10.5 | 5.1 | 15.6 |
    | SS188-3 | 2022  | 10.5 | 5.1 | 15.6 |
    | SS190-1 | 2023  | 1.5  | 1.0 | 2.5  |
    | SS190-2 | 2023  | 1.5  | 1.0 | 2.5  |
    | SS190-3 | 2023  | 1.5  | 1.0 | 2.5  |

    - The **ID** column must match the numeric portion of the corresponding spectra names.  
    - **Class** (optional): used for visualization or grouped CV.  
    - Columns such as **DHA**, **EPA**, or **PUFA** serve as target analytes.  
    - NexChem_AI automatically loops through each column after the **Class** column and builds one model per target.  
      *Only one target column is required.*

    
    ---

    **Developed by Ben Panitz – FAU Bioanalytical Core & Aquaculture Research**
    """)


# === Step 1: Data Loading ===
if tab == "Data Loading":
    st.header("Step 1: Load Raw Calibration Data")
    
    st.subheader("📂 Output Directory")

    default_outdir = st.session_state.get(
    "outdir",
    str(Path.home() / "Desktop")
    )
    
    outdir = st.text_input(
        "Output Directory",
        value=default_outdir
    )
    
    if st.button("Set Output Directory"):
        os.makedirs(outdir, exist_ok=True)
    
        spectra_root = os.path.join(outdir, "spectra")
        raw_dir = os.path.join(spectra_root, "raw")
        preproc_dir = os.path.join(spectra_root, "preprocessed")
        models_dir = os.path.join(outdir, "models")
        pred_dir = os.path.join(outdir, "predictions")
    
        os.makedirs(raw_dir, exist_ok=True)
        os.makedirs(preproc_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)
    
        st.session_state["outdir"] = outdir
        st.session_state["raw_spectra_dir"] = raw_dir
        st.session_state["preproc_spectra_dir"] = preproc_dir
        st.session_state["models_dir"] = models_dir
        st.session_state["pred_dir"] = pred_dir
    
        st.success(f"Results will be saved to: {outdir}")
        
    with st.expander("📁 Upload Calibration Raman Spectra (.zip of .spc files)"):
        zip_file = st.file_uploader("Upload a .zip file", type="zip")

    with st.expander("📄 Upload Calibration Y-block (Excel with 'ID' column)"):
        y_file = st.file_uploader("Upload GCMS target file", type="xlsx")

    # --- Button to actually load data into session_state ---
    if st.button("Load calibration data"):
        if not zip_file or not y_file:
            st.warning("Please upload BOTH the Raman .zip and the GC-MS Y-block file.")
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                zip_path = os.path.join(tmpdir, "calib_data.zip")
                with open(zip_path, "wb") as f:
                    f.write(zip_file.read())

                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(os.path.join(tmpdir, "unzipped"))

                raman_path = os.path.join(tmpdir, "unzipped")
                spectra_dir = st.session_state["raw_spectra_dir"]
                os.makedirs(spectra_dir, exist_ok=True)

                # ⬇️ Load raw spectra only (no preprocessing yet!)
                sample_spectra, shifts, sample_groups = load_raman(raman_path, spectra_dir)

                # Load and validate Y-block
                y_df = pd.read_excel(y_file)
                # index by ID for later use in modeling/preprocessing
                y_df.set_index("ID", inplace=True)

                # ✅ Store raw data for next steps & for fast re-draws
                st.session_state["raw_spectra"] = sample_spectra
                st.session_state["y_block"] = y_df
                # Originals — used by exclusion UI to restore from
                st.session_state["raw_spectra_original"] = sample_spectra
                st.session_state["y_block_original"] = y_df.copy()
                st.session_state.pop("excluded_ids", None)

                st.success("✅ Raw calibration data loaded.")
                st.write(f"**Raman spectra loaded**: {len(sample_spectra)} samples")
                st.write(f"**GCMS targets loaded**: {y_df.shape[0]} rows")

    # --- If data already loaded, show visualizations (no re-loading) ---
    if "raw_spectra" in st.session_state and "y_block" in st.session_state:
        sample_spectra = st.session_state["raw_spectra"]
        y_df = st.session_state["y_block"]

        st.write(f"**Raman spectra loaded**: {len(sample_spectra)} samples")
        st.write(f"**GCMS targets loaded**: {y_df.shape[0]} rows")

        # Optional: Display Y-block data
        with st.expander("🔬 Preview GC-MS Y-block"):
            st.dataframe(y_df, use_container_width=True)

        # Analyte correlation heatmap
        _analyte_cols = [c for c in y_df.columns if c != "Class"]
        if len(_analyte_cols) >= 2:
            with st.expander("📊 Analyte Correlation Map", expanded=False):
                _corr_fig = plot_analyte_correlation_map(y_df[_analyte_cols])
                _corr_buf = io.BytesIO()
                _corr_fig.savefig(_corr_buf, format="png", dpi=150, bbox_inches="tight")
                _corr_buf.seek(0)
                st.image(_corr_buf, width=550)
                _corr_pdf_buf = io.BytesIO()
                _corr_fig.savefig(_corr_pdf_buf, format="pdf", bbox_inches="tight")
                _corr_pdf_buf.seek(0)
                st.download_button(
                    label="⬇️ Download PDF",
                    data=_corr_pdf_buf,
                    file_name="analyte_correlation_map.pdf",
                    mime="application/pdf",
                )
                plt.close(_corr_fig)

        # === Spectra Overlay: Color by replicate or analyte ===
        _dl_figs: list = []
        st.subheader("📊 Spectra Overlay")

        _raw_color_options = ["Replicate"]
        if "Class" in y_df.columns:
            _raw_color_options.append("Class")
        _raw_color_options.append("Analyte (Y-block)")

        color_mode = st.radio(
            "Color spectra by:",
            _raw_color_options,
            index=0,
            horizontal=True,
            key="raw_color_mode"
        )

        # --- Option 1: color by replicate ---
        if color_mode == "Replicate":
            pfig = go.Figure()
            for sid, spectra in sample_spectra.items():
                for spectrum in spectra:
                    pfig.add_trace(go.Scatter(
                        x=spectrum.spectral_axis,
                        y=spectrum.spectral_data,
                        mode="lines",
                        name=sid,
                        line=dict(width=1),
                        opacity=0.7,
                        hovertemplate="<b>%{fullData.name}</b><br>Shift: %{x:.1f} cm⁻¹<br>Intensity: %{y:.2f}<extra></extra>",
                    ))
            pfig.update_layout(
                title="Overlay of Raw Spectra",
                xaxis_title="Raman Shift (cm⁻¹)",
                yaxis_title="Intensity",
                showlegend=False,
                height=500,
            )
            st.plotly_chart(pfig, use_container_width=True)

        # --- Option 2: color by class ---
        elif color_mode == "Class":
            class_by_id, color_by_class = ensure_class_colors_from_y(y_df, "Class")
            pfig = go.Figure()
            seen_classes = set()
            for sid, spectra in sample_spectra.items():
                cls = class_by_id.get(sid, "Unknown")
                rgba = color_by_class.get(cls, (0.5, 0.5, 0.5, 1.0))
                color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"
                for spectrum in spectra:
                    pfig.add_trace(go.Scatter(
                        x=spectrum.spectral_axis,
                        y=spectrum.spectral_data,
                        mode="lines",
                        name=cls,
                        legendgroup=cls,
                        showlegend=cls not in seen_classes,
                        line=dict(color=color_str, width=1),
                        opacity=0.7,
                        hovertemplate=f"<b>ID: {sid}</b><br>Class: {cls}<br>Shift: %{{x:.1f}} cm⁻¹<br>Intensity: %{{y:.2f}}<extra></extra>",
                    ))
                seen_classes.add(cls)
            pfig.update_layout(
                title="Overlay of Raw Spectra (Colored by Class)",
                xaxis_title="Raman Shift (cm⁻¹)",
                yaxis_title="Intensity",
                height=500,
            )
            st.plotly_chart(pfig, use_container_width=True)

        # --- Option 3: color by analyte from Y-block ---
        elif color_mode == "Analyte (Y-block)":
            st.markdown("### 🎨 Spectra Colored by Analyte")

            # analyte columns: everything except "Class" (if present)
            analytes_available = [col for col in y_df.columns if col not in ["Class"]]
            if not analytes_available:
                st.warning("No analyte columns available in Y-block to color by.")
            else:
                analyte_to_plot = st.selectbox(
                    "Select analyte to color by:",
                    analytes_available,
                    index=0,
                    key="raw_analyte_select"
                )

                try:
                    fig, ax = plot_spectra_colored_by_analyte(
                        sample_spectra=sample_spectra,
                        y_df=y_df.reset_index(),  # make sure we have an 'ID' column
                        analyte_col=analyte_to_plot,
                        id_col="ID",
                        use_first_replicate=True,
                        cmap_name="viridis",
                        title=f"Raw Spectra Colored by {analyte_to_plot}"
                    )

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf, caption=f"Colored Overlay by {analyte_to_plot}", width=800)
                    _dl_figs.append(fig)
                    plt.close(fig)

                except Exception as e:
                    st.warning(f"Could not generate analyte-colored overlay: {e}")

        # === Individual Raw Spectra Viewer ===
        st.subheader("🔍 Individual Spectra")

        def sort_key(sample_id):
            match = re.match(r"(\d+)-(\d+)", sample_id)
            if match:
                return (0, int(match.group(1)), int(match.group(2)))
            return (1, sample_id, 0)

        available_ids = sorted(sample_spectra.keys(), key=sort_key)
        selected_ids = st.multiselect("Select sample(s) to overlay", available_ids)

        if selected_ids:
            fig, ax = plt.subplots(figsize=(8, 6))
            for sample_id in selected_ids:
                for spectrum in sample_spectra[sample_id]:
                    ax.plot(spectrum.spectral_axis, spectrum.spectral_data, label=sample_id)

            ax.set_title("Raw Spectra")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity")
            ax.legend(loc="best", fontsize="small")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, width=800)
            _dl_figs.append(fig)
            plt.close(fig)

        # === Download Data Loading Figures ===
        if _dl_figs:
            st.download_button(
                "⬇️ Download Figures as PDF",
                data=figures_to_pdf_bytes(_dl_figs),
                file_name="data_loading_figures.pdf",
                mime="application/pdf",
            )

        # === Sample Exclusion ===
        st.markdown("---")
        st.subheader("🗑️ Exclude Samples")

        _spectra_original = st.session_state.get("raw_spectra_original", sample_spectra)
        _y_original       = st.session_state.get("y_block_original", y_df)

        def _sort_key(x):
            return int(x) if str(x).isdigit() else str(x)

        _all_groups = sorted(_y_original.index.astype(str).tolist(), key=_sort_key)

        _excluded_default = st.session_state.get("excluded_ids", [])
        _excluded = st.multiselect(
            "Select sample groups to exclude:",
            options=_all_groups,
            default=_excluded_default,
            help="Removes the selected groups from both the spectra and Y-block before preprocessing. "
                 "Use the individual spectra viewer above to identify outliers."
        )

        _col1, _col2 = st.columns([1, 1])
        with _col1:
            if st.button("Apply exclusions"):
                _excl_set = set(str(e) for e in _excluded)
                _filtered_spectra = {
                    k: v for k, v in _spectra_original.items()
                    if k.split("-")[0] not in _excl_set
                }
                _filtered_y = _y_original[~_y_original.index.astype(str).isin(_excl_set)]

                st.session_state["excluded_ids"] = _excluded
                st.session_state["raw_spectra"]  = _filtered_spectra
                st.session_state["y_block"]      = _filtered_y

                # Reset holdout — it was computed on a different set of samples
                if st.session_state.get("holdout_active", False):
                    st.session_state["holdout_active"] = False
                    for _k in ["raw_spectra_full", "y_block_full", "raw_spectra_holdout",
                               "y_block_holdout", "holdout_group", "holdout_group_col"]:
                        st.session_state.pop(_k, None)
                    st.warning("Hold-out split was reset — re-apply it below.")

                if st.session_state.get("preprocessing_done", False):
                    st.warning("Preprocessing was run on the previous dataset — re-run preprocessing.")

                st.success(
                    f"✅ {len(_excluded)} group(s) excluded. "
                    f"{len(_filtered_spectra)} spectra remaining."
                )
                st.rerun()

        with _col2:
            if _excluded_default and st.button("Clear exclusions"):
                st.session_state["excluded_ids"] = []
                st.session_state["raw_spectra"]  = _spectra_original
                st.session_state["y_block"]      = _y_original.copy()
                if st.session_state.get("holdout_active", False):
                    st.session_state["holdout_active"] = False
                    for _k in ["raw_spectra_full", "y_block_full", "raw_spectra_holdout",
                               "y_block_holdout", "holdout_group", "holdout_group_col"]:
                        st.session_state.pop(_k, None)
                    st.warning("Hold-out split was reset.")
                st.rerun()

        if _excluded_default:
            st.info(f"Currently excluded: **{', '.join(str(e) for e in _excluded_default)}**")

        # === Hold-out Group Split ===
        st.markdown("---")
        st.subheader("✂️ Hold-out Group Split")

        # Always use the full Y-block for the UI (before any split is applied)
        _y_for_split = st.session_state.get("y_block_full", y_df)

        holdout_enabled = st.checkbox(
            "Enable hold-out group split",
            value=st.session_state.get("holdout_active", False),
            help=(
                "Split the loaded dataset into a training set and a hold-out "
                "prediction set based on any categorical column in the Y-block "
                "(e.g. season, batch, site). The hold-out group can then be used "
                "directly in the Prediction tab without uploading separate files."
            ),
        )

        # User just disabled — restore originals
        if not holdout_enabled and st.session_state.get("holdout_active", False):
            st.session_state["raw_spectra"] = st.session_state.pop("raw_spectra_full")
            st.session_state["y_block"] = st.session_state.pop("y_block_full")
            st.session_state["holdout_active"] = False
            for _k in ["raw_spectra_holdout", "y_block_holdout", "holdout_group", "holdout_group_col"]:
                st.session_state.pop(_k, None)
            st.rerun()

        if holdout_enabled:
            # Offer all columns that look categorical (object dtype or ≤30 unique values)
            cat_cols = [
                col for col in _y_for_split.columns
                if _y_for_split[col].dtype == object or _y_for_split[col].nunique() <= 30
            ]
            if not cat_cols:
                st.warning("No suitable group columns found in the Y-block (need a text or low-cardinality column).")
            else:
                group_col = st.selectbox("Group column:", cat_cols, key="holdout_group_col_select")
                unique_groups = sorted(_y_for_split[group_col].dropna().unique().tolist(), key=str)
                holdout_groups = st.multiselect(
                    "Hold-out group(s):",
                    unique_groups,
                    default=st.session_state.get("holdout_group", []) if isinstance(st.session_state.get("holdout_group"), list) else [],
                    key="holdout_group_select",
                    help="Select one or more groups to combine into the hold-out set (e.g. two low-sample seasons)."
                )

                if holdout_groups:
                    _mask_holdout = _y_for_split[group_col].isin(holdout_groups)
                    n_holdout = int(_mask_holdout.sum())
                    n_train   = int((~_mask_holdout).sum())
                    st.info(f"Training: **{n_train}** rows | Hold-out: **{n_holdout}** rows")

                if st.button("Apply split"):
                    if not holdout_groups:
                        st.warning("Select at least one hold-out group.")
                        st.stop()

                    _mask_holdout = _y_for_split[group_col].isin(holdout_groups)
                    holdout_ids = set(_y_for_split[_mask_holdout].index.astype(str))
                    train_ids   = set(_y_for_split[~_mask_holdout].index.astype(str))

                    full_spectra = st.session_state.get("raw_spectra_full", st.session_state["raw_spectra"])

                    raw_spectra_train   = {k: v for k, v in full_spectra.items() if str(k) in train_ids}
                    raw_spectra_holdout = {k: v for k, v in full_spectra.items() if str(k) in holdout_ids}

                    # Keep the group column — the aligner extracts "Class" automatically
                    # and returns it as class_labels, so it never reaches the model.
                    y_block_train   = _y_for_split[~_mask_holdout]
                    y_block_holdout = _y_for_split[_mask_holdout]

                    # Preserve originals (only on first apply)
                    if "raw_spectra_full" not in st.session_state:
                        st.session_state["raw_spectra_full"] = st.session_state["raw_spectra"]
                    if "y_block_full" not in st.session_state:
                        st.session_state["y_block_full"] = _y_for_split

                    # Overwrite active datasets with training split
                    st.session_state["raw_spectra"] = raw_spectra_train
                    st.session_state["y_block"]     = y_block_train

                    # Store hold-out for Prediction tab
                    st.session_state["raw_spectra_holdout"] = raw_spectra_holdout
                    st.session_state["y_block_holdout"]     = y_block_holdout

                    st.session_state["holdout_active"]    = True
                    st.session_state["holdout_group"]     = holdout_groups
                    st.session_state["holdout_group_col"] = group_col

                    _label = ", ".join(str(g) for g in holdout_groups)
                    st.success(
                        f"✅ Split applied — Training: {len(raw_spectra_train)} spectra | "
                        f"Hold-out ({_label}): {len(raw_spectra_holdout)} spectra"
                    )
                    st.rerun()

        if st.session_state.get("holdout_active", False):
            _hg = st.session_state["holdout_group"]
            _hg_label = ", ".join(str(g) for g in _hg) if isinstance(_hg, list) else str(_hg)
            st.info(
                f"🔀 Active split — Hold-out group(s): **{_hg_label}** "
                f"(column: **{st.session_state['holdout_group_col']}**)"
            )

        # === Append Dataset ===
        st.subheader("➕ Append Dataset")

        with st.expander("📁 Upload Append Raman Spectra (.zip of .spc files)"):
            append_zip_file = st.file_uploader("Upload a .zip file", type="zip", key="append_zip")

        with st.expander("📄 Upload Append Y-block (Excel with 'ID' column)"):
            append_y_file = st.file_uploader("Upload append Y-block", type="xlsx", key="append_y")

        if st.button("Load Append Data"):
            if not append_zip_file or not append_y_file:
                st.warning("Please upload BOTH the append .zip and the append Y-block file.")
            else:
                with tempfile.TemporaryDirectory() as _atmpdir:
                    _azip_path = os.path.join(_atmpdir, "append_data.zip")
                    with open(_azip_path, "wb") as _f:
                        _f.write(append_zip_file.read())
                    with zipfile.ZipFile(_azip_path, "r") as _zref:
                        _zref.extractall(os.path.join(_atmpdir, "unzipped"))

                    _aspectra, _ashifts, _ = load_raman(
                        os.path.join(_atmpdir, "unzipped"),
                        st.session_state["raw_spectra_dir"],
                    )

                # Axis compatibility check
                _cur_spectra = st.session_state["raw_spectra"]
                _first_val = next(iter(_cur_spectra.values()))
                _first_spectrum = _first_val[0] if isinstance(_first_val, list) else _first_val
                _cur_len = len(_first_spectrum.spectral_data)
                if len(_ashifts) != _cur_len:
                    st.error(
                        f"Wavenumber axis mismatch: current dataset has {_cur_len} points, "
                        f"append dataset has {len(_ashifts)}. Crop/resample to the same axis before appending."
                    )
                else:
                    _ay_df = pd.read_excel(append_y_file)
                    _ay_df.set_index("ID", inplace=True)
                    st.session_state["_append_spectra"] = _aspectra
                    st.session_state["_append_y"] = _ay_df
                    st.success(f"Append data loaded: {len(_aspectra)} spectra, {_ay_df.shape[0]} Y rows.")

        # Phase 2 & 3 — column picker + confirm merge
        if "_append_spectra" in st.session_state and "_append_y" in st.session_state:
            _cur_y   = st.session_state["y_block"]
            _app_y   = st.session_state["_append_y"]
            _cur_cols = set(_cur_y.columns.tolist())
            _app_cols = set(_app_y.columns.tolist())
            _all_cols = sorted(_cur_cols | _app_cols)
            _shared   = sorted(_cur_cols & _app_cols)
            _only_cur = sorted(_cur_cols - _app_cols)
            _only_app = sorted(_app_cols - _cur_cols)

            st.markdown("**Y-block column comparison**")
            _col_info = pd.DataFrame({
                "Column": _all_cols,
                "In current dataset": ["✅" if c in _cur_cols else "—" for c in _all_cols],
                "In append dataset":  ["✅" if c in _app_cols else "—" for c in _all_cols],
            })
            st.dataframe(_col_info, use_container_width=True, hide_index=True)

            _selected_cols = st.multiselect(
                "Columns to keep in merged Y-block:",
                options=_all_cols,
                default=_shared,
            )
            if _only_cur and any(c in _selected_cols for c in _only_cur):
                st.info("Some selected columns are absent from the append dataset — those rows will have NaN.")
            if _only_app and any(c in _selected_cols for c in _only_app):
                st.info("Some selected columns are absent from the current dataset — those rows will have NaN.")

            _app_spectra = st.session_state["_append_spectra"]
            _overlap_ids = set(st.session_state["raw_spectra"].keys()) & set(_app_spectra.keys())
            if _overlap_ids:
                st.warning(f"Duplicate sample IDs found — append will overwrite {len(_overlap_ids)} existing spectra: {sorted(_overlap_ids)}")

            if st.button("Confirm Merge"):
                if not _selected_cols:
                    st.error("Select at least one Y-block column to keep.")
                else:
                    _merged_spectra = {**st.session_state["raw_spectra"], **_app_spectra}
                    _merged_y = pd.concat([_cur_y, _app_y], axis=0)[_selected_cols]

                    st.session_state["raw_spectra"]          = _merged_spectra
                    st.session_state["y_block"]              = _merged_y
                    st.session_state["raw_spectra_original"] = _merged_spectra.copy()
                    st.session_state["y_block_original"]     = _merged_y.copy()

                    # Clear stale state
                    st.session_state.pop("excluded_ids", None)
                    st.session_state.pop("_append_spectra", None)
                    st.session_state.pop("_append_y", None)
                    if st.session_state.get("holdout_active", False):
                        st.session_state["holdout_active"] = False
                        for _k in ["raw_spectra_full", "y_block_full", "raw_spectra_holdout",
                                   "y_block_holdout", "holdout_group", "holdout_group_col"]:
                            st.session_state.pop(_k, None)
                        st.warning("Hold-out split was reset after merge — re-apply it if needed.")

                    st.success(
                        f"✅ Datasets merged: {len(_merged_spectra)} total spectra, "
                        f"{_merged_y.shape[0]} Y rows, {len(_selected_cols)} columns kept."
                    )
                    st.rerun()

# === TAB 2: Preprocessing ===
if tab == "Preprocessing":
    if "raw_spectra" not in st.session_state or "y_block" not in st.session_state:
        st.warning("Please load calibration data in the 'Data Loading' tab first.")
    else:
        st.header("Step 2: Preprocess Spectra")

        from ramanspy import Spectrum

        selected_method = st.selectbox("Choose preprocessing method:", list(PREPROCESS_OPTIONS.keys()))

        # === Common Parameters ===
        crop_min = st.number_input("Crop region min (cm⁻¹)", value=DEFAULT_CROP_REGION[0])
        crop_max = st.number_input("Crop region max (cm⁻¹)", value=DEFAULT_CROP_REGION[1])
        crop_region = (crop_min, crop_max)

        # === Method-Specific parameters ===
        deriv_order = None
        avg_replicates = False
        if selected_method == "1. Savgol-SNV-MeanCenter":
            deriv_order = st.selectbox("Derivative order", options=[0, 1, 2], index=1)
            avg_replicates = st.checkbox(
                "Average replicates",
                value=False,
                help="Group replicates by sample prefix (before '-') and average after preprocessing."
            )

        # AsLS/SavGol parameters (optional)
        if selected_method == "2. Baseline-Smooth-SNV":
            asls_lambda = st.number_input("AsLS lambda", value=1e5, format="%.1e")
            asls_p = st.number_input("AsLS asymmetry p", value=0.001, format="%.3f")
            sg_window = st.number_input("Savitzky–Golay window length", value=13)
            sg_polyorder = st.number_input("Savitzky–Golay polynomial order", value=2)
            avg_replicates_2 = st.checkbox(
                "Average replicates",
                value=False,
                key="avg_rep_2",
                help="Group replicates by sample prefix (before '-') and average after preprocessing."
            )

        # Reference-EMSC parameters
        if selected_method == "3. Average Replicates: EMSC":
            emsc_p_order = st.number_input(
                "EMSC polynomial extension order", value=6, min_value=0, max_value=10,
                help="Degree of polynomial added to the EMSC design matrix (Liland 2016).")

            _ref_mode_label = st.selectbox(
                "EMSC reference mode",
                options=["Whole dataset", "By class", "By sample"],
                help=(
                    "**Whole dataset** — one reference from the global mean of all replicates. "
                    "**By class** — one reference per class (e.g. season); requires a class column in the Y-block. "
                    "**By sample** — one reference per biological sample (mean of its own replicates); always computed fresh."
                ),
            )
            _ref_mode_map = {"Whole dataset": "dataset", "By class": "class", "By sample": "sample"}
            emsc_ref_mode = _ref_mode_map[_ref_mode_label]

            emsc_class_col = None
            if emsc_ref_mode == "class":
                _y_for_emsc = st.session_state.get("y_block")
                if _y_for_emsc is not None:
                    _cat_cols = list(_y_for_emsc.select_dtypes(exclude=[np.number]).columns)
                    if not _cat_cols:
                        _cat_cols = list(_y_for_emsc.columns)
                    emsc_class_col = st.selectbox(
                        "Class column (used to group reference spectra):",
                        _cat_cols,
                        help="Each unique value in this column gets its own EMSC reference spectrum."
                    )
                else:
                    st.warning("Y-block not loaded — cannot use 'By class' mode.")
                    emsc_ref_mode = "dataset"

            # Reference baseline method
            _bl_method_label = st.selectbox(
                "Reference baseline correction method",
                options=["Lieber polynomial", "ALS (Asymmetric Least Squares)"],
                help=(
                    "**Lieber** — iterative polynomial constrained ≤ spectrum; best for fluorescence-heavy backgrounds. "
                    "**ALS** — asymmetric least-squares (Eilers & Boelens 2005); better for smooth scatter baselines in biological Raman."
                ),
            )
            emsc_bl_method = "lieber" if _bl_method_label == "Lieber polynomial" else "als"

            poly_ref_order = 4
            emsc_als_lam = 6.31e5
            emsc_als_p = 0.01
            if emsc_bl_method == "lieber":
                poly_ref_order = st.number_input(
                    "Polynomial order for reference baseline", value=4, min_value=1, max_value=10,
                    help="Order of iterative polynomial fit applied to the group mean to produce the EMSC reference spectrum.")
            else:
                emsc_als_lam = st.number_input(
                    "ALS smoothness λ", value=6.31e5, format="%.2e",
                    help="Smoothness parameter (λ). Paper 1 equivalent: 10^5.8 ≈ 6.3×10⁵.")
                emsc_als_p = st.number_input(
                    "ALS asymmetry p", value=0.01, format="%.4f",
                    help="Asymmetric weight (0 < p < 1). Paper 1 used p = 0.01.")

            # SavGol smoothing before EMSC
            apply_sg_smooth = st.checkbox(
                "Apply Savitzky–Golay smoothing before EMSC",
                value=False,
                help="Smooths each replicate (deriv=0) before fitting the EMSC model, "
                     "stabilising the scatter coefficient estimate. Paper 1 used window=9, poly=2."
            )
            emsc_sg_window = 9
            emsc_sg_polyorder = 2
            if apply_sg_smooth:
                emsc_sg_window = st.number_input("SG window length", value=9, min_value=3)
                emsc_sg_polyorder = st.number_input("SG polynomial order", value=2, min_value=1)

            # Mean-centering toggle
            apply_mean_center = st.checkbox(
                "Apply global mean-centering after EMSC",
                value=True,
                help="Subtracts the dataset-wide mean spectrum. Required for PLS; optional for other models."
            )

        run_button = st.button("Run Preprocessing")

        if run_button:
            sample_spectra = st.session_state["raw_spectra"]
            spectra_dir = st.session_state["preproc_spectra_dir"]
            os.makedirs(spectra_dir, exist_ok=True)

            # === SELECTED PIPELINE ===
            if selected_method == "1. Savgol-SNV-MeanCenter":
                if avg_replicates:
                    sample_groups = defaultdict(list)
                    for sid in sample_spectra.keys():
                        gid = sid.split("-")[0]
                        sample_groups[gid].append(sid)
                    preprocessed_spectra, cropped_axis, group_plot_dict, preproc_state = group_preprocess_savgol_snv_mc(
                        sample_spectra, sample_groups, spectra_dir,
                        crop_region=crop_region,
                        derivative_order=deriv_order,
                        return_state=True
                    )
                    st.session_state["group_plots"] = group_plot_dict
                    st.session_state["y_block_grouped"] = avg_y_block(st.session_state["y_block"])
                    st.session_state["sample_groups"] = sample_groups
                else:
                    preprocessed_spectra, cropped_axis, preproc_state = preprocess_savgol_snv_mc(
                        sample_spectra, spectra_dir,
                        crop_region=crop_region,
                        derivative_order=deriv_order,
                        return_state=True
                    )
                st.session_state["preproc_state"] = preproc_state
                st.session_state["trained_avg_replicates"] = avg_replicates

            elif selected_method == "2. Baseline-Smooth-SNV":
                if avg_replicates_2:
                    sample_groups = defaultdict(list)
                    for sid in sample_spectra.keys():
                        gid = sid.split("-")[0]
                        sample_groups[gid].append(sid)
                    preprocessed_spectra, cropped_axis, group_plot_dict = group_preprocess_asls_savgol_snv(
                        sample_spectra, sample_groups, spectra_dir,
                        crop_region=crop_region,
                        asls_lambda=asls_lambda,
                        asls_p=asls_p,
                        sg_window=sg_window,
                        sg_polyorder=sg_polyorder
                    )
                    st.session_state["group_plots"] = group_plot_dict
                    st.session_state["y_block_grouped"] = avg_y_block(st.session_state["y_block"])
                    st.session_state["sample_groups"] = sample_groups
                else:
                    preprocessed_spectra, cropped_axis = preprocess_asls_savgol_snv(
                        sample_spectra, spectra_dir,
                        crop_region=crop_region,
                        asls_lambda=asls_lambda,
                        asls_p=asls_p,
                        sg_window=sg_window,
                        sg_polyorder=sg_polyorder
                    )

                # === Save AsLS/Savitzky parameters for use during prediction ===
                st.session_state["trained_asls_lambda"] = asls_lambda
                st.session_state["trained_asls_p"] = asls_p
                st.session_state["trained_sg_window"] = sg_window
                st.session_state["trained_sg_polyorder"] = sg_polyorder
                st.session_state["trained_avg_replicates_2"] = avg_replicates_2

            elif selected_method == "4. None":
                preprocessed_spectra, cropped_axis = preprocess_none(
                    sample_spectra, spectra_dir,
                    crop_region=crop_region
                )

            elif selected_method == "3. Average Replicates: EMSC":
                sample_groups = defaultdict(list)
                for sid in sample_spectra.keys():
                    gid = sid.split("-")[0]
                    sample_groups[gid].append(sid)

                # Build class_lookup: gid → class_label (only for "class" mode)
                _class_lookup = None
                if emsc_ref_mode == "class" and emsc_class_col is not None:
                    _y_for_lookup = st.session_state["y_block"]
                    _class_lookup = {}
                    for _idx in _y_for_lookup.index:
                        _gid = str(_idx).split("-")[0]
                        _class_lookup[_gid] = str(_y_for_lookup.loc[_idx, emsc_class_col])

                preprocessed_spectra, cropped_axis, group_plot_dict, preproc_state = group_preprocess_ref_emsc_mc(
                    sample_spectra, sample_groups, spectra_dir,
                    crop_region=crop_region,
                    poly_ref_order=int(poly_ref_order),
                    emsc_p_order=int(emsc_p_order),
                    reference_mode=emsc_ref_mode,
                    class_lookup=_class_lookup,
                    ref_baseline_method=emsc_bl_method,
                    als_lam=float(emsc_als_lam),
                    als_p=float(emsc_als_p),
                    apply_sg_smooth=apply_sg_smooth,
                    sg_window=int(emsc_sg_window),
                    sg_polyorder=int(emsc_sg_polyorder),
                    apply_mean_center=apply_mean_center,
                    return_state=True
                )
                st.session_state["preproc_state"] = preproc_state
                st.session_state["trained_poly_ref_order"] = int(poly_ref_order)
                st.session_state["trained_emsc_p_order"] = int(emsc_p_order)
                st.session_state["trained_emsc_ref_mode"] = emsc_ref_mode
                st.session_state["trained_emsc_class_col"] = emsc_class_col
                st.session_state["trained_emsc_bl_method"] = emsc_bl_method
                st.session_state["trained_emsc_als_lam"] = float(emsc_als_lam)
                st.session_state["trained_emsc_als_p"] = float(emsc_als_p)
                st.session_state["trained_emsc_sg_smooth"] = apply_sg_smooth
                st.session_state["trained_emsc_sg_window"] = int(emsc_sg_window)
                st.session_state["trained_emsc_sg_polyorder"] = int(emsc_sg_polyorder)
                st.session_state["trained_emsc_mean_center"] = apply_mean_center

                st.session_state["group_plots"] = group_plot_dict
                st.session_state["y_block_grouped"] = avg_y_block(st.session_state["y_block"])
                st.session_state["sample_groups"] = sample_groups

            # === Save results to session ===
            st.session_state["trained_preprocess_key"] = selected_method
            st.session_state["trained_crop_region"] = crop_region
            st.session_state["trained_deriv_order"] = deriv_order
            st.session_state["trained_axis"] = cropped_axis
            st.session_state["preprocessed_spectra"] = preprocessed_spectra
            st.session_state["cropped_axis"] = cropped_axis
            st.session_state["preprocessing_done"] = True
            st.session_state["trained_is_group"] = (
                (selected_method == "1. Savgol-SNV-MeanCenter" and avg_replicates) or
                (selected_method == "2. Baseline-Smooth-SNV" and avg_replicates_2) or
                selected_method == "3. Average Replicates: EMSC"
            )

            st.success(f"✅ Preprocessing complete using: {selected_method}")
            st.write(f"Processed {len(preprocessed_spectra)} entries")

        # === Display Results ===
        if st.session_state.get("preprocessing_done", False):
            preprocessed_spectra = st.session_state["preprocessed_spectra"]
            _is_grouped = st.session_state.get("trained_is_group", False)
            y_df = st.session_state.get("y_block_grouped" if _is_grouped else "y_block", None)

            # ---------- Overlay with color-mode toggle ----------
            _prep_figs: list = []
            st.subheader("📊 Preprocessed Spectra Overlay")

            color_options = ["Replicate"]
            if y_df is not None:
                if "Class" in y_df.columns:
                    color_options.append("Class")
                color_options.append("Analyte (Y-block)")

            color_mode = st.radio(
                "Color spectra by:",
                color_options,
                index=0,
                horizontal=True,
                key="preproc_color_mode"
            )

            if color_mode == "Replicate":
                pfig = go.Figure()
                for sid, spectra in preprocessed_spectra.items():
                    if isinstance(spectra, Spectrum):
                        spectra = [spectra]
                    for spectrum in spectra:
                        pfig.add_trace(go.Scatter(
                            x=spectrum.spectral_axis,
                            y=spectrum.spectral_data,
                            mode="lines",
                            name=sid,
                            line=dict(width=1),
                            opacity=0.7,
                            hovertemplate="<b>%{fullData.name}</b><br>Shift: %{x:.1f} cm⁻¹<br>Intensity: %{y:.2f}<extra></extra>",
                        ))
                pfig.update_layout(
                    title="Overlay of All Preprocessed Spectra",
                    xaxis_title="Raman Shift (cm⁻¹)",
                    yaxis_title="Intensity (a.u.)",
                    showlegend=False,
                    height=500,
                )
                st.plotly_chart(pfig, use_container_width=True)

            elif color_mode == "Class":
                class_by_id, color_by_class = ensure_class_colors_from_y(y_df, "Class")
                pfig = go.Figure()
                seen_classes = set()
                for sid, spectra in preprocessed_spectra.items():
                    if isinstance(spectra, Spectrum):
                        spectra = [spectra]
                    cls = class_by_id.get(sid, "Unknown")
                    rgba = color_by_class.get(cls, (0.5, 0.5, 0.5, 1.0))
                    color_str = f"rgba({int(rgba[0]*255)},{int(rgba[1]*255)},{int(rgba[2]*255)},{rgba[3]:.2f})"
                    for spectrum in spectra:
                        pfig.add_trace(go.Scatter(
                            x=spectrum.spectral_axis,
                            y=spectrum.spectral_data,
                            mode="lines",
                            name=cls,
                            legendgroup=cls,
                            showlegend=cls not in seen_classes,
                            line=dict(color=color_str, width=1),
                            opacity=0.7,
                            hovertemplate=f"<b>ID: {sid}</b><br>Class: {cls}<br>Shift: %{{x:.1f}} cm⁻¹<br>Intensity: %{{y:.2f}}<extra></extra>",
                        ))
                    seen_classes.add(cls)
                pfig.update_layout(
                    title="Overlay of All Preprocessed Spectra (Colored by Class)",
                    xaxis_title="Raman Shift (cm⁻¹)",
                    yaxis_title="Intensity (a.u.)",
                    height=500,
                )
                st.plotly_chart(pfig, use_container_width=True)

            elif color_mode == "Analyte (Y-block)":
                # Color by analyte from Y-block (e.g. EPA+DHA, PUFA)
                st.markdown("### 🎨 Preprocessed Spectra Colored by Analyte")

                analytes_available = [col for col in y_df.columns if col not in ["Class"]]
                if not analytes_available:
                    st.warning("No analyte columns available in Y-block to color by.")
                else:
                    analyte_to_plot = st.selectbox(
                        "Select analyte to color by:",
                        analytes_available,
                        index=0,
                        key="preproc_analyte_select"
                    )

                    try:
                        fig, ax = plot_spectra_colored_by_analyte(
                            sample_spectra=preprocessed_spectra,
                            y_df=y_df,
                            analyte_col=analyte_to_plot,
                            id_col="ID",  # will fall back to index if no 'ID' col
                            use_first_replicate=True,
                            cmap_name="viridis",
                            title=f"Preprocessed Spectra Colored by {analyte_to_plot}"
                        )

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        st.image(buf, caption=f"Colored Overlay by {analyte_to_plot}", width=800)
                        _prep_figs.append(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.warning(f"Could not generate analyte-colored overlay: {e}")

            # ---------- Individual preprocessed spectra view ----------
            st.subheader("🔍 Individual Spectra")

            def sort_key(sample_id):
                import re
                match = re.match(r"(\d+)-(\d+)", sample_id)
                if match:
                    return int(match.group(1)), int(match.group(2))
                return sample_id

            available_ids = sorted(preprocessed_spectra.keys(), key=sort_key)
            selected_ids = st.multiselect("Select sample(s) to overlay", available_ids)

            if selected_ids:
                group_plots = st.session_state.get("group_plots", {})

                # If all selected IDs are group-averaged (have stored plots)
                if all(sample_id in group_plots for sample_id in selected_ids):
                    for sample_id in selected_ids:
                        st.subheader(f"Overlay Plot for Group: {sample_id}")
                        fig = group_plots[sample_id]

                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        st.image(buf, width=800)
                        _prep_figs.append(fig)
                        plt.close(fig)

                else:
                    # Overlay individual spectra for selected IDs
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for sample_id in selected_ids:
                        spectra = preprocessed_spectra[sample_id]
                        if isinstance(spectra, Spectrum):
                            spectra = [spectra]
                        for spectrum in spectra:
                            ax.plot(
                                spectrum.spectral_axis,
                                spectrum.spectral_data,
                                label=sample_id
                            )

                    ax.set_title("Overlay of Selected Preprocessed Spectra")
                    ax.set_xlabel("Raman Shift (cm⁻¹)")
                    ax.set_ylabel("Intensity")
                    ax.legend(loc="best", fontsize="small")

                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf, width=800)
                    _prep_figs.append(fig)
                    plt.close(fig)

            # === Download Preprocessing Figures ===
            if _prep_figs:
                st.download_button(
                    "⬇️ Download Preprocessing Figures as PDF",
                    data=figures_to_pdf_bytes(_prep_figs),
                    file_name="preprocessing_figures.pdf",
                    mime="application/pdf",
                )







# === TAB 3: Modeling ===
if tab == "Modeling":
    if "preprocessed_spectra" not in st.session_state or "y_block" not in st.session_state:
        st.warning("Please run preprocessing first in the 'Preprocessing' tab.")
    else:
        from models.run_loops import run_regression_loop
        from preprocessors.aligner import align_xy, align_group_xy

        st.header("Step 3: Build Model")
        modeling_mode = st.radio("Modeling type", ["Regression", "Classification"],
                                 horizontal=True, key="modeling_mode_radio",
                                 label_visibility="collapsed")

        # ── Classification branch ─────────────────────────────────────────────
        if modeling_mode == "Classification":
            from models.classification_wrappers import PLSDA_model
            from plotting.plot_classifier import (
                plot_plsda_cv_curve,
                plot_plsda_confusion_matrix,
                plot_plsda_scores,
                plot_plsda_score_distribution,
                plot_plsda_loadings,
            )

            st.subheader("Choose Classification Model")
            st.markdown("**PLS-DA** (Partial Least Squares Discriminant Analysis)")

            enable_manual_cls = st.checkbox(
                "Manually select number of latent variables?", value=False,
                key="cls_manual_lv")
            manual_param_cls = None
            if enable_manual_cls:
                manual_param_cls = st.number_input(
                    "Manual n_components:", min_value=1, max_value=20, value=3,
                    key="cls_n_comp")
            max_lv_cls = 15

            st.subheader("Choose Cross-Validation")
            n_folds_cls = st.number_input(
                "Number of K-Folds:", min_value=2, max_value=20, value=8,
                key="cls_n_folds")

            _first_val_cls = list(st.session_state["preprocessed_spectra"].values())[0]
            _is_group_avg_cls = not isinstance(_first_val_cls, list)

            use_group_kfold_cls = False
            if not _is_group_avg_cls:
                use_group_kfold_cls = st.checkbox(
                    "Keep replicates together in the same fold?", value=False,
                    key="cls_group_kfold")

            st.subheader("Class Column")
            _yb = st.session_state["y_block"]
            _class_col_options = list(_yb.columns)
            class_col_cls = st.selectbox(
                "Select class column:", options=_class_col_options,
                key="cls_col_select")

            if st.button("Run PLS-DA", key="cls_run_btn"):
                raw_X_cls = st.session_state["preprocessed_spectra"]
                axis_cls  = st.session_state["cropped_axis"]

                if _is_group_avg_cls:
                    raw_Y_cls = st.session_state.get(
                        "y_block_grouped", st.session_state["y_block"])
                    (filtered_X_cls, filtered_Y_cls,
                     filtered_ids_cls, class_labels_cls,
                     unmatched_cls) = align_group_xy(raw_X_cls, raw_Y_cls)
                    group_labels_cls = None
                else:
                    raw_Y_cls = st.session_state["y_block"]
                    (filtered_X_cls, filtered_Y_cls,
                     filtered_ids_cls, filtered_groups_cls,
                     class_labels_cls, unmatched_cls) = align_xy(raw_X_cls, raw_Y_cls)
                    group_labels_cls = (
                        filtered_groups_cls if use_group_kfold_cls else None)

                if unmatched_cls:
                    st.warning(
                        f"⚠️ Samples excluded (no matching spectra): "
                        f"{', '.join(sorted(str(u) for u in unmatched_cls))}")

                # Resolve class labels: align_xy strips a column literally
                # named "Class" and returns it as class_labels_cls.
                if class_col_cls in filtered_Y_cls.columns:
                    y_labels_cls = filtered_Y_cls[class_col_cls].values.astype(str)
                elif class_labels_cls is not None:
                    y_labels_cls = np.array(class_labels_cls, dtype=str)
                else:
                    st.error(
                        f"Could not locate column '{class_col_cls}' after "
                        f"alignment. Check your Y block.")
                    st.stop()

                if "models_dir" not in st.session_state:
                    st.error("Please set the Output Directory in the Data Loading tab.")
                    st.stop()

                with st.spinner("Running PLS-DA…"):
                    cls_results = PLSDA_model(
                        x            = filtered_X_cls,
                        y_labels     = y_labels_cls,
                        directory    = st.session_state["models_dir"],
                        axis         = axis_cls,
                        max_lv       = max_lv_cls,
                        analyte      = class_col_cls,
                        groups       = group_labels_cls,
                        manual_param = manual_param_cls,
                        sample_ids   = np.array(filtered_ids_cls),
                        n_folds      = int(n_folds_cls),
                    )

                st.session_state["cls_results"]  = cls_results
                st.session_state["cls_built"]    = True
                st.session_state["cls_analyte"]  = class_col_cls
                st.success("PLS-DA complete!")

            # ── Results ──────────────────────────────────────────────────────
            if st.session_state.get("cls_built", False):
                cls_res     = st.session_state["cls_results"]
                cls_analyte = st.session_state.get("cls_analyte", "")
                cls_classes = cls_res["classes"]
                opt_lv      = cls_res["optimal_param"]

                st.subheader("Model Summary")
                _mc1, _mc2, _mc3 = st.columns(3)
                _mc1.metric("Optimal LVs",   opt_lv)
                _mc2.metric("Cal Accuracy",  f"{cls_res['cal_accuracy']:.3f}")
                _mc3.metric("CV Accuracy",   f"{cls_res['cv_accuracy']:.3f}")

                if st.checkbox("Show CV accuracy table", value=False,
                               key="cls_show_cv_table"):
                    st.dataframe(cls_res["cv_table_df"], use_container_width=True)

                if st.checkbox("Show CV fold assignments", value=False,
                               key="cls_show_folds"):
                    _fdf = cls_res.get("fold_df")
                    if _fdf is not None:
                        st.dataframe(_fdf, use_container_width=True)

                st.subheader("Plots")

                with st.expander("Plot options", expanded=False):
                    _co1, _co2 = st.columns(2)
                    with _co1:
                        show_ell_cls = st.checkbox(
                            "Show 95% confidence ellipses", value=True,
                            key="cls_show_ell")
                        ell_alpha_cls = st.slider(
                            "Ellipse opacity", 0.05, 0.5, 0.18, step=0.01,
                            key="cls_ell_alpha")
                        label_samples_cls = st.checkbox(
                            "Label samples on scores plot", value=False,
                            key="cls_label_samples")
                    with _co2:
                        _lv_max = max(opt_lv, 2)
                        lv_x_cls = st.number_input(
                            "Scores plot X-axis (LV#)",
                            min_value=1, max_value=_lv_max, value=1,
                            key="cls_lv_x")
                        lv_y_cls = st.number_input(
                            "Scores plot Y-axis (LV#)",
                            min_value=1, max_value=_lv_max,
                            value=min(2, opt_lv),
                            key="cls_lv_y")
                        _lv_options = list(range(1, opt_lv + 1))
                        loadings_lvs_cls = st.multiselect(
                            "Loadings plot — LVs to show",
                            options=_lv_options,
                            default=_lv_options[:min(2, opt_lv)],
                            key="cls_loadings_lvs")
                        top_n_bands_cls = st.number_input(
                            "Label top N bands per LV (0 = none)",
                            min_value=0, max_value=20, value=0,
                            key="cls_top_n_bands")

                # Row 1 — CV curve | Scores plot
                _pc1, _pc2 = st.columns(2)
                with _pc1:
                    st.pyplot(plot_plsda_cv_curve(
                        cls_res["cv_results"], opt_lv, analyte=cls_analyte))
                with _pc2:
                    if opt_lv >= 2:
                        _slabels = (cls_res["sample_ids"]
                                    if label_samples_cls else None)
                        st.pyplot(plot_plsda_scores(
                            cls_res["x_scores"], cls_res["y_true"], cls_classes,
                            lv_x=lv_x_cls, lv_y=lv_y_cls,
                            show_ellipses=show_ell_cls,
                            ellipse_alpha=ell_alpha_cls,
                            analyte=cls_analyte,
                            sample_labels=_slabels))
                    else:
                        st.info("Scores plot requires ≥ 2 latent variables.")

                # Row 2 — Confusion matrices
                _pc3, _pc4 = st.columns(2)
                with _pc3:
                    st.pyplot(plot_plsda_confusion_matrix(
                        cls_res["y_true"], cls_res["y_pred_cal"],
                        cls_classes, suffix="", analyte=cls_analyte))
                with _pc4:
                    st.pyplot(plot_plsda_confusion_matrix(
                        cls_res["y_true"], cls_res["y_pred_cv"],
                        cls_classes, suffix=" (CV)", analyte=cls_analyte))

                # Row 3 — Score distributions
                _pc5, _pc6 = st.columns(2)
                with _pc5:
                    st.pyplot(plot_plsda_score_distribution(
                        cls_res["y_score_cal"], cls_res["y_true"],
                        cls_classes, suffix="", analyte=cls_analyte))
                with _pc6:
                    st.pyplot(plot_plsda_score_distribution(
                        cls_res["y_score_cv"], cls_res["y_true"],
                        cls_classes, suffix=" (CV)", analyte=cls_analyte))

                # Row 4 — Loadings
                if loadings_lvs_cls:
                    _x_load = cls_res["model"].pls_.x_loadings_
                    st.pyplot(plot_plsda_loadings(
                        _x_load, cls_res["axis"],
                        lv_indices=loadings_lvs_cls,
                        analyte=cls_analyte,
                        top_n_bands=int(top_n_bands_cls)))

            st.stop()
        # ── End classification branch ─────────────────────────────────────────

        # === Model selection ===
        st.subheader("Choose Regression Model")
        model_name = st.selectbox("Model Type:", ["PLS", "MLP"])

        manual_param  = None
        param_range   = None
        n_folds       = None
        param_grid    = None
        selector = None

        if model_name == "PLS":
            enable_manual_param = st.checkbox("Manually select number of PLS components?", value=False)
            if enable_manual_param:
                manual_param = st.number_input("Manual n_components:", min_value=1, max_value=20, value=5)
            param_range = list(range(1, 11))

        elif model_name == "MLP":
            # --- Preset grid definitions ---
            _COARSE_GRID = {
                'pls__n_components':      [4, 6, 10],
                'mlp__hidden_layer_sizes': [(5,), (10,), (20,)],
                'mlp__alpha':             [0.001, 0.01, 0.05, 0.1],
                'mlp__learning_rate_init':[0.001, 0.005, 0.01],
                'mlp__activation':        ['relu'],
                'mlp__early_stopping':    [True],
                'mlp__solver':            ['adam'],
            }
            _FINE_GRID = {
                'pls__n_components':      [4, 5, 6, 7, 8, 9, 10, 12],
                'mlp__hidden_layer_sizes': [(5,), (10,), (15,), (20,), (10, 1), (20, 1)],
                'mlp__alpha':             [0.001, 0.005, 0.01, 0.03, 0.06, 0.1],
                'mlp__learning_rate_init':[0.001, 0.003, 0.006, 0.01],
                'mlp__activation':        ['relu'],
                'mlp__early_stopping':    [True],
                'mlp__solver':            ['adam'],
            }

            preset = st.selectbox("Parameter grid preset:", ["Coarse", "Fine", "Custom"])

            if preset == "Coarse":
                param_grid = _COARSE_GRID
                with st.expander("Coarse grid details"):
                    st.markdown(
                        f"- **PLS components:** {_COARSE_GRID['pls__n_components']}  \n"
                        f"- **Hidden layers:** {[str(s) for s in _COARSE_GRID['mlp__hidden_layer_sizes']]}  \n"
                        f"- **Alpha:** {_COARSE_GRID['mlp__alpha']}  \n"
                        f"- **Learning rate:** {_COARSE_GRID['mlp__learning_rate_init']}  \n"
                        f"- **Activation:** relu"
                    )

            elif preset == "Fine":
                param_grid = _FINE_GRID
                with st.expander("Fine grid details"):
                    st.markdown(
                        f"- **PLS components:** {_FINE_GRID['pls__n_components']}  \n"
                        f"- **Hidden layers:** {[str(s) for s in _FINE_GRID['mlp__hidden_layer_sizes']]}  \n"
                        f"- **Alpha:** {_FINE_GRID['mlp__alpha']}  \n"
                        f"- **Learning rate:** {_FINE_GRID['mlp__learning_rate_init']}  \n"
                        f"- **Activation:** relu"
                    )

            else:  # Custom
                def _parse_ints(text):
                    return [int(v.strip()) for v in text.split(",") if v.strip()]

                def _parse_floats(text):
                    return [float(v.strip()) for v in text.split(",") if v.strip()]

                pls_text   = st.text_input("PLS components (comma-separated ints, range 3–12):", "4, 6, 8, 10")
                nodes1_text = st.text_input("1st layer nodes (comma-separated ints, range 1–20):", "5, 10, 20")
                nodes2_text = st.text_input("2nd layer nodes (0 = none, comma-separated):", "0, 1")
                alpha_text  = st.text_input("Alpha values (comma-separated floats, range 0.01–0.1):", "0.01, 0.05, 0.1")
                lr_text     = st.text_input("Learning rates (comma-separated floats, range 0.001–0.01):", "0.001, 0.005, 0.01")

                try:
                    pls_components = _parse_ints(pls_text)
                    nodes_first    = _parse_ints(nodes1_text)
                    nodes_second   = _parse_ints(nodes2_text)
                    alpha_values   = _parse_floats(alpha_text)
                    learning_rates = _parse_floats(lr_text)
                except ValueError:
                    st.error("Could not parse one of the custom inputs — check for non-numeric values.")
                    pls_components = [6]
                    nodes_first = [10]
                    nodes_second = [0]
                    alpha_values = [0.05]
                    learning_rates = [0.005]

                hidden_layer_sizes = []
                for n1 in nodes_first:
                    for n2 in nodes_second:
                        if n2 == 0:
                            hidden_layer_sizes.append((n1,))
                        else:
                            hidden_layer_sizes.append((n1, n2))

                param_grid = {
                    'pls__n_components':       pls_components,
                    'mlp__hidden_layer_sizes': hidden_layer_sizes,
                    'mlp__alpha':              alpha_values,
                    'mlp__learning_rate_init': learning_rates,
                    'mlp__activation':         ['relu'],
                    'mlp__early_stopping':     [True],
                    'mlp__solver':             ['adam'],
                }

            # --- Combo count ---
            if param_grid:
                from math import prod
                n_combos = prod(len(v) for v in param_grid.values())
                st.info(f"Grid size: **{n_combos} combinations** to fit")

        # === Cross-validation options ===
        st.subheader("Choose Cross-Validation")
        use_group_kfold = st.checkbox("Use Grouped K-Fold CV?", value=False)
        
        group_strategy = None
        if use_group_kfold:
            group_strategy = st.radio("Group K-Fold by:", options=["Replicate", "Class"], index=0)
                
                
        n_folds = st.number_input("Number of K-Folds:", min_value=2, max_value=20, value=8)

        if model_name == "PLS":
            st.subheader("Variable Selection")
            var_select_method = st.selectbox(
                "Method:",
                options=["None", "VIP threshold", "iPLS (block)"],
                help="Select spectral variables before building the final model.",
            )
            if var_select_method == "VIP threshold":
                vip_threshold = st.slider(
                    "VIP threshold",
                    min_value=0.25, max_value=2.0, value=1.0, step=0.05,
                    help="Retain variables with VIP ≥ threshold. Conventional cutoff is 1.0.",
                )
                from models.selectors.vip import VIPSelector
                selector = VIPSelector(threshold=float(vip_threshold))
            elif var_select_method == "iPLS (block)":
                block_size = st.number_input(
                    "Block size (variables per block)",
                    min_value=5, max_value=500, value=100, step=5,
                    help="Number of contiguous spectral variables per block.",
                )
                from models.selectors.block import BlockSelector
                selector = BlockSelector(block_size=int(block_size))

        st.markdown("### Visualization Options")
        
        # Determine whether preprocessing is group-averaged
        first_val = list(st.session_state["preprocessed_spectra"].values())[0]
        is_group_avg = not isinstance(first_val, list)
        
        # Set available coloring options based on mode
        coloring_choices = ["None", "Class"]
        if not is_group_avg:
            coloring_choices.append("Replicate")
        
        coloring_option = st.selectbox(
            "Color predicted vs. actual plots by:",
            options=coloring_choices,
            index=0
        )
        
        # === Start modeling ===
        if st.button("Train Model"):
            raw_X = st.session_state["preprocessed_spectra"]
            axis = st.session_state["cropped_axis"]
            sample_groups = st.session_state.get("sample_groups")

            first_val = list(raw_X.values())[0]
            is_group_avg = not isinstance(first_val, list)

            # Use group-averaged Y when spectra are group-averaged; never mutate "y_block"
            if is_group_avg:
                raw_Y = st.session_state.get("y_block_grouped", st.session_state["y_block"])
            else:
                raw_Y = st.session_state["y_block"]

            if is_group_avg:
                filtered_X, filtered_Y, filtered_sample_ids, classes, unmatched_ids = align_group_xy(raw_X, raw_Y)
                st.session_state["classes"] = classes
            else:
                filtered_X, filtered_Y, filtered_sample_ids, filtered_groups, classes, unmatched_ids = align_xy(raw_X, raw_Y)

            group_labels = None
            if use_group_kfold:
                if group_strategy == "Replicate":
                    group_labels = filtered_groups
                elif group_strategy == "Class":
                    group_labels = classes

            if "models_dir" not in st.session_state:
                st.error("Please set the Output Directory in the Data Loading tab.")
                st.stop()
            
            results_dir = st.session_state["models_dir"]
            
            
            color_labels = None
            if coloring_option == "Class":
                color_labels = classes
            elif coloring_option == "Replicate":
                color_labels = filtered_groups
            st.session_state["color_labels"] = color_labels


            model_results = run_regression_loop(
                filtered_X,
                filtered_Y,
                results_dir,
                axis,
                groups=group_labels,
                model_name=model_name,
                param_range=param_range,
                param_grid=param_grid,
                manual_param=manual_param,
                n_folds=n_folds,
                sample_ids=filtered_sample_ids,
                class_labels=color_labels,
                selector=selector,
            )

            st.session_state["model_results"] = model_results
            st.session_state["model_built"] = True
            st.session_state["unmatched_ids"] = unmatched_ids
            st.success("✅ Model training complete!")

        # === Display Model Summary (after training or reload) ===
        if st.session_state.get("model_built", False):
            model_results = st.session_state["model_results"]
            unmatched_ids = st.session_state.get("unmatched_ids", [])
            classes = st.session_state.get("classes", None)
            filtered_groups = st.session_state.get("filtered_groups", None)

            st.subheader("Model Summary")

            if unmatched_ids:
                st.warning(f"❗ Unmatched Sample IDs: {', '.join(sorted(unmatched_ids))}")
                
            # === CV Sample Assignment Toggle ===
            if st.checkbox("Show CV sample fold assignments", value=False):
                fold_df = None
            
                # Pull the first available fold_df from any analyte
                for result in model_results.values():
                    fold_df = result.get("fold_df")
                    if fold_df is not None:
                        break
            
                if fold_df is not None:
                    st.dataframe(fold_df, use_container_width=True)
                else:
                    st.info("No fold assignment data was found.")    

            show_cv_tables = st.checkbox("Show cross-validation tables", value=False)

            for analyte in model_results.keys():
                result = model_results[analyte]
                summary = result.get("summary", None)
                if summary:
                    st.markdown(summary)


                if show_cv_tables:
                    cv_table = result.get("cv_table_df")
                    if cv_table is not None:
                        st.markdown(f"**{analyte} – CV Metrics**")
                        st.dataframe(cv_table.style.format(precision=4))

            # === CV Diagnostic Plots ===
            st.subheader("Cross Validation Diagnostics")

            # ── Plot style options ──────────────────────────────────────────
            with st.expander("⚙️ CV plot options", expanded=False):
                _cs = st.session_state.get("cv_plot_style", {})
                _cc1, _cc2, _cc3 = st.columns(3)
                with _cc1:
                    _cv_tick_fs  = st.number_input("Tick label size",  value=_cs.get("tick_fontsize",  10), min_value=6, step=1, key="cv_tick_fs")
                    _cv_label_fs = st.number_input("Axis label size",  value=_cs.get("label_fontsize", 12), min_value=6, step=1, key="cv_label_fs")
                with _cc2:
                    _cv_fw = st.number_input("Fig width",  value=_cs.get("fig_width",  8.0), min_value=2.0, step=0.5, format="%.1f", key="cv_fw")
                    _cv_fh = st.number_input("Fig height", value=_cs.get("fig_height", 6.0), min_value=2.0, step=0.5, format="%.1f", key="cv_fh")
                with _cc3:
                    _cv_xlabel    = st.text_input("X axis label",     value=_cs.get("xlabel", ""),      key="cv_xlabel",    help="Blank = use parameter name")
                    _cv_ylabel_r2 = st.text_input("Y label (R² plot)",   value=_cs.get("ylabel_r2",  "R²"),  key="cv_ylabel_r2")
                    _cv_ylabel_rmse = st.text_input("Y label (RMSE plot)", value=_cs.get("ylabel_rmse", "RMSE"), key="cv_ylabel_rmse")

                _ct1, _ct2 = st.columns(2)
                with _ct1:
                    _cv_title_r2   = st.text_input("R² plot title (blank = auto)",   value=_cs.get("title_r2",   ""), key="cv_title_r2")
                with _ct2:
                    _cv_title_rmse = st.text_input("RMSE plot title (blank = auto)", value=_cs.get("title_rmse", ""), key="cv_title_rmse")

                _cr2_1, _cr2_2, _crmse_1, _crmse_2 = st.columns(4)
                _cv_auto_r2   = st.checkbox("Auto Y range (R²)",   value=_cs.get("axis_auto_r2",   True), key="cv_auto_r2")
                _cv_auto_rmse = st.checkbox("Auto Y range (RMSE)", value=_cs.get("axis_auto_rmse", True), key="cv_auto_rmse")
                _cv_ymin_r2 = _cv_ymax_r2 = _cv_ymin_rmse = _cv_ymax_rmse = None
                if not _cv_auto_r2:
                    _ra1, _ra2 = st.columns(2)
                    with _ra1:
                        _cv_ymin_r2 = st.number_input("R² Y min", value=float(_cs.get("ymin_r2") or 0.0), format="%.3f", key="cv_ymin_r2")
                    with _ra2:
                        _cv_ymax_r2 = st.number_input("R² Y max", value=float(_cs.get("ymax_r2") or 1.0), format="%.3f", key="cv_ymax_r2")
                if not _cv_auto_rmse:
                    _rm1, _rm2 = st.columns(2)
                    with _rm1:
                        _cv_ymin_rmse = st.number_input("RMSE Y min", value=float(_cs.get("ymin_rmse") or 0.0), format="%.3f", key="cv_ymin_rmse")
                    with _rm2:
                        _cv_ymax_rmse = st.number_input("RMSE Y max", value=float(_cs.get("ymax_rmse") or 1.0), format="%.3f", key="cv_ymax_rmse")

                st.session_state["cv_plot_style"] = {
                    "tick_fontsize":  _cv_tick_fs,
                    "label_fontsize": _cv_label_fs,
                    "fig_width":      _cv_fw,
                    "fig_height":     _cv_fh,
                    "xlabel":         _cv_xlabel,
                    "ylabel_r2":      _cv_ylabel_r2,
                    "ylabel_rmse":    _cv_ylabel_rmse,
                    "title_r2":       _cv_title_r2,
                    "title_rmse":     _cv_title_rmse,
                    "axis_auto_r2":   _cv_auto_r2,
                    "ymin_r2":        _cv_ymin_r2,
                    "ymax_r2":        _cv_ymax_r2,
                    "axis_auto_rmse": _cv_auto_rmse,
                    "ymin_rmse":      _cv_ymin_rmse,
                    "ymax_rmse":      _cv_ymax_rmse,
                }

            _cv_style = st.session_state.get("cv_plot_style", {})

            # ── Static Pred vs Actual options ──────────────────────────────
            with st.expander("⚙️ Static Pred vs Actual options", expanded=False):
                _ps = st.session_state.get("pva_plot_style", {})
                _pa1, _pa2, _pa3 = st.columns(3)
                with _pa1:
                    _pva_tick_fs  = st.number_input("Tick label size",  value=_ps.get("tick_fontsize",  10), min_value=6, step=1, key="pva_tick_fs")
                    _pva_label_fs = st.number_input("Axis label size",  value=_ps.get("label_fontsize", 12), min_value=6, step=1, key="pva_label_fs")
                with _pa2:
                    _pva_pt_size = st.number_input("Point size", value=_ps.get("point_size", 50), min_value=1, step=1, key="pva_pt_size")
                with _pa3:
                    _pva_fw_col, _pva_fh_col = st.columns(2)
                    with _pva_fw_col:
                        _pva_fw = st.number_input("Fig width",  value=_ps.get("pva_fig_width",  8.0), min_value=2.0, step=0.5, format="%.1f", key="pva_fw")
                    with _pva_fh_col:
                        _pva_fh = st.number_input("Fig height", value=_ps.get("pva_fig_height", 6.0), min_value=2.0, step=0.5, format="%.1f", key="pva_fh")
                _pva_auto = st.checkbox("Auto axis range", value=_ps.get("pva_axis_auto", True), key="pva_axis_auto")
                _pva_lo = _pva_hi = None
                if not _pva_auto:
                    _plo1, _plo2 = st.columns(2)
                    with _plo1:
                        _pva_lo = st.number_input("Axis min", value=float(_ps.get("pva_lo") or 0.0), format="%.3f", key="pva_lo")
                    with _plo2:
                        _pva_hi = st.number_input("Axis max", value=float(_ps.get("pva_hi") or 1.0), format="%.3f", key="pva_hi")
                _px1, _px2 = st.columns(2)
                with _px1:
                    _pva_xlabel = st.text_input("X axis label", value=_ps.get("pva_xlabel", "Actual Values"),    key="pva_xlabel")
                with _px2:
                    _pva_ylabel = st.text_input("Y axis label", value=_ps.get("pva_ylabel", "Predicted Values"), key="pva_ylabel")

                st.session_state["pva_plot_style"] = {
                    "tick_fontsize":  _pva_tick_fs,
                    "label_fontsize": _pva_label_fs,
                    "point_size":     _pva_pt_size,
                    "pva_fig_width":  _pva_fw,
                    "pva_fig_height": _pva_fh,
                    "pva_axis_auto":  _pva_auto,
                    "pva_lo":         _pva_lo,
                    "pva_hi":         _pva_hi,
                    "pva_xlabel":     _pva_xlabel,
                    "pva_ylabel":     _pva_ylabel,
                }

            _pva_style = st.session_state.get("pva_plot_style", {})

            for analyte in model_results.keys():
                result = model_results[analyte]

                st.subheader(f"🔬 {analyte}")

                if model_name != "MLP":
                    cv_data = result.get("cv_results", {})
                    if cv_data:
                        _xlabel_resolved = _cv_style.get("xlabel") or result.get("param_name", "n_components")
                        _style_with_xlabel = {**_cv_style, "xlabel": _xlabel_resolved}
                        fig_r2, fig_rmse = _build_cv_figures(
                            param_range=result["param_range"],
                            r2_cv=cv_data.get("pooled_r2_CV", []),
                            r2_cal=cv_data.get("mean_r2_cal", []),
                            rmse_cv=cv_data.get("pooled_rmse_CV", []),
                            rmse_cal=cv_data.get("mean_rmse_cal", []),
                            param_name=result.get("param_name", "n_components"),
                            analyte=analyte,
                            model_name=model_name,
                            style=_style_with_xlabel,
                        )
                        _col_r2, _col_rmse = st.columns(2)
                        with _col_r2:
                            _buf = io.BytesIO()
                            fig_r2.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
                            _buf.seek(0)
                            st.image(_buf, caption=f"CV R² vs {result.get('param_name','n_components')} — {analyte}", use_container_width=True)
                            # Update saved file so download reflects current style
                            _r2_path = result.get("cv_r2_plot_path")
                            if _r2_path:
                                fig_r2.savefig(_r2_path, dpi=300, bbox_inches="tight")
                                fig_r2.savefig(str(Path(_r2_path).with_suffix(".pdf")), bbox_inches="tight")
                            plt.close(fig_r2)
                        with _col_rmse:
                            _buf = io.BytesIO()
                            fig_rmse.savefig(_buf, format="png", dpi=150, bbox_inches="tight")
                            _buf.seek(0)
                            st.image(_buf, caption=f"CV RMSE vs {result.get('param_name','n_components')} — {analyte}", use_container_width=True)
                            _rmse_path = result.get("cv_rmse_plot_path")
                            if _rmse_path:
                                fig_rmse.savefig(_rmse_path, dpi=300, bbox_inches="tight")
                                fig_rmse.savefig(str(Path(_rmse_path).with_suffix(".pdf")), bbox_inches="tight")
                            plt.close(fig_rmse)

                # Interactive CV predicted vs actual
                if result.get("y_true") is not None and result.get("y_pred_cv") is not None:
                    fig = plot_pred_vs_actual_interactive(
                        y_true=result["y_true"],
                        y_pred=result["y_pred_cv"],
                        title=f"CV Predicted vs Actual — {analyte}",
                        sample_ids=result.get("sample_ids"),
                        class_labels=st.session_state.get("color_labels"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    _show_static = st.checkbox(
                        "Show static pred vs actual",
                        value=False,
                        key=f"show_static_pva_{analyte}",
                    )
                    if _show_static:
                        # Static CV pred vs actual — matches downloaded version
                        _fig_pva = _build_pred_vs_actual_fig(
                            y_true=np.asarray(result["y_true"]).ravel(),
                            y_pred=np.asarray(result["y_pred_cv"]).ravel(),
                            title=f"CV Predicted vs Actual — {analyte}",
                            class_labels=st.session_state.get("color_labels"),
                            style=_pva_style,
                        )
                        _buf_pva = io.BytesIO()
                        _fig_pva.savefig(_buf_pva, format="png", dpi=150, bbox_inches="tight")
                        _buf_pva.seek(0)
                        st.image(_buf_pva, caption=f"CV Pred vs Actual (static) — {analyte}", width=500)
                        # Update saved file so download reflects current style
                        _pva_path = result.get("cv_pred_plot_path")
                        if _pva_path:
                            _fig_pva.savefig(_pva_path, dpi=300, bbox_inches="tight")
                            _fig_pva.savefig(str(Path(_pva_path).with_suffix(".pdf")), bbox_inches="tight")
                        plt.close(_fig_pva)

                        # --- TEMP: paper figure ---
                        plot_pred_vs_actual_paper(
                            y_true=np.asarray(result["y_true"]).ravel(),
                            y_pred=np.asarray(result["y_pred_cv"]).ravel(),
                            directory=st.session_state.get("outdir"),
                            filename="fig_paper.png",
                            title=f"CV Predicted vs Actual — {analyte}",
                            class_labels=st.session_state.get("color_labels"),
                        )
                        _paper_path = os.path.join(st.session_state.get("outdir"), "fig_paper.png")
                        st.image(_paper_path, caption=f"Paper plot — {analyte}", width=500)
                        # --- END TEMP ---

                st.divider()

            # === Diagnostic Plots ===
            st.subheader("📉 Loadings, Variables & Scores")

            _diag_choices = [
                "T² vs Q Residuals",
                "Final Predicted vs Actual",
                "Regression Coefficients",
                "VIP Scores",
                "LV1 vs LV2 Scores",
            ]
            _selected_diag = st.selectbox(
                "Select plot to display:",
                options=_diag_choices,
                key="diag_plot_select",
            )

            # ── VIP plot options (only visible when VIP is selected) ─────────
            if _selected_diag == "VIP Scores":
                with st.expander("⚙️ VIP plot options", expanded=False):
                    _vs = st.session_state.get("vip_plot_style", {})
                    _vc1, _vc2, _vc3 = st.columns(3)
                    with _vc1:
                        _vip_tick_fs  = st.number_input("Tick label size",  value=_vs.get("tick_fontsize",  10), min_value=6, step=1, key="vip_tick_fs")
                        _vip_label_fs = st.number_input("Axis title size",   value=_vs.get("label_fontsize", 11), min_value=6, step=1, key="vip_label_fs")
                    with _vc2:
                        _vip_top_n    = st.number_input("Top N to label (0 = none)", value=_vs.get("top_n", 0), min_value=0, step=1, key="vip_top_n")
                    with _vc3:
                        _vip_xmin     = st.number_input("X-axis min (0 = auto)", value=_vs.get("xmin", 0), min_value=0, step=10, key="vip_xmin")
                        _vip_xmax     = st.number_input("X-axis max (0 = auto)", value=_vs.get("xmax", 0), min_value=0, step=10, key="vip_xmax")
                        _vip_ymin     = st.number_input("Y-axis min (0 = auto)", value=_vs.get("ymin", 0.0), step=0.1, format="%.2f", key="vip_ymin")
                        _vip_ymax     = st.number_input("Y-axis max (0 = auto)", value=_vs.get("ymax", 0.0), step=0.1, format="%.2f", key="vip_ymax")
                    st.session_state["vip_plot_style"] = {
                        "tick_fontsize":  _vip_tick_fs,
                        "label_fontsize": _vip_label_fs,
                        "top_n":          _vip_top_n,
                        "xmin":           _vip_xmin,
                        "xmax":           _vip_xmax,
                        "ymin":           _vip_ymin,
                        "ymax":           _vip_ymax,
                    }

            _vip_style = st.session_state.get("vip_plot_style", {})
            _vip_xlim  = None
            if _vip_style.get("xmin") and _vip_style.get("xmax"):
                _vip_xlim = (_vip_style["xmin"], _vip_style["xmax"])
            _vip_ylim  = None
            if _vip_style.get("ymin") and _vip_style.get("ymax"):
                _vip_ylim = (_vip_style["ymin"], _vip_style["ymax"])

            # ── Coef plot options (only visible when Coef is selected) ───────
            if _selected_diag == "Regression Coefficients":
                with st.expander("⚙️ Coefficient plot options", expanded=False):
                    _cs2 = st.session_state.get("coef_plot_style", {})
                    _cc1, _cc2, _cc3 = st.columns(3)
                    with _cc1:
                        _coef_tick_fs  = st.number_input("Tick label size",  value=_cs2.get("tick_fontsize",  10), min_value=6, step=1, key="coef_tick_fs")
                        _coef_label_fs = st.number_input("Axis title size",   value=_cs2.get("label_fontsize", 11), min_value=6, step=1, key="coef_label_fs")
                    with _cc2:
                        _coef_top_n    = st.number_input("Top N to label (0 = none)", value=_cs2.get("top_n", 0), min_value=0, step=1, key="coef_top_n")
                    with _cc3:
                        _coef_xmin     = st.number_input("X-axis min (0 = auto)", value=_cs2.get("xmin", 0), min_value=0, step=10, key="coef_xmin")
                        _coef_xmax     = st.number_input("X-axis max (0 = auto)", value=_cs2.get("xmax", 0), min_value=0, step=10, key="coef_xmax")
                        _coef_ymin     = st.number_input("Y-axis min (0 = auto)", value=_cs2.get("ymin", 0.0), step=0.001, format="%.3f", key="coef_ymin")
                        _coef_ymax     = st.number_input("Y-axis max (0 = auto)", value=_cs2.get("ymax", 0.0), step=0.001, format="%.3f", key="coef_ymax")
                    st.session_state["coef_plot_style"] = {
                        "tick_fontsize":  _coef_tick_fs,
                        "label_fontsize": _coef_label_fs,
                        "top_n":          _coef_top_n,
                        "xmin":           _coef_xmin,
                        "xmax":           _coef_xmax,
                        "ymin":           _coef_ymin,
                        "ymax":           _coef_ymax,
                    }

            _coef_style = st.session_state.get("coef_plot_style", {})
            _coef_xlim  = None
            if _coef_style.get("xmin") and _coef_style.get("xmax"):
                _coef_xlim = (_coef_style["xmin"], _coef_style["xmax"])
            _coef_ylim  = None
            if _coef_style.get("ymin") and _coef_style.get("ymax"):
                _coef_ylim = (_coef_style["ymin"], _coef_style["ymax"])

            for analyte in model_results.keys():
                result = model_results[analyte]
                diag_plots = result.get("diagnostic_plots", [])
                entry = next((e for e in diag_plots if e["caption"] == _selected_diag), None)
                if entry is None or not os.path.exists(entry["path"]):
                    continue

                st.markdown(f"**🔬 {analyte} ({result.get('model_type', '')})**")

                if entry["caption"] == "VIP Scores" and result.get("model_type") == "PLS":
                    from plotting.plot_regression import plot_vip_scores as _plot_vip
                    from models.vip import calculate_vip as _calc_vip
                    _vip_s = result.get("vip_scores")
                    _vip_a = result.get("vip_axis")
                    if _vip_s is None:
                        _vip_s = _calc_vip(result["model"])
                    if _vip_a is None:
                        _vip_a = st.session_state.get("cropped_axis")
                    _vfig = _plot_vip(
                        None, None,
                        _vip_a,
                        directory=None,
                        model_name="PLS",
                        analyte=analyte,
                        vip=_vip_s,
                        style={**_vip_style, "xlim": _vip_xlim, "ylim": _vip_ylim},
                    )
                    _vbuf = io.BytesIO()
                    _vfig.savefig(_vbuf, format="png", dpi=150, bbox_inches="tight")
                    _vbuf.seek(0)
                    st.image(_vbuf, caption=entry["caption"], width=500)
                    _vip_path = result.get("vip_plot_path")
                    if _vip_path:
                        _vfig.savefig(_vip_path, dpi=300, bbox_inches="tight")
                        _vfig.savefig(str(Path(_vip_path).with_suffix(".pdf")), bbox_inches="tight")
                    plt.close(_vfig)
                elif entry["caption"] == "Regression Coefficients" and result.get("model_type") == "PLS":
                    from plotting.plot_regression import plot_coefficients as _plot_coef
                    _coef_arr = result.get("coef_array")
                    _coef_ax  = result.get("coef_axis")
                    if _coef_arr is None:
                        _coef_arr = np.asarray(result["model"].coef_).flatten()
                    if _coef_ax is None:
                        _coef_ax = st.session_state.get("cropped_axis")
                    _cfig = _plot_coef(
                        _coef_ax,
                        _coef_arr,
                        directory=None,
                        model_name="PLS",
                        analyte=analyte,
                        style={**_coef_style, "xlim": _coef_xlim, "ylim": _coef_ylim},
                    )
                    _cbuf = io.BytesIO()
                    _cfig.savefig(_cbuf, format="png", dpi=150, bbox_inches="tight")
                    _cbuf.seek(0)
                    st.image(_cbuf, caption=entry["caption"], width=500)
                    _coef_path = result.get("coef_plot_path")
                    if _coef_path:
                        _cfig.savefig(_coef_path, dpi=300, bbox_inches="tight")
                        _cfig.savefig(str(Path(_coef_path).with_suffix(".pdf")), bbox_inches="tight")
                    plt.close(_cfig)
                else:
                    st.image(entry["path"], caption=entry["caption"], width=500)

            # === Download Modeling Figures ===
            _model_paths = []
            for _res in model_results.values():
                _mtype = _res.get("model_type", "")
                if _mtype != "MLP":
                    for _key in ("cv_r2_plot_path", "cv_rmse_plot_path"):
                        _p = _res.get(_key)
                        if _p and os.path.exists(_p):
                            _model_paths.append(_p)
                _p = _res.get("cv_pred_plot_path")
                if _p and os.path.exists(_p):
                    _model_paths.append(_p)
                for _entry in _res.get("diagnostic_plots", []):
                    if os.path.exists(_entry["path"]):
                        _model_paths.append(_entry["path"])
            if _model_paths:
                st.download_button(
                    "⬇️ Download Modeling Figures as PDF",
                    data=pdf_paths_to_pdf_bytes(_model_paths),
                    file_name="modeling_figures.pdf",
                    mime="application/pdf",
                )


# === TAB 4: Prediction ===
from models.prediction_eval import evaluate_on_prediction_set
from preprocessors.aligner import align_xy, align_group_xy

def _flatten_preprocessed_to_matrix(preprocessed_dict):
    """Return (X, sample_ids) from {sample_id: Spectrum or [Spectrum, ...]}."""
    rows = []
    ids = []
    for sid, bundle in preprocessed_dict.items():
        if isinstance(bundle, list):           # list of Spectrum
            for sp in bundle:
                rows.append(sp.spectral_data)
                ids.append(sid)
        else:                                  # single Spectrum
            sp = bundle
            rows.append(sp.spectral_data)
            ids.append(sid)
    return np.vstack(rows), ids

if tab == "Prediction":
    st.header("Step 4: Predict on External Dataset")

    # === Source selection ===
    _holdout_active = st.session_state.get("holdout_active", False)
    _hg_raw         = st.session_state.get("holdout_group", "")
    _holdout_group  = ", ".join(str(g) for g in _hg_raw) if isinstance(_hg_raw, list) else str(_hg_raw)

    if _holdout_active:
        _pred_source = st.radio(
            "Prediction source:",
            options=["Upload zip", f"Use held-out group: {_holdout_group}"],
            horizontal=True,
        )
        use_holdout = _pred_source.startswith("Use held-out")
    else:
        use_holdout = False

    # === Upload Inputs (only when not using holdout) ===
    if not use_holdout:
        with st.expander("📁 Upload Prediction Spectra (.zip of .spc files)"):
            pred_zip_file = st.file_uploader("Upload a .zip file of Raman .spc files", type="zip", key="pred_zip")

        with st.expander("📄 (Optional) Upload Prediction Y-block (Excel with 'ID' column)"):
            pred_y_file = st.file_uploader("Upload reference Y file (optional)", type="xlsx", key="pred_y")
    else:
        pred_zip_file = None
        pred_y_file   = None

    # === Pull training-time choices ===
    trained_key      = st.session_state.get("trained_preprocess_key")
    trained_axis     = st.session_state.get("trained_axis")
    crop_region      = st.session_state.get("trained_crop_region", (800, 1800))
    deriv_order      = st.session_state.get("trained_deriv_order", 1)
    trained_avg_replicates   = st.session_state.get("trained_avg_replicates", False)
    trained_avg_replicates_2 = st.session_state.get("trained_avg_replicates_2", False)
    trained_is_group = (
        (trained_key == "1. Savgol-SNV-MeanCenter" and trained_avg_replicates) or
        (trained_key == "2. Baseline-Smooth-SNV" and trained_avg_replicates_2) or
        trained_key == "3. Average Replicates: EMSC"
    )
    preproc_state           = st.session_state.get("preproc_state")
    trained_emsc_ref_mode   = st.session_state.get("trained_emsc_ref_mode", "dataset")
    trained_emsc_class_col  = st.session_state.get("trained_emsc_class_col")
    trained_emsc_bl_method  = st.session_state.get("trained_emsc_bl_method", "lieber")
    trained_emsc_als_lam    = st.session_state.get("trained_emsc_als_lam", 6.31e5)
    trained_emsc_als_p      = st.session_state.get("trained_emsc_als_p", 0.01)
    trained_emsc_sg_smooth  = st.session_state.get("trained_emsc_sg_smooth", False)
    trained_emsc_sg_window  = st.session_state.get("trained_emsc_sg_window", 9)
    trained_emsc_sg_poly    = st.session_state.get("trained_emsc_sg_polyorder", 2)
    trained_emsc_mc         = st.session_state.get("trained_emsc_mean_center", True)

    # (optional) AsLS / Savgol params if you decide to store them from the Preprocessing tab
    asls_lambda  = st.session_state.get("trained_asls_lambda", 1e5)
    asls_p       = st.session_state.get("trained_asls_p", 0.001)
    sg_window    = st.session_state.get("trained_sg_window", 13)
    sg_polyorder = st.session_state.get("trained_sg_polyorder", 2)

    if trained_key is None:
        st.error("No trained preprocessing found. Please run the Preprocessing tab first.")

    if trained_key in PREPROCESS_OPTIONS:
        st.info(f"Using preprocessing from training: **{trained_key}**")
    else:
        st.error(f"Unrecognized preprocessing key: {trained_key}")
        st.stop()

    apply_correction = st.checkbox(
        "Apply slope/bias correction",
        value=False,
        help="Fits y_pred = slope × y_true + bias on the prediction set and reports corrected R² and RMSE alongside uncorrected. Requires a Y reference file.",
    )

    plot_style = st.radio(
        "Plot style:",
        options=["Standard", "Journal"],
        horizontal=True,
        help="Journal style shows prediction set as blue open circles and calibration set as grey open circles. Requires a Y reference file.",
    )

    if plot_style == "Journal":
        with st.expander("⚙️ Journal plot options", expanded=False):
            _js = st.session_state.get("journal_plot_style", {})
            _c1, _c2, _c3 = st.columns(3)
            with _c1:
                _ms_cal  = st.number_input("Cal marker size",  value=_js.get("marker_size_cal",  40), min_value=1, step=1)
                _ms_pred = st.number_input("Pred marker size", value=_js.get("marker_size_pred", 45), min_value=1, step=1)
            with _c2:
                _al_cal  = st.slider("Cal alpha",  0.0, 1.0, float(_js.get("alpha_cal",  0.7)), step=0.05)
                _al_pred = st.slider("Pred alpha", 0.0, 1.0, float(_js.get("alpha_pred", 1.0)), step=0.05)
            with _c3:
                _fw = st.number_input("Fig width",  value=_js.get("fig_width",  7.0), min_value=2.0, step=0.5, format="%.1f")
                _fh = st.number_input("Fig height", value=_js.get("fig_height", 6.0), min_value=2.0, step=0.5, format="%.1f")
                _tick_fontsize = st.number_input("Tick label size", value=_js.get("tick_fontsize", 10), min_value=6, step=1)

            _axis_auto = st.checkbox("Auto axis range", value=_js.get("axis_auto", True))
            _axis_lo = _axis_hi = None
            if not _axis_auto:
                _ac1, _ac2 = st.columns(2)
                with _ac1:
                    _axis_lo = st.number_input("Axis min", value=float(_js.get("axis_lo") or 0.0), format="%.3f")
                with _ac2:
                    _axis_hi = st.number_input("Axis max", value=float(_js.get("axis_hi") or 1.0), format="%.3f")

            _lc1, _lc2, _lc3 = st.columns(3)
            with _lc1:
                _xlabel = st.text_input("X axis label", value=_js.get("xlabel", "Actual Values"))
            with _lc2:
                _ylabel = st.text_input("Y axis label", value=_js.get("ylabel", "Predicted Values"))
            with _lc3:
                _custom_title = st.text_input("Plot title (blank = auto)", value=_js.get("custom_title", ""))

            _show_legend = st.checkbox("Show legend", value=_js.get("show_legend", True))

            st.session_state["journal_plot_style"] = {
                "marker_size_cal":  _ms_cal,
                "marker_size_pred": _ms_pred,
                "alpha_cal":        _al_cal,
                "alpha_pred":       _al_pred,
                "fig_width":        _fw,
                "fig_height":       _fh,
                "axis_auto":        _axis_auto,
                "axis_lo":          _axis_lo,
                "axis_hi":          _axis_hi,
                "xlabel":           _xlabel,
                "ylabel":           _ylabel,
                "custom_title":     _custom_title,
                "tick_fontsize":    _tick_fontsize,
                "show_legend":      _show_legend,
            }

    if st.button("Run Prediction"):
        if use_holdout:
            # === Source: held-out group from session state ===
            pred_sample_spectra = st.session_state["raw_spectra_holdout"]
        elif pred_zip_file is None:
            st.warning("Please upload a zip file of Raman spectra first.")
            st.stop()
        else:
            # === Source: uploaded zip ===
            pred_dir = st.session_state["pred_dir"]
            spectra_dir = os.path.join(pred_dir, "spectra")
            os.makedirs(spectra_dir, exist_ok=True)

            with zipfile.ZipFile(pred_zip_file, "r") as zip_ref:
                zip_ref.extractall(spectra_dir)

            pred_sample_spectra, _, _ = load_raman(pred_dir, spectra_dir)

        if True:  # always runs after spectra are ready
            pred_dir    = st.session_state["pred_dir"]
            spectra_dir = os.path.join(pred_dir, "spectra")
            os.makedirs(spectra_dir, exist_ok=True)

            # Grouping logic (only for method 3)
            if trained_is_group:
                from collections import defaultdict
                pred_sample_groups = defaultdict(list)
                for sample_id in pred_sample_spectra.keys():
                    group_id = sample_id.split("-")[0]
                    pred_sample_groups[group_id].append(sample_id)
            else:
                pred_sample_groups = None

            pre_func = PREPROCESS_OPTIONS[trained_key]

            # Initialise Y-block flags — resolved later after preprocessing
            _has_y = False
            y_pred_df = None

            # === Apply the SAME preprocessing & params ===
            if trained_key == "1. Savgol-SNV-MeanCenter":
                if trained_avg_replicates:
                    pred_preprocessed, pred_axis, _group_plot_dict = group_preprocess_savgol_snv_mc(
                        pred_sample_spectra, pred_sample_groups, spectra_dir,
                        crop_region=crop_region,
                        derivative_order=deriv_order,
                        use_state=preproc_state
                    )
                else:
                    pred_preprocessed, pred_axis = pre_func(
                        pred_sample_spectra, spectra_dir,
                        crop_region=crop_region,
                        derivative_order=deriv_order,
                        use_state=preproc_state
                    )

            elif trained_key == "2. Baseline-Smooth-SNV":
                # Stateless: no use_state needed
                if trained_avg_replicates_2:
                    pred_preprocessed, pred_axis, _group_plot_dict = group_preprocess_asls_savgol_snv(
                        pred_sample_spectra, pred_sample_groups, spectra_dir,
                        crop_region=crop_region,
                        asls_lambda=asls_lambda,
                        asls_p=asls_p,
                        sg_window=sg_window,
                        sg_polyorder=sg_polyorder
                    )
                else:
                    pred_preprocessed, pred_axis = pre_func(
                        pred_sample_spectra, spectra_dir,
                        crop_region=crop_region,
                        asls_lambda=asls_lambda,
                        asls_p=asls_p,
                        sg_window=sg_window,
                        sg_polyorder=sg_polyorder
                    )

            elif trained_key == "4. None":
                pred_preprocessed, pred_axis = preprocess_none(
                    pred_sample_spectra, spectra_dir,
                    crop_region=crop_region
                )

            elif trained_key == "3. Average Replicates: EMSC":
                # Build class_lookup for prediction samples (only needed for class mode)
                _pred_class_lookup = None
                if trained_emsc_ref_mode == "class" and trained_emsc_class_col:
                    if _has_y:
                        _pred_class_lookup = {}
                        for _idx in y_pred_df.index:
                            _gid = str(_idx).split("-")[0]
                            if trained_emsc_class_col in y_pred_df.columns:
                                _pred_class_lookup[_gid] = str(y_pred_df.loc[_idx, trained_emsc_class_col])
                    else:
                        st.info(
                            "No Y-block uploaded for prediction — class mode will fall back "
                            "to the first available training class reference."
                        )

                pred_preprocessed, pred_axis, _group_plot_dict = pre_func(
                    pred_sample_spectra, pred_sample_groups, spectra_dir,
                    crop_region=crop_region,
                    poly_ref_order=st.session_state.get("trained_poly_ref_order", 4),
                    emsc_p_order=st.session_state.get("trained_emsc_p_order", 6),
                    reference_mode=trained_emsc_ref_mode,
                    class_lookup=_pred_class_lookup,
                    ref_baseline_method=trained_emsc_bl_method,
                    als_lam=trained_emsc_als_lam,
                    als_p=trained_emsc_als_p,
                    apply_sg_smooth=trained_emsc_sg_smooth,
                    sg_window=trained_emsc_sg_window,
                    sg_polyorder=trained_emsc_sg_poly,
                    apply_mean_center=trained_emsc_mc,
                    use_state=preproc_state
                )

            else:
                st.error(f"Unhandled preprocessing key: {trained_key}")
                st.stop()

            # === Y-block alignment ===
            filtered_pred_sample_ids = list(pred_preprocessed.keys())
            Y_pred_true = None

            try:
                # Resolve Y source: holdout session state or uploaded file
                if use_holdout:
                    y_pred_df = st.session_state["y_block_holdout"].copy()
                    # Index is already set to ID
                    _has_y = True
                elif pred_y_file is not None:
                    y_pred_df = pd.read_excel(pred_y_file)
                    if "ID" not in y_pred_df.columns:
                        st.warning("Prediction Y file must contain an 'ID' column.")
                        st.stop()
                    y_pred_df["ID"] = y_pred_df["ID"].astype(str).str.strip()
                    y_pred_df.set_index("ID", inplace=True)
                    _has_y = True
                else:
                    _has_y = False

                if _has_y:
                    if trained_is_group:
                        # Group Y the SAME way as training (prefix before first "-")
                        num_cols = y_pred_df.select_dtypes(include=[np.number]).columns
                        y_pred_df["_group_id"] = y_pred_df.index.map(lambda s: str(s).split("-")[0])
                        y_pred_group = y_pred_df.groupby("_group_id")[num_cols].mean()

                        x_groups = set(map(str, pred_preprocessed.keys()))
                        y_groups = set(map(str, y_pred_group.index))
                        if x_groups.isdisjoint(y_groups):
                            st.warning("Heads-up: no overlap after grouping — check group prefixes/casing/whitespace.")

                        # align_group_xy returns 5 values
                        filtered_X_pred, filtered_Y_pred, filtered_pred_sample_ids, class_labels, unmatched_ids = align_group_xy(
                            pred_preprocessed, y_pred_group
                        )
                    else:
                        # align_xy returns 6 values
                        (filtered_X_pred, filtered_Y_pred,
                         filtered_pred_sample_ids, filtered_group_labels,
                         class_labels, unmatched_ids) = align_xy(pred_preprocessed, y_pred_df)

                    Y_pred_true = filtered_Y_pred.values if hasattr(filtered_Y_pred, "values") else filtered_Y_pred

                else:
                    # Build X if no Y provided
                    filtered_X_pred, filtered_pred_sample_ids = _flatten_preprocessed_to_matrix(pred_preprocessed)
                    Y_pred_true = None

            except Exception as e:
                st.warning(f"Y-block alignment failed: {e}")
                filtered_X_pred, filtered_pred_sample_ids = _flatten_preprocessed_to_matrix(pred_preprocessed)
                Y_pred_true = None

            # === Run Predictions ===
            model_results = st.session_state.get("model_results", {})
            axis = trained_axis if trained_axis is not None else pred_axis
            
            if "models_dir" not in st.session_state:
                st.error("Please set the Output Directory in the Data Loading tab.")
                st.stop()
                        
            results_dir = st.session_state["pred_dir"]
            
            if not model_results:
                st.warning("No trained models found. Please train models first in the Modeling tab.")
            else:
                prediction_outputs = []
                for analyte, result in model_results.items():
                    model_obj = result["model"]
                    y_mean = result["cv_results"]["y_mean"]
                    model_name = result.get("model_type", "PLS")

                    output = evaluate_on_prediction_set(
                        model=model_obj,
                        X_pred=filtered_X_pred,
                        y_mean=y_mean,
                        axis=axis,
                        analyte=analyte,
                        directory=results_dir,
                        Y_pred_true=Y_pred_true[:, filtered_Y_pred.columns.get_loc(analyte)] if Y_pred_true is not None else None,
                        model_name=model_name,
                        sample_ids=filtered_pred_sample_ids,
                        selected_mask=(
                            result["selection"].selected_mask
                            if result.get("selection") is not None else None
                        ),
                        apply_slope_bias_correction=apply_correction,
                    )

                    # Overwrite saved plot with journal style if selected and
                    # reference Y is available (needed for the cal set overlay)
                    if plot_style == "Journal" and output.get("y_true") is not None:
                        _jstyle = st.session_state.get("journal_plot_style", {})
                        plot_pred_vs_actual_journal(
                            y_true_pred=output["y_true"],
                            y_pred_pred=output["y_pred"],
                            y_true_cal=result["y_true"],
                            y_pred_cal=result["y_pred_cal"],
                            directory=results_dir,
                            title=f"External Predicted vs. Actual for {analyte} ({model_name})",
                            filename=f"External_Pred_vs_Actual_{model_name}_{analyte}.png",
                            style=_jstyle,
                        )
                        if output.get("y_pred_corrected") is not None:
                            plot_pred_vs_actual_journal(
                                y_true_pred=output["y_true"],
                                y_pred_pred=output["y_pred_corrected"],
                                y_true_cal=result["y_true"],
                                y_pred_cal=result["y_pred_cal"],
                                directory=results_dir,
                                title=f"Corrected Predicted vs. Actual for {analyte} ({model_name})",
                                filename=f"External_Pred_vs_Actual_Corrected_{model_name}_{analyte}.png",
                                style=_jstyle,
                            )

                    prediction_outputs.append(output)

                st.session_state["prediction_outputs"] = prediction_outputs

    # === Display Results (persists across tab switches) ===
    if "prediction_outputs" in st.session_state:
        st.subheader("📈 Prediction Results")
        for output in st.session_state["prediction_outputs"]:
            st.markdown(f"### {output['analyte']} ({output['model_name']})")

            if "r2_pred" in output:
                has_correction = output.get("r2_corrected") is not None
                if has_correction:
                    metrics_df = pd.DataFrame({
                        "": ["Uncorrected", "Corrected"],
                        "R²": [f"{output['r2_pred']:.4f}", f"{output['r2_corrected']:.4f}"],
                        "RMSE": [f"{output['rmsep']:.4f}", f"{output['rmse_corrected']:.4f}"],
                    }).set_index("")
                    st.table(metrics_df)
                    st.caption(f"Slope = {output['slope']:.4f}  |  Bias = {output['bias']:.4f}")
                else:
                    st.markdown(f"**R²_pred**: {output['r2_pred']:.4f}  \n**RMSEP**: {output['rmsep']:.4f}")

            if "csv_path" in output and os.path.exists(output["csv_path"]):
                df = pd.read_csv(output["csv_path"])
                st.dataframe(df)

            if output.get("y_true") is not None and output.get("y_pred") is not None:
                has_correction = output.get("y_pred_corrected") is not None
                if has_correction:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = plot_pred_vs_actual_interactive(
                            y_true=output["y_true"],
                            y_pred=output["y_pred"],
                            title=f"Uncorrected — {output['analyte']}",
                            sample_ids=output.get("sample_ids"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    with col2:
                        fig = plot_pred_vs_actual_interactive(
                            y_true=output["y_true"],
                            y_pred=output["y_pred_corrected"],
                            title=f"Corrected — {output['analyte']}",
                            sample_ids=output.get("sample_ids"),
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    fig = plot_pred_vs_actual_interactive(
                        y_true=output["y_true"],
                        y_pred=output["y_pred"],
                        title=f"Predicted vs Actual — {output['analyte']}",
                        sample_ids=output.get("sample_ids"),
                    )
                    st.plotly_chart(fig, use_container_width=True)

                # Journal plot display (shown below interactive when Journal style selected)
                if plot_style == "Journal":
                    _pp = output.get("pred_plot_path")
                    if _pp and os.path.exists(_pp):
                        st.image(_pp, caption="Journal plot", width=600)
                    _ppc = output.get("pred_plot_corrected_path")
                    if _ppc and os.path.exists(_ppc):
                        st.image(_ppc, caption="Journal plot (corrected)", width=600)

    # === Download Prediction Figures ===
    if "prediction_outputs" in st.session_state:
        _pred_paths = []
        for out in st.session_state["prediction_outputs"]:
            if "pred_plot_path" in out and os.path.exists(out["pred_plot_path"]):
                _pred_paths.append(out["pred_plot_path"])
            if "pred_plot_corrected_path" in out and out["pred_plot_corrected_path"] and os.path.exists(out["pred_plot_corrected_path"]):
                _pred_paths.append(out["pred_plot_corrected_path"])
        if _pred_paths:
            st.download_button(
                "⬇️ Download Prediction Figures as PDF",
                data=pdf_paths_to_pdf_bytes(_pred_paths),
                file_name="prediction_figures.pdf",
                mime="application/pdf",
            )


# === TAB 5: PCA Analysis ===
if tab == "PCA":
    st.header("Step 4: PCA Visualization")

    if "y_block" not in st.session_state:
        st.warning("Please load a Y-block first in the 'Data Loading' tab.")
    else:
        from models.wrappers import PCA_model, PCADA_model
        from plotting.plot_PCA import plot_pca_loadings, plot_yblock_pca_loadings, plot_pcada_cv_curve
        from models.cross_val import PCADA_CV
        from preprocessors.aligner import align_xy, align_group_xy
        import os

        st.markdown("Run PCA on your preprocessed spectra or Y-block data and visualize selected PCs with loadings.")

        # === User Controls ===
        with st.expander("⚙️ PCA Display Settings", expanded=True):
            _raman_ready = "preprocessed_spectra" in st.session_state
            _source_options = ["Raman spectra", "Y-block (fatty acids)"] if _raman_ready else ["Y-block (fatty acids)"]
            pca_source = st.radio("Data source", _source_options, index=0, horizontal=True)
            use_pcada = st.checkbox("PCA-DA (LDA on PCA scores)", value=False)
            show_ellipses = st.checkbox("Show 95% confidence ellipses", value=True)
            ellipse_alpha = st.select_slider(
                "Ellipse transparency",
                options=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
                value=0.25
            )
            top_n = st.number_input(
                "Number of top bands to label on loadings plot (per PC, 0 = none)",
                min_value=0, value=4, step=1
            )
            n_components = st.slider("Number of PCA components", min_value=2, max_value=15, value=5, step=1)
            if not use_pcada:
                col1, col2 = st.columns(2)
                with col1:
                    pc_x = st.number_input("PC for X-axis", min_value=1, max_value=n_components, value=1, step=1)
                with col2:
                    pc_y = st.number_input("PC for Y-axis", min_value=1, max_value=n_components, value=2, step=1)
            else:
                pc_x, pc_y = 1, 2
                _pca_da_n_folds = st.number_input("CV folds", min_value=2, max_value=20, value=5, step=1, key="pcada_n_folds")

        with st.expander("⚙️ PCA plot options", expanded=False):
            _ps = st.session_state.get("pca_plot_style", {})
            _pc1, _pc2, _pc3 = st.columns(3)
            with _pc1:
                _pca_tick_fs  = st.number_input("Tick label size",  value=_ps.get("tick_fontsize",  10), min_value=6, step=1, key="pca_tick_fs")
                _pca_label_fs = st.number_input("Axis label size",  value=_ps.get("label_fontsize", 12), min_value=6, step=1, key="pca_label_fs")
            with _pc2:
                _pca_pt_size = st.number_input("Point size", value=_ps.get("point_size", 50), min_value=1, step=1, key="pca_pt_size")
            with _pc3:
                _pca_fw_col, _pca_fh_col = st.columns(2)
                with _pca_fw_col:
                    _pca_fw = st.number_input("Fig width",  value=_ps.get("fig_width",  8.0), min_value=2.0, step=0.5, format="%.1f", key="pca_fw")
                with _pca_fh_col:
                    _pca_fh = st.number_input("Fig height", value=_ps.get("fig_height", 6.0), min_value=2.0, step=0.5, format="%.1f", key="pca_fh")
            _px1, _px2 = st.columns(2)
            with _px1:
                _pca_score_title = st.text_input("Score plot title (blank = default)", value=_ps.get("score_title", ""), key="pca_score_title")
            with _px2:
                _pca_loadings_title = st.text_input("Loadings plot title (blank = default)", value=_ps.get("loadings_title", ""), key="pca_loadings_title")
            _pca_show_legend = st.checkbox("Show legend", value=_ps.get("show_legend", True), key="pca_show_legend")
            st.session_state["pca_plot_style"] = {
                "tick_fontsize":  _pca_tick_fs,
                "label_fontsize": _pca_label_fs,
                "point_size":     _pca_pt_size,
                "fig_width":      _pca_fw,
                "fig_height":     _pca_fh,
                "score_title":    _pca_score_title,
                "loadings_title": _pca_loadings_title,
                "show_legend":    _pca_show_legend,
            }

        _pca_style       = st.session_state.get("pca_plot_style", {})
        pca_tick_fs      = _pca_style.get("tick_fontsize",  10)
        pca_label_fs     = _pca_style.get("label_fontsize", 12)
        pca_pt_size      = _pca_style.get("point_size",     50)
        pca_fw           = _pca_style.get("fig_width",       8.0)
        pca_fh           = _pca_style.get("fig_height",      6.0)
        pca_score_title    = _pca_style.get("score_title",    "").strip() or None
        pca_loadings_title = _pca_style.get("loadings_title", "").strip() or None
        pca_show_legend    = _pca_style.get("show_legend", True)

        # === Directory ===
        if "models_dir" not in st.session_state:
            st.error("Please set the Output Directory in the Data Loading tab.")
            st.stop()
        results_dir = st.session_state["models_dir"]

        # === Prepare Data Based on Source ===
        if pca_source == "Raman spectra":
            if "preprocessed_spectra" not in st.session_state:
                st.error("Raman spectra not available. Please run preprocessing first.")
                st.stop()
            raw_X = st.session_state["preprocessed_spectra"]
            axis = st.session_state["cropped_axis"]
            first_val = list(raw_X.values())[0]
            is_group_avg = not isinstance(first_val, list)
            if is_group_avg:
                raw_Y = st.session_state.get("y_block_grouped", st.session_state["y_block"])
            else:
                raw_Y = st.session_state["y_block"]
            if is_group_avg:
                filtered_X, _, _, classes, _ = align_group_xy(raw_X, raw_Y)
            else:
                filtered_X, _, _, _, classes, _ = align_xy(raw_X, raw_Y)
            feature_names = None
        else:
            from sklearn.preprocessing import StandardScaler as _SS
            raw_Y = st.session_state["y_block"]
            numeric_cols = raw_Y.select_dtypes(include="number").columns.tolist()
            if not numeric_cols:
                st.error("No numeric columns found in Y-block.")
                st.stop()
            _Y_num = raw_Y[numeric_cols].dropna()
            filtered_X = _SS().fit_transform(_Y_num.values)
            feature_names = numeric_cols
            axis = np.arange(len(numeric_cols))
            classes = (raw_Y.loc[_Y_num.index, "Class"].values
                       if "Class" in raw_Y.columns else np.zeros(len(_Y_num)))

        # === Run PCA / PCA-DA and Plot ===
        if use_pcada:
            _has_class = "Class" in st.session_state["y_block"].columns
            if not _has_class:
                st.warning("PCA-DA requires a ‘Class’ column in the Y block. Falling back to PCA.")
                use_pcada = False

        if use_pcada:
            st.subheader("📊 PCA-DA Score Plot (LD1 vs LD2)")
            pca_results = PCADA_model(
                X=filtered_X,
                classes=classes,
                axis=axis,
                directory=results_dir,
                n_components=n_components,
                show_ellipses=show_ellipses,
                ellipse_alpha=ellipse_alpha,
                label_fontsize=pca_label_fs,
                tick_fontsize=pca_tick_fs,
                point_size=pca_pt_size,
                fig_width=pca_fw,
                fig_height=pca_fh,
                title=pca_score_title,
                show_legend=pca_show_legend,
            )
            score_img = os.path.join(results_dir, "PCA_DA_LD1_vs_LD2.png")
        else:
            st.subheader(f"📊 PCA Score Plot (PC{pc_x} vs PC{pc_y})")
            pca_results = PCA_model(
                X=filtered_X,
                classes=classes,
                axis=axis,
                directory=results_dir,
                n_components=n_components,
                show_ellipses=show_ellipses,
                ellipse_alpha=ellipse_alpha,
                pc_x=pc_x,
                pc_y=pc_y,
                label_fontsize=pca_label_fs,
                tick_fontsize=pca_tick_fs,
                point_size=pca_pt_size,
                fig_width=pca_fw,
                fig_height=pca_fh,
                title=pca_score_title,
                show_legend=pca_show_legend,
            )
            score_img = os.path.join(results_dir, f"PCA_PC{pc_x}_vs_PC{pc_y}.png")

        st.image(score_img, width=800)

        # === PCA-DA CV Accuracy Curve + Confusion Matrix ===
        if use_pcada:
            st.subheader("📈 PCA-DA Cross-Validation")
            _pcada_groups = st.session_state.get("groups")
            _pcada_n_folds = int(st.session_state.get("pcada_n_folds", 5))

            _cv_accuracies = PCADA_CV(
                X=filtered_X,
                classes=classes,
                max_components=n_components,
                n_folds=_pcada_n_folds,
                groups=_pcada_groups,
            )

            _cv_fig = plot_pcada_cv_curve(
                accuracies=_cv_accuracies,
                selected_n=n_components,
                directory=results_dir,
            )
            _cv_buf = io.BytesIO()
            _cv_fig.savefig(_cv_buf, format="png", dpi=150, bbox_inches="tight")
            _cv_buf.seek(0)
            st.image(_cv_buf, width=700)
            plt.close(_cv_fig)

            # Confusion matrix from full-data LDA predictions
            from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
            _lda_model  = pca_results["lda_model"]
            _pca_model  = pca_results["pca_model"]
            _X_lda_pred = _lda_model.predict(_pca_model.transform(filtered_X))
            _cm = confusion_matrix(classes, _X_lda_pred, labels=np.unique(classes))
            _cm_fig, _cm_ax = plt.subplots(figsize=(max(4, len(np.unique(classes))),
                                                     max(3, len(np.unique(classes)))))
            ConfusionMatrixDisplay(confusion_matrix=_cm,
                                   display_labels=np.unique(classes)).plot(ax=_cm_ax, colorbar=False)
            _cm_ax.set_title("Confusion Matrix (training data)")
            _cm_fig.tight_layout()
            _cm_buf = io.BytesIO()
            _cm_fig.savefig(_cm_buf, format="png", dpi=150, bbox_inches="tight")
            _cm_buf.seek(0)
            st.image(_cm_buf, width=500)
            plt.close(_cm_fig)

        # === Loadings Plot ===
        st.subheader("📈 PCA Loadings Plot")
        if pca_source == "Raman spectra":
            plot_pca_loadings(
                pca_model=pca_results["pca_model"],
                axis=axis,
                directory=results_dir,
                components=[pc_x - 1, pc_y - 1],
                top_n=top_n,
                label_fontsize=pca_label_fs,
                tick_fontsize=pca_tick_fs,
                fig_width=pca_fw,
                fig_height=pca_fh,
                title=pca_loadings_title,
                show_legend=pca_show_legend,
            )
            loadings_img = os.path.join(results_dir, "PCA_Loadings_Annotated.png")
        else:
            plot_yblock_pca_loadings(
                pca_model=pca_results["pca_model"],
                feature_names=feature_names,
                directory=results_dir,
                components=[pc_x - 1, pc_y - 1],
                top_n=top_n,
                label_fontsize=pca_label_fs,
                tick_fontsize=pca_tick_fs,
                fig_width=pca_fw,
                fig_height=pca_fh,
                title=pca_loadings_title,
                show_legend=pca_show_legend,
            )
            loadings_img = os.path.join(results_dir, "PCA_Loadings_YBlock_Annotated.png")
        st.image(loadings_img, width=800)

        _pca_paths = [p for p in [score_img, loadings_img] if os.path.exists(p)]
        if _pca_paths:
            st.download_button(
                "⬇️ Download PCA Figures as PDF",
                data=pdf_paths_to_pdf_bytes(_pca_paths),
                file_name="pca_figures.pdf",
                mime="application/pdf",
            )
