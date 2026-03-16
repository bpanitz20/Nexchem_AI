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
import re
from plotting.plot_raw import plot_spectra_colored_by_analyte
from preprocessors.raman_preprocess import (
    preprocess_savgol_emsc_mc,
    preprocess_savgol_snv_mc,
    group_preprocess_savgol_emsc_mc,
    group_preprocess_savgol_snv_mc,
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
    "3. Average Replicates: Savgol-SNV-MeanCenter": group_preprocess_savgol_snv_mc,
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

    ### 🧭 User Guide
    Follow the tabs on the left side of the page. Below are instructions for each step:

    **1️⃣ Load Calibration Data**
    - Upload a `.zip` containing your Thermo **Raman spectra (.spc)** files.  
    - Upload an **Excel Y-block** containing target concentrations.  
      - Make sure samples use the correct naming convention (see bottom).  
      - Ensure sample names in the spectra and Excel **ID** column match exactly.  
    - Visualize raw data to verify spectra were loaded correctly.

    **2️⃣ Preprocess Spectra**
    - Choose one of the preprocessing pipelines (e.g., Savitzky-Golay, SNV, EMSC).  
    - Adjust parameters or leave defaults.  
    - Visualize preprocessed spectra.  
    - ⚠️ *If switching preprocessing pipelines causes an error, refresh the app and reload your data.*

    **3️⃣ Modeling**
    - Choose model parameters or leave blank to use automatic selection.  
    - Choose CV parameters or use defaults.  
    - Visualize predicted vs. actual plots and model performance metrics.

    **4️⃣ Prediction Tab**
    - Upload new Raman spectra to generate predictions from your saved calibration model.  
    - If you upload a Y-block for the prediction set, external predicted vs. actual plots will be generated.

    **5️⃣ PCA Tab**
    - Generates a PCA plot using your spectral data.  
    - Select which principal components (PCs) to display and view loadings.  
    - Uses only the **Class** column from the Y-block for grouping and coloring.

    ---

    ### 🧩 File Naming Convention
    Raman spectra should follow this format:
    ```
    SampleID-Replicate_.spc
    ```

    **Example:**
    ```
    188-1_450mw_10s.spc
    188-2_450mw_10s.spc
    188-3_450mw_10s.spc
    190-1_450mw_10s.spc
    190-2_450mw_10s.spc
    190-3_450mw_10s.spc
    ```

    - Anything can come after the underscore (`_`) — for example, acquisition parameters.  
    - NexChem_AI automatically detects the number after the dash for replicate grouping and visualization.  
    - If you are using a different experimental design (e.g., time series), you can still use this format to separate samples in CV and visualization.

    ---

    ### 📊 Y-Block Excel Format
    The Y-block must contain at least these columns:

    | ID    | Class | DHA | EPA | PUFA |
    |:------|:------|----:|----:|----:|
    | 188-1 | 2022  | 10.5 | 5.1 | 15.6 |
    | 188-2 | 2022  | 10.5 | 5.1 | 15.6 |
    | 188-3 | 2022  | 10.5 | 5.1 | 15.6 |
    | 190-1 | 2023  | 1.5  | 1.0 | 2.5  |
    | 190-2 | 2023  | 1.5  | 1.0 | 2.5  |
    | 190-3 | 2023  | 1.5  | 1.0 | 2.5  |

    - The **ID** column must match the numeric portion of the corresponding spectra names.  
    - **Class** (optional): used for visualization or grouped CV.  
    - Columns such as **DHA**, **EPA**, or **PUFA** serve as target analytes.  
    - NexChem_AI automatically loops through each column after the **Class** column and builds one model per target.  
      *Only one target column is required.*

    ---

    ### 💡 Tips
    - Use consistent **instrument settings** across calibration and prediction runs.  
    - Keep all files for a calibration set in the same `.zip`.  
    - Export results and plots directly from the **Results** directory.

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

        # === Spectra Overlay: Color by replicate or analyte ===
        _dl_figs: list = []
        st.subheader("📊 Spectra Overlay")

        color_mode = st.radio(
            "Color spectra by:",
            ["Replicate", "Analyte (Y-block)"],
            index=0,
            horizontal=True,
            key="raw_color_mode"
        )

        # --- Option 1: color by replicate (pre-made image, if exists) ---
        if color_mode == "Replicate":
            overlay_path = os.path.join("ignore", "Overlay_Raw.png")  
            
            fig, ax = plt.subplots(figsize=(8, 5))
            for sid, spectra in sample_spectra.items():
                for spectrum in spectra:
                    ax.plot(spectrum.spectral_axis, spectrum.spectral_data, alpha=0.7)
            ax.set_title("Raw Spectra (Colored by Replicate)")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity")

            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, caption="Spectra colored by replicate", width=800)
            _dl_figs.append(fig)
            plt.close(fig)

        # --- Option 2: color by analyte from Y-block ---
        else:
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
                return int(match.group(1)), int(match.group(2))
            return sample_id  # fallback

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
        if selected_method in ["1. Savgol-SNV-MeanCenter",
                               "3. Average Replicates: Savgol-SNV-MeanCenter"]:
            deriv_order = st.selectbox("Derivative order", options=[0, 1, 2], index=1)

        # AsLS/SavGol parameters (optional)
        if selected_method == "2. Baseline-Smooth-SNV":
            asls_lambda = st.number_input("AsLS lambda", value=1e5, format="%.1e")
            asls_p = st.number_input("AsLS asymmetry p", value=0.001, format="%.3f")
            sg_window = st.number_input("Savitzky–Golay window length", value=13)
            sg_polyorder = st.number_input("Savitzky–Golay polynomial order", value=2)

        run_button = st.button("Run Preprocessing")

        if run_button:
            sample_spectra = st.session_state["raw_spectra"]
            spectra_dir = st.session_state["preproc_spectra_dir"]
            os.makedirs(spectra_dir, exist_ok=True)

            # === SELECTED PIPELINE ===
            if selected_method == "1. Savgol-SNV-MeanCenter":
                preprocessed_spectra, cropped_axis, preproc_state = preprocess_savgol_snv_mc(
                    sample_spectra, spectra_dir,
                    crop_region=crop_region,
                    derivative_order=deriv_order,
                    return_state=True
                )
                st.session_state["preproc_state"] = preproc_state

            elif selected_method == "2. Baseline-Smooth-SNV":
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

            elif selected_method == "3. Average Replicates: Savgol-SNV-MeanCenter":
                # Create group dict
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
                st.session_state["preproc_state"] = preproc_state

                st.session_state["group_plots"] = group_plot_dict
                st.session_state["y_block_grouped"] = avg_y_block(st.session_state["y_block"])
                st.session_state["sample_groups"] = sample_groups

            elif selected_method == "4. None":
                preprocessed_spectra, cropped_axis = preprocess_none(
                    sample_spectra, spectra_dir,
                    crop_region=crop_region
                )

            # === Save results to session ===
            st.session_state["trained_preprocess_key"] = selected_method
            st.session_state["trained_crop_region"] = crop_region
            st.session_state["trained_deriv_order"] = deriv_order
            st.session_state["trained_axis"] = cropped_axis
            st.session_state["preprocessed_spectra"] = preprocessed_spectra
            st.session_state["cropped_axis"] = cropped_axis
            st.session_state["preprocessing_done"] = True
            st.session_state["trained_is_group"] = (
                selected_method == "3. Average Replicates: Savgol-SNV-MeanCenter"
                )

            st.success(f"✅ Preprocessing complete using: {selected_method}")
            st.write(f"Processed {len(preprocessed_spectra)} entries")

        # === Display Results ===
        if st.session_state.get("preprocessing_done", False):
            preprocessed_spectra = st.session_state["preprocessed_spectra"]
            y_df = st.session_state.get("y_block", None)

            # ---------- Overlay with color-mode toggle ----------
            _prep_figs: list = []
            st.subheader("📊 Preprocessed Spectra Overlay")

            color_options = ["Replicate"]
            if y_df is not None:
                color_options.append("Analyte (Y-block)")

            color_mode = st.radio(
                "Color spectra by:",
                color_options,
                index=0,
                horizontal=True,
                key="preproc_color_mode"
            )

            if color_mode == "Replicate":
                # Neutral overlay, all spectra same color
                fig, ax = plt.subplots(figsize=(8, 5))
                for sid, spectra in preprocessed_spectra.items():
                    if isinstance(spectra, Spectrum):
                        spectra = [spectra]
                    for spectrum in spectra:
                        ax.plot(spectrum.spectral_axis, spectrum.spectral_data, alpha=0.7)

                ax.set_title("Overlay of All Preprocessed Spectra (Colored by Replicate)")
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity")

                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, width=800)
                _prep_figs.append(fig)
                plt.close(fig)

            else:
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

        st.header("Step 3: Build Regression Model")

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
            enable_grid_customization = st.checkbox("Customize MLP Parameter Grid?", value=False)
            if enable_grid_customization:
                
                # PLS X-block compression
                pls_components = st.multiselect(
                    "PLS components (X-block compression):",
                    options=[2, 3, 4, 5, 6, 7, 8, 10, 12, 14],
                    default=[6]
                )
                
                # Nodes per layer 
                nodes_first = st.multiselect(
                    "Nodes 1st layer:",
                    options=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16],
                    default=[10]      # you can set [2] if you want strictly 2
                )
        
                nodes_second = st.multiselect(
                    "Nodes 2nd layer (0 = none):",
                    options=[0, 1, 2, 3, 4, 5, 6],
                    default=[0]      # 0 = single layer, 2 = 2nd layer with 2 nodes
                )
        
                # Build hidden_layer_sizes list like Eigenvector
                hidden_layer_sizes = []
                for n1 in nodes_first:
                    for n2 in nodes_second:
                        if n2 == 0:
                            hidden_layer_sizes.append((n1,))
                        else:
                            hidden_layer_sizes.append((n1, n2))

                alpha_values = st.multiselect("Alpha values:", options=[0.029, 0.03, 0.03, 0.032, 0.035, 0.36, 0.037, 0.038, 0.04, 0.045, 0.05, 0.55, 0.6], default=[0.03])
                learning_rates = st.multiselect("Learning rates:", options=[  0.002, 0.003, 0.0035, 0.004, 0.0042, 0.0043, 0.0044, 0.0045, 0.0046, 0.0047, 0.005, 0.0052, 0.0054, 0.006], default=[0.004])

                activations = st.multiselect(
                    "Activation functions:",
                    options=["relu", "tanh"],
                    default=["relu"]
                )


                param_grid = {
            'pls__n_components': pls_components,          
            'mlp__hidden_layer_sizes': hidden_layer_sizes,
            'mlp__activation': activations,
            'mlp__alpha': alpha_values,
            'mlp__learning_rate_init': learning_rates,
            'mlp__early_stopping': [True],
            'mlp__solver': ['adam']
        }

        # === Cross-validation options ===
        st.subheader("Choose Cross-Validation")
        use_group_kfold = st.checkbox("Use Grouped K-Fold CV?", value=False)
        
        group_strategy = None
        if use_group_kfold:
            group_strategy = st.radio("Group K-Fold by:", options=["Replicate", "Class"], index=0)
                
                
        n_folds = st.number_input("Number of K-Folds:", min_value=2, max_value=20, value=8)

        if model_name == "PLS":
            st.subheader("Variable Selection")
            use_block = st.checkbox(
                "iPLS",
                value=False,
                help="Bottom-up variable selection with iPLS.",
            )
            if use_block:
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
            
            for analyte in model_results.keys():
                result = model_results[analyte]
                plot_paths = []
                captions = []

                if model_name != "MLP":
                    r2_plot = result.get("cv_r2_plot_path")
                    rmse_plot = result.get("cv_rmse_plot_path")
                    if r2_plot and os.path.exists(r2_plot):
                        plot_paths.append(r2_plot)
                        captions.append(f"CV R² vs n_components for {analyte}")
                    if rmse_plot and os.path.exists(rmse_plot):
                        plot_paths.append(rmse_plot)
                        captions.append(f"CV RMSE vs n_components for {analyte}")
    
                pred_plot = result.get("cv_pred_plot_path")
                if pred_plot and os.path.exists(pred_plot):
                    plot_paths.append(pred_plot)
                    captions.append(f"CV Predicted vs Actual for {analyte}")
                    
                    if plot_paths:
                        cols = st.columns(len(plot_paths))
                        for i, (col, path) in enumerate(zip(cols, plot_paths)):
                            with col:
                                st.image(path, caption=captions[i], use_container_width=True)    

            # === Diagnostic Plots ===
            st.subheader("📉 Loadings, Variables & Scores")

            for analyte in model_results.keys():
                result = model_results[analyte]
                diag_plots = result.get("diagnostic_plots", [])
                if not diag_plots:
                    continue

                model_type = result.get("model_type", "")
                st.markdown(f"**🔬 {analyte} ({model_type})**")

                for i in range(0, len(diag_plots), 2):
                    pair = diag_plots[i : i + 2]
                    cols = st.columns(len(pair))
                    for col, entry in zip(cols, pair):
                        if os.path.exists(entry["path"]):
                            with col:
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

    # === Upload Inputs ===
    with st.expander("📁 Upload Prediction Spectra (.zip of .spc files)"):
        pred_zip_file = st.file_uploader("Upload a .zip file of Raman .spc files", type="zip", key="pred_zip")

    with st.expander("📄 (Optional) Upload Prediction Y-block (Excel with 'ID' column)"):
        pred_y_file = st.file_uploader("Upload reference Y file (optional)", type="xlsx", key="pred_y")

    # === Pull training-time choices ===
    trained_key      = st.session_state.get("trained_preprocess_key")
    trained_axis     = st.session_state.get("trained_axis")
    crop_region      = st.session_state.get("trained_crop_region", (800, 1800))
    deriv_order      = st.session_state.get("trained_deriv_order", 1)
    trained_is_group = (
    trained_key == "3. Average Replicates: Savgol-SNV-MeanCenter"
    )
    preproc_state    = st.session_state.get("preproc_state")

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

    if st.button("Run Prediction"):
        if pred_zip_file is None:
            st.warning("Please upload a zip file of Raman spectra first.")
        else:
            # === Extract and Load Spectra ===
            pred_dir = st.session_state["pred_dir"]
                      
            spectra_dir = os.path.join(pred_dir, "spectra")
            os.makedirs(spectra_dir, exist_ok=True)

            with zipfile.ZipFile(pred_zip_file, "r") as zip_ref:
                zip_ref.extractall(spectra_dir)

            pred_sample_spectra, _, _ = load_raman(pred_dir, spectra_dir)

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

            # === Apply the SAME preprocessing & params ===
            if trained_key == "1. Savgol-SNV-MeanCenter":
                pred_preprocessed, pred_axis = pre_func(
                    pred_sample_spectra, spectra_dir,
                    crop_region=crop_region,
                    derivative_order=deriv_order,
                    use_state=preproc_state
                )

            elif trained_key == "2. Baseline-Smooth-SNV":
                # Stateless: no use_state needed
                pred_preprocessed, pred_axis = pre_func(
                    pred_sample_spectra, spectra_dir,
                    crop_region=crop_region,
                    asls_lambda=asls_lambda,
                    asls_p=asls_p,
                    sg_window=sg_window,
                    sg_polyorder=sg_polyorder
                )

            elif trained_key == "3. Average Replicates: Savgol-SNV-MeanCenter":
                pred_preprocessed, pred_axis, _group_plot_dict = pre_func(
                    pred_sample_spectra, pred_sample_groups, spectra_dir,
                    crop_region=crop_region,
                    derivative_order=deriv_order,
                    use_state=preproc_state
                )

            elif trained_key == "4. None":
                pred_preprocessed, pred_axis = preprocess_none(
                    pred_sample_spectra, spectra_dir,
                    crop_region=crop_region
                )

            else:
                st.error(f"Unhandled preprocessing key: {trained_key}")
                st.stop()

            # === Optional Y-block alignment ===
            filtered_pred_sample_ids = list(pred_preprocessed.keys())
            Y_pred_true = None

            try:
                if pred_y_file is not None:
                    y_pred_df = pd.read_excel(pred_y_file)
                    if "ID" not in y_pred_df.columns:
                        st.warning("Prediction Y file must contain an 'ID' column.")
                        st.stop()

                    # normalize ID strings
                    y_pred_df["ID"] = y_pred_df["ID"].astype(str).str.strip()
                    y_pred_df.set_index("ID", inplace=True)

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
                    )
                    prediction_outputs.append(output)

                st.session_state["prediction_outputs"] = prediction_outputs

                # === Display Results ===
                st.subheader("📈 Prediction Results")
                for output in prediction_outputs:
                    st.markdown(f"### {output['analyte']} ({output['model_name']})")

                    if "r2_pred" in output:
                        st.markdown(f"**R²_pred**: {output['r2_pred']:.4f}  \n**RMSEP**: {output['rmsep']:.4f}")

                    if "csv_path" in output and os.path.exists(output["csv_path"]):
                        df = pd.read_csv(output["csv_path"])
                        st.dataframe(df)

                    if "pred_plot_path" in output and os.path.exists(output["pred_plot_path"]):
                        st.image(output["pred_plot_path"], caption="Predicted vs Actual", use_container_width=True)

    # === Download Prediction Figures ===
    if "prediction_outputs" in st.session_state:
        _pred_paths = [
            out["pred_plot_path"]
            for out in st.session_state["prediction_outputs"]
            if "pred_plot_path" in out and os.path.exists(out["pred_plot_path"])
        ]
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

    if "preprocessed_spectra" not in st.session_state or "y_block" not in st.session_state:
        st.warning("Please run preprocessing first in the 'Preprocessing' tab.")
    else:
        # ✅ Correct imports
        from models.wrappers import PCA_model
        from plotting.plot_PCA import plot_pca_loadings
        from preprocessors.aligner import align_xy, align_group_xy
        import os

        st.markdown("Run PCA on your preprocessed spectra and visualize selected PCs with loadings.")

        # === User Controls ===
        with st.expander("⚙️ PCA Display Settings", expanded=True):
            show_ellipses = st.checkbox("Show 95% confidence ellipses", value=True)
            ellipse_alpha = st.select_slider(
                "Ellipse transparency",
                options=[0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5],
                value=0.25
            )
            top_n = st.selectbox(
                "Number of top bands to label on loadings plot (per PC)",
                options=list(range(1, 11)),
                index=3
            )
            # 🔽 New: how many PCs to compute
            n_components = st.slider("Number of PCA components", min_value=2, max_value=15, value=5, step=1)
            # 🔽 New: which PCs to plot (1-based for users)
            col1, col2 = st.columns(2)
            with col1:
                pc_x = st.number_input("PC for X-axis", min_value=1, max_value=n_components, value=1, step=1)
            with col2:
                pc_y = st.number_input("PC for Y-axis", min_value=1, max_value=n_components, value=2, step=1)

        # === Get Preprocessed Data ===
        raw_X = st.session_state["preprocessed_spectra"]
        axis = st.session_state["cropped_axis"]

        # Determine replicate structure
        first_val = list(raw_X.values())[0]
        is_group_avg = not isinstance(first_val, list)

        # Use group-averaged Y when spectra are group-averaged
        if is_group_avg:
            raw_Y = st.session_state.get("y_block_grouped", st.session_state["y_block"])
        else:
            raw_Y = st.session_state["y_block"]

        if is_group_avg:
            filtered_X, _, _, classes, _ = align_group_xy(raw_X, raw_Y)
        else:
            filtered_X, _, _, _, classes, _ = align_xy(raw_X, raw_Y)

        #Directory
        if "models_dir" not in st.session_state:
            st.error("Please set the Output Directory in the Data Loading tab.")
            st.stop()
        
        results_dir = st.session_state["models_dir"]

        # === Run PCA and Plot ===
        st.subheader(f"📊 PCA Score Plot (PC{pc_x} vs PC{pc_y})")
        pca_results = PCA_model(
            X=filtered_X,
            classes=classes,
            axis=axis,
            directory=results_dir,
            n_components=n_components,
            show_ellipses=show_ellipses,
            ellipse_alpha=ellipse_alpha,
            pc_x=pc_x,              # 🔽 pass through
            pc_y=pc_y               # 🔽 pass through
        )
        # Match the PCA_model’s dynamic filename
        score_img = os.path.join(results_dir, f"PCA_PC{pc_x}_vs_PC{pc_y}.png")
        st.image(score_img, width=800)

        # === Loadings Plot ===
        st.subheader("📈 PCA Loadings Plot")
        plot_pca_loadings(
            pca_model=pca_results["pca_model"],
            axis=axis,
            directory=results_dir,
            components=[pc_x - 1, pc_y - 1],  # 🔽 show loadings for the same PCs
            top_n=top_n
        )
        loadings_img = os.path.join(results_dir, "PCA_Loadings_Annotated.png")
        st.image(loadings_img, width=800)

        _pca_paths = [p for p in [score_img, loadings_img] if os.path.exists(p)]
        if _pca_paths:
            st.download_button(
                "⬇️ Download PCA Figures as PDF",
                data=pdf_paths_to_pdf_bytes(_pca_paths),
                file_name="pca_figures.pdf",
                mime="application/pdf",
            )
