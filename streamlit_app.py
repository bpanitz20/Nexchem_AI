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
from preprocessors.raman_preprocess import (
    preprocess_pipeline_1,
    preprocess_pipeline_2,
    group_preprocess,
    group_preprocess_2,
    avg_y_block,
    preprocess_none
)
from collections import defaultdict



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

    with st.expander("📁 Upload Calibration Raman Spectra (.zip of .spc files)"):
        zip_file = st.file_uploader("Upload a .zip file", type="zip")

    with st.expander("📄 Upload Calibration Y-block (Excel with 'ID' column)"):
        y_file = st.file_uploader("Upload GCMS target file", type="xlsx")

    if zip_file and y_file:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "calib_data.zip")
            with open(zip_path, "wb") as f:
                f.write(zip_file.read())

            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(os.path.join(tmpdir, "unzipped"))

            raman_path = os.path.join(tmpdir, "unzipped")
            spectra_dir = os.path.join(tmpdir, "ignore")
            os.makedirs(spectra_dir, exist_ok=True)

            # ⬇️ Load raw spectra only (no preprocessing yet!)
            sample_spectra, shifts, sample_groups = load_raman(raman_path, spectra_dir)

            # Load and validate Y-block
            y_df = pd.read_excel(y_file)
            y_df.set_index("ID", inplace=True)

            st.success("✅ Raw calibration data loaded.")
            st.write(f"**Raman spectra loaded**: {len(sample_spectra)} samples")
            st.write(f"**GCMS targets loaded**: {y_df.shape[0]} rows")

            # Optional: Display Y-block data
            with st.expander("🔬 Preview GC-MS Y-block"):
                st.dataframe(y_df, use_container_width=True)
                
            # === Automatically Display Overlay ===
            overlay_path = os.path.join(spectra_dir, "Overlay_Raw.png")
            if os.path.exists(overlay_path):
                st.subheader("📊 Overlay of All Raw Spectra")
                st.image(overlay_path, width=800)
                
            # === Interactive Raw Overlay ===
            st.subheader("🔍 Raw Spectra Visualization")

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
            
                ax.set_title("Overlay of Selected Raw Spectra")
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity")
                ax.legend(loc="best", fontsize="small")
            
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                buf.seek(0)
                st.image(buf, caption=None, width=800)  # Control actual width in pixels
                plt.close(fig)

            # ✅ Store raw data for next step
            st.session_state["raw_spectra"] = sample_spectra
            st.session_state["y_block"] = y_df

# === TAB 2: Preprocessing ===
if tab == "Preprocessing":
    if "raw_spectra" not in st.session_state or "y_block" not in st.session_state:
        st.warning("Please load calibration data in the 'Data Loading' tab first.")
    else:
        st.header("Step 2: Preprocess Spectra")

        from ramanspy import Spectrum

        preprocess_options = {
            "1. Savgol-SNV-MeanCenter": preprocess_pipeline_2,
            "2. Savgol-EMSC": preprocess_pipeline_1,
            "3. Average Replicates: Savgol-SNV-MeanCenter": group_preprocess_2,
            "4. Average Replicates: Savgol-EMSC": group_preprocess,
            "5. None": preprocess_none
        }

        selected_method = st.selectbox("Choose preprocessing method:", list(preprocess_options.keys()))

        # === Common Parameters ===
        crop_min = st.number_input("Crop region min (cm⁻¹)", value=800)
        crop_max = st.number_input("Crop region max (cm⁻¹)", value=1800)
        crop_region = (crop_min, crop_max)

        # === Method-Specific Parameters ===
        deriv_order = None
        emsc_p_order = None

        if selected_method in ["1. Savgol-SNV-MeanCenter", "3. Average Replicates: Savgol-SNV-MeanCenter"]:
            deriv_order = st.selectbox("Derivative order", options=[0, 1, 2], index=1)

        if selected_method in ["2. Savgol-EMSC", "4. Average Replicates: Savgol-EMSC"]:
            deriv_order = st.selectbox("Derivative order", options=[0, 1, 2], index=1)
            emsc_p_order = st.number_input("EMSC polynomial order", min_value=1, max_value=6, value=2)

        run_button = st.button("Run Preprocessing")

        if run_button:
            sample_spectra = st.session_state["raw_spectra"]
            y_df = st.session_state["y_block"]
            spectra_dir = "./temp_preprocess"
            os.makedirs(spectra_dir, exist_ok=True)

            # === Run Preprocessing ===
            if selected_method == "1. Savgol-SNV-MeanCenter":
                preprocessed_spectra, cropped_axis = preprocess_pipeline_2(
                    sample_spectra, spectra_dir,
                    crop_region=crop_region, derivative_order=deriv_order
                )

            elif selected_method == "3. Average Replicates: Savgol-SNV-MeanCenter":
                sample_groups = defaultdict(list)
                for sample_id in sample_spectra.keys():
                    group_id = sample_id.split("-")[0]
                    sample_groups[group_id].append(sample_id)
                preprocessed_spectra, cropped_axis, group_plot_dict = group_preprocess_2(
                sample_spectra, sample_groups, spectra_dir,
                crop_region=crop_region, derivative_order=deriv_order
                )
                st.session_state["group_plots"] = group_plot_dict  # store the figures for UI
                group_avg_Y = avg_y_block(st.session_state["y_block"])
                st.session_state["y_block"] = group_avg_Y
                

            elif selected_method == "2. Savgol-EMSC":
                preprocessed_spectra, cropped_axis = preprocess_pipeline_1(
                    sample_spectra, spectra_dir,
                    crop_region=crop_region,
                    emsc_p_order=emsc_p_order,
                    deriv_order=deriv_order
                )

            elif selected_method == "4. Average Replicates: Savgol-EMSC":
                sample_groups = defaultdict(list)
                for sample_id in sample_spectra.keys():
                    group_id = sample_id.split("-")[0]
                    sample_groups[group_id].append(sample_id)

                preprocessed_spectra, cropped_axis, group_replicates, group_plot_dict = group_preprocess(
                    sample_spectra, sample_groups, spectra_dir,
                    crop_region=crop_region,
                    emsc_p_order=emsc_p_order,
                    deriv_order=deriv_order
                    )

                st.session_state["group_plots"] = group_plot_dict
                group_avg_Y = avg_y_block(st.session_state["y_block"])
                st.session_state["y_block"] = group_avg_Y
  
            elif selected_method == "5. None":
                preprocessed_spectra, cropped_axis = preprocess_none(
                    sample_spectra, spectra_dir,
                    crop_region=crop_region,
                    )

            st.session_state["preprocessed_spectra"] = preprocessed_spectra
            st.session_state["cropped_axis"] = cropped_axis
            st.session_state["preprocessing_done"] = True

            st.success(f"✅ Preprocessing complete using: {selected_method}")
            st.write(f"Processed {len(preprocessed_spectra)} entries")

        # === Display Results if available ===
        if st.session_state.get("preprocessing_done", False):
            preprocessed_spectra = st.session_state["preprocessed_spectra"]

            st.subheader("📊 Overlay of All Preprocessed Spectra")
            fig, ax = plt.subplots(figsize=(8, 5))
            for sample_id, spectra in preprocessed_spectra.items():
                if isinstance(spectra, Spectrum):
                    spectra = [spectra]
                for spectrum in spectra:
                    ax.plot(spectrum.spectral_axis, spectrum.spectral_data, alpha=0.7)
            
            ax.relim()
            ax.autoscale_view()
            ax.set_title("Overlay of All Preprocessed Spectra")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity")
            
            buf = io.BytesIO()
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            st.image(buf, width=800)
            plt.close(fig)

            st.subheader("🔍 Preprocessed Spectra Visualization")

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
            
                # Check if all selected samples are group-averaged
                if all(sample_id in group_plots for sample_id in selected_ids):
                    # Option: show individual group plots stacked with fixed width
                    for sample_id in selected_ids:
                        st.subheader(f"Overlay Plot for Group: {sample_id}")
                        fig = group_plots[sample_id]
                        
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        st.image(buf, caption=None, width=800)
                        plt.close(fig)  # optional cleanup
                else:
                    # Overlay individual spectra
                    fig, ax = plt.subplots(figsize=(8, 5))
                    for sample_id in selected_ids:
                        spectra = preprocessed_spectra[sample_id]
                        if isinstance(spectra, Spectrum):
                            spectra = [spectrum]
                        for spectrum in spectra:
                            ax.plot(spectrum.spectral_axis, spectrum.spectral_data, label=sample_id)
            
                    ax.set_title("Overlay of Selected Preprocessed Spectra")
                    ax.set_xlabel("Raman Shift (cm⁻¹)")
                    ax.set_ylabel("Intensity")
                    ax.legend(loc="best", fontsize="small")
            
                    buf = io.BytesIO()
                    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                    buf.seek(0)
                    st.image(buf, caption=None, width=800)
                    plt.close(fig)

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

        manual_param = None
        param_range = None
        n_folds = None
        param_grid = None

        if model_name == "PLS":
            enable_manual_param = st.checkbox("Manually select number of PLS components?", value=False)
            if enable_manual_param:
                manual_param = st.number_input("Manual n_components:", min_value=1, max_value=20, value=5)
            param_range = list(range(1, 11))

        elif model_name == "MLP":
            enable_grid_customization = st.checkbox("Customize MLP Parameter Grid?", value=False)
            if enable_grid_customization:
                hidden_layer_sizes = st.multiselect(
                    "Hidden layer sizes:",
                    options=[(50,), (100,), (50, 50), (100, 50), (50, 100)],
                    default=[(50,), (100,)]
                )
                alpha_values = st.multiselect("Alpha values:", options=[0.1, 0.01, 0.001, 0.0001], default=[0.01, 0.001])
                learning_rates = st.multiselect("Learning rates:", options=[0.0001, 0.001, 0.005, 0.01], default=[0.001])

                param_grid = {
                    'hidden_layer_sizes': hidden_layer_sizes,
                    'activation': ['relu'],
                    'alpha': alpha_values,
                    'learning_rate_init': learning_rates,
                    'early_stopping': [True],
                    'solver': ['adam']
                }

        # === Cross-validation options ===
        st.subheader("Choose Cross-Validation")
        use_group_kfold = st.checkbox("Use Grouped K-Fold CV?", value=False)
        
        group_strategy = None
        if use_group_kfold:
            group_strategy = st.radio("Group K-Fold by:", options=["Replicate", "Class"], index=0)
                
                
        n_folds = st.number_input("Number of K-Folds:", min_value=2, max_value=20, value=8)
        
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
            raw_Y = st.session_state["y_block"]
            axis = st.session_state["cropped_axis"]
            sample_groups = st.session_state.get("sample_groups")

            first_val = list(raw_X.values())[0]
            is_group_avg = not isinstance(first_val, list)

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

            results_dir = "./Model_Results"
            os.makedirs(results_dir, exist_ok=True)
            
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
                class_labels=color_labels
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

            # === Loadings / Importance Section ===
            st.subheader("📉 Loadings & Variables")

            for analyte in model_results.keys():
                result = model_results[analyte]

                if model_name == "PLS":
                    vip = result.get("vip_plot_path")
                    coef = result.get("coef_plot_path")
                    t2q = result.get("t2_plot_path")
                    final_pred = result.get("final_pred_plot_path")

                    st.markdown(f"**🔬 {analyte} (PLS)**")
                    plot_paths = [t2q, final_pred, coef, vip]
                    plot_labels = ["T² vs Q Residuals", "Final Predicted vs Actual", "Regression Coefficients", "VIP Scores"]

                    rows = [st.columns(2), st.columns(2)]
                    for i, (path, label) in enumerate(zip(plot_paths, plot_labels)):
                        if path and os.path.exists(path):
                            col = rows[i // 2][i % 2]
                            with col:
                                st.image(path, caption=label, width=500)

                elif model_name == "MLP":
                    st.markdown(f"**🔬 {analyte} (MLP)**")
                    final_pred = result.get("final_pred_plot_path")
                    feat_imp = result.get("feature_importance_path")

                    plot_paths = [final_pred, feat_imp]
                    plot_labels = ["Final Predicted vs Actual", "Feature Importance"]

                    cols = st.columns(2)
                    for i, (path, label) in enumerate(zip(plot_paths, plot_labels)):
                        if path and os.path.exists(path):
                            with cols[i]:
                                st.image(path, caption=label, use_container_width=True)
 
            # === PLS LV Score Plots ===
            st.subheader("🎯 Scores")
                  
            for analyte in model_results.keys():
                result = model_results[analyte]
                scoreplot_path = result.get("scoreplot_path")
            
                # Display only if the plot exists
                if scoreplot_path and os.path.exists(scoreplot_path):
                    st.image(scoreplot_path, caption=f"LV1 vs LV2 Scores – {analyte}", width=500)

# === TAB 4: Prediction ===
from models.prediction_eval import evaluate_on_prediction_set
from preprocessors.aligner import align_xy

if tab == "Prediction":
    st.header("Step 4: Predict on External Dataset")

    # === Upload Inputs ===
    with st.expander("📁 Upload Prediction Spectra (.zip of .spc files)"):
        pred_zip_file = st.file_uploader("Upload a .zip file of Raman .spc files", type="zip", key="pred_zip")

    with st.expander("📄 (Optional) Upload Prediction Y-block (Excel with 'ID' column)"):
        pred_y_file = st.file_uploader("Upload reference Y file (optional)", type="xlsx", key="pred_y")

    if st.button("Run Prediction"):
        if pred_zip_file is None:
            st.warning("Please upload a zip file of Raman spectra first.")
        else:
            # === Extract and Load Spectra ===
            pred_dir = "./temp_prediction"
            spectra_dir = os.path.join(pred_dir, "spectra")
            os.makedirs(spectra_dir, exist_ok=True)

            with zipfile.ZipFile(pred_zip_file, "r") as zip_ref:
                zip_ref.extractall(spectra_dir)

            pred_sample_spectra, _, _ = load_raman(pred_dir, spectra_dir)
            pred_preprocessed, _ = preprocess_pipeline_2(pred_sample_spectra, spectra_dir)

            # === Flatten X_pred ===
            X_pred = np.vstack([
                spectrum.spectral_data
                for spectra in pred_preprocessed.values()
                for spectrum in spectra
            ])
            pred_sample_ids = list(pred_preprocessed.keys())

            # === Optional Y-block ===
            filtered_pred_sample_ids = pred_sample_ids
            Y_pred_true = None

            if pred_y_file is not None:
                y_pred_df = pd.read_excel(pred_y_file)
                y_pred_df.set_index("ID", inplace=True)
                try:
                    filtered_X_pred, filtered_Y_pred, filtered_pred_sample_ids, _, _, _ = align_xy(
                        pred_preprocessed, y_pred_df
                    )
                    Y_pred_true = filtered_Y_pred.values
                except Exception as e:
                    st.warning(f"Y-block alignment failed: {e}")

            # === Run Predictions ===
            model_results = st.session_state.get("model_results", {})
            axis = st.session_state.get("cropped_axis", None)
            results_dir = "./Model_Results"

            prediction_outputs = []

            for analyte, result in model_results.items():
                model_obj = result["model"]
                y_mean = result["cv_results"]["y_mean"]
                model_name = "PLS" if "PLS" in result["final_pred_plot_path"] else "MLP"

                output = evaluate_on_prediction_set(
                    model=model_obj,
                    X_pred=filtered_X_pred,
                    y_mean=y_mean,
                    axis=axis,
                    analyte=analyte,
                    directory=results_dir,
                    Y_pred_true=Y_pred_true[:, 0] if Y_pred_true is not None else None,
                    model_name=model_name,
                    sample_ids=filtered_pred_sample_ids
                )
                prediction_outputs.append(output)

            # === Display Results ===
            st.subheader("📈 Prediction Results")

            for output in prediction_outputs:
                st.markdown(f"### {output['analyte']} ({output['model_name']})")

                # Show metrics if Y available
                if "r2_pred" in output:
                    st.markdown(f"**R²_pred**: {output['r2_pred']:.4f}  \n**RMSEP**: {output['rmsep']:.4f}")

                # Show DataFrame of predicted values
                if "csv_path" in output and os.path.exists(output["csv_path"]):
                    df = pd.read_csv(output["csv_path"])
                    st.dataframe(df)

                # Show plot
                if "pred_plot_path" in output and os.path.exists(output["pred_plot_path"]):
                    st.image(output["pred_plot_path"], caption="Predicted vs Actual", use_container_width=True)


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
        raw_Y = st.session_state["y_block"]
        axis = st.session_state["cropped_axis"]

        # Determine replicate structure
        first_val = list(raw_X.values())[0]
        is_group_avg = not isinstance(first_val, list)

        if is_group_avg:
            filtered_X, _, _, classes, _ = align_group_xy(raw_X, raw_Y)
        else:
            filtered_X, _, _, _, classes, _ = align_xy(raw_X, raw_Y)

        # Create results directory
        results_dir = "./Model_Results"
        os.makedirs(results_dir, exist_ok=True)

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
        st.image(os.path.join(results_dir, "PCA_Loadings_Annotated.png"), width=800)
