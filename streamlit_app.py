#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:39:25 2025

@author: bp
"""

import streamlit as st
import io
import pandas as pd
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
    avg_y_block
)
from collections import defaultdict


st.set_page_config(page_title="NexChem App", layout="wide")
st.title("🔬 NexChem Chemometric App")

# === Sidebar Tabs ===
tab = st.sidebar.radio("Navigation", ["Data Loading", "Preprocessing", "Modeling", "Prediction"], index=0)

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

            elif selected_method == "4. Average Replicates: Savgol-EMSC-MeanCenter":
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
        n_folds = st.number_input("Number of K-Folds:", min_value=2, max_value=20, value=8)

        # === Start modeling ===
        if st.button("Train Model"):
            raw_X = st.session_state["preprocessed_spectra"]
            raw_Y = st.session_state["y_block"]
            axis = st.session_state["cropped_axis"]
            sample_groups = st.session_state.get("sample_groups")

            # Determine whether replicates or group-averaged spectra were used
            first_val = list(raw_X.values())[0]
            is_group_avg = not isinstance(first_val, list)

            if is_group_avg:
                filtered_X, filtered_Y, filtered_sample_ids, filtered_groups, unmatched_ids = align_group_xy(raw_X, raw_Y)
            else:
                filtered_X, filtered_Y, filtered_sample_ids, filtered_groups, unmatched_ids = align_xy(raw_X, raw_Y)

            results_dir = "./Model_Results"
            os.makedirs(results_dir, exist_ok=True)

            model_results = run_regression_loop(
                filtered_X,
                filtered_Y,
                results_dir,
                axis,
                groups=filtered_groups if use_group_kfold else None,
                model_name=model_name,
                param_range=param_range,
                param_grid=param_grid,
                manual_param=manual_param,
                n_folds=n_folds,
                sample_ids=filtered_sample_ids
            )

            # ✅ Store results in session_state so they're retained on rerun
            st.session_state["model_results"] = model_results
            st.session_state["model_built"] = True
            st.success("✅ Model training complete!")
            
      
            # === Model Summary ===
            st.subheader("Model Summary")
            
            # Show unmatched sample IDs (if any)
            if unmatched_ids:
                st.warning(f"❗ Unmatched Sample IDs: {', '.join(sorted(unmatched_ids))}")
                   
                        
            # Show per-analyte summary
            for analyte in raw_Y.columns:
                result = model_results[analyte]
                summary = result.get("summary", None)
                if summary:
                    st.markdown(summary)
            
            
            # CV Results Section
            st.subheader("Cross Validation Results")
            
            for analyte in raw_Y.columns:
                result = model_results[analyte]
            
                # === Show CV table ===
                cv_table = result.get("cv_table_df")
                if cv_table is not None:
                    st.markdown(f"**{analyte}**")
                    st.dataframe(cv_table.style.format(precision=4))
            
                # === Collect available plots ===
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
            
                # === Display plots in flexible columns ===
                if plot_paths:
                    cols = st.columns(len(plot_paths))
                    for i, (col, path) in enumerate(zip(cols, plot_paths)):
                        with col:
                            st.image(path, width=400)
                             
            # === Loadings / Variable Importance Section ===
            st.subheader("📉 Loadings & Variables")
            
            for analyte in raw_Y.columns:
                result = model_results[analyte]
            
                if model_name == "PLS":
                    # Collect plot paths
                    vip = result.get("vip_plot_path")
                    coef = result.get("coef_plot_path")
                    t2q = result.get("t2_plot_path")
                    final_pred = result.get("final_pred_plot_path")
            
                    st.markdown(f"**🔬 {analyte} (PLS)**")
            
                    plot_paths = [t2q, final_pred, coef, vip]
                    plot_labels = ["T² vs Q Residuals", "Final Predicted vs Actual", "Regression Coefficients", "VIP Scores"]
            
                    # Layout in 2x2 grid
                    rows = [st.columns(2), st.columns(2)]
                    for i, (path, label) in enumerate(zip(plot_paths, plot_labels)):
                        if path and os.path.exists(path):
                            col = rows[i // 2][i % 2]
                            with col:
                                st.image(path, width=500)
                elif model_name == "MLP":
                    st.markdown(f"**🔬 {analyte} (MLP)**")
                
                    final_pred = result.get("final_pred_plot_path")
                    feat_imp = result.get("feature_importance_path")
                
                    plot_paths = [final_pred, feat_imp]
                    plot_labels = ["Final Predicted vs Actual", "Feature Importance"]
                
                    rows = [st.columns(2)]
                    for i, (path, label) in enumerate(zip(plot_paths, plot_labels)):
                        if path and os.path.exists(path):
                            col = rows[0][i % 2]
                            with col:
                                st.image(path, use_column_width=True)
                                                      