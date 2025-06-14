#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:39:25 2025

@author: bp
"""

import streamlit as st
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
    group_preprocess_2
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

            # === Automatically Display Overlay ===
            overlay_path = os.path.join(spectra_dir, "Overlay_Raw.png")
            if os.path.exists(overlay_path):
                st.subheader("📊 Overlay of All Raw Spectra")
                st.image(overlay_path, use_column_width=True)
                
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
                fig, ax = plt.subplots(figsize=(8, 5))
                for sample_id in selected_ids:
                    for spectrum in sample_spectra[sample_id]:
                        ax.plot(spectrum.spectral_axis, spectrum.spectral_data, label=sample_id)

                ax.set_title("Overlay of Selected Raw Spectra")
                ax.set_xlabel("Raman Shift (cm⁻¹)")
                ax.set_ylabel("Intensity")
                ax.legend(loc="best", fontsize="small")
                st.pyplot(fig)
           
            # Store raw data in session for next tab
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
                preprocessed_spectra, cropped_axis = group_preprocess_2(
                    sample_spectra, sample_groups, spectra_dir,
                    crop_region=crop_region, derivative_order=deriv_order
                )

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

                preprocessed_spectra, cropped_axis, group_replicates = group_preprocess(
                    sample_spectra, sample_groups, spectra_dir,
                    crop_region=crop_region,
                    emsc_p_order=emsc_p_order,
                    deriv_order=deriv_order
                )
                st.session_state["group_replicates"] = group_replicates

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
                    spectra = [spectra]  # wrap single spectrum
                for spectrum in spectra:
                    ax.plot(spectrum.spectral_axis, spectrum.spectral_data, alpha=0.7)
            ax.relim()
            ax.autoscale_view()
            ax.set_title("Overlay of All Preprocessed Spectra")
            ax.set_xlabel("Raman Shift (cm⁻¹)")
            ax.set_ylabel("Intensity")
            st.pyplot(fig)

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
                fig2, ax2 = plt.subplots(figsize=(8, 5))
                for sample_id in selected_ids:
                    spectra = preprocessed_spectra[sample_id]
                    if isinstance(spectra, Spectrum):
                        spectra = [spectra]
                    for spectrum in spectra:
                        ax2.plot(spectrum.spectral_axis, spectrum.spectral_data, label=sample_id)
                ax2.relim()
                ax2.autoscale_view()
                ax2.set_title("Overlay of Selected Preprocessed Spectra")
                ax2.set_xlabel("Raman Shift (cm⁻¹)")
                ax2.set_ylabel("Intensity")
                ax2.legend(loc="best", fontsize="small")
                st.pyplot(fig2)
