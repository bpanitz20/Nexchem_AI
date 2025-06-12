#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 11 20:39:25 2025

@author: bp
"""

# streamlit_app.py

import streamlit as st
import pandas as pd
import tempfile
import os
from models.wrappers import PLS_model, MLPRegressor_model  # Add more later
from models.prediction_eval import evaluate_on_prediction_set
from preprocessors.raman_preprocess import preprocess_pipeline_2
from loaders.raman_loader import load_raman
from preprocessors.aligner import align_xy
import numpy as np

st.set_page_config(page_title="NexChem Model Builder", layout="wide")

st.title("🔬 NexChem_AI — Chemometric Modeling Platform")

# === Sidebar: File Uploads ===
st.sidebar.header("Upload Data")

raman_files_dir = st.sidebar.file_uploader("Upload Raman .spc files (zipped)", type=["zip"])
target_file = st.sidebar.file_uploader("Upload GCMS Target File (.xlsx)", type=["xlsx"])

# === Sidebar: Model Selection ===
st.sidebar.header("Model Selection")
selected_models = st.sidebar.multiselect("Select models to train", ["PLS", "MLP"], default=["PLS"])

# === Main Button ===
if st.sidebar.button("Run Models"):

    if raman_files_dir is None or target_file is None:
        st.error("Please upload both Raman spectra and GC-MS target files.")
    else:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "raman.zip")
            with open(zip_path, "wb") as f:
                f.write(raman_files_dir.read())
            os.system(f"unzip -q {zip_path} -d {tmpdir}/raman_data")

            spectra_dir = os.path.join(tmpdir, "output_spectra")
            os.makedirs(spectra_dir, exist_ok=True)

            sample_spectra, shifts, sample_groups = load_raman(f"{tmpdir}/raman_data", spectra_dir)
            preprocessed_spectra, cropped_axis = preprocess_pipeline_2(sample_spectra, spectra_dir)

            y_df = pd.read_excel(target_file)
            y_df.set_index("ID", inplace=True)

            X, Y, sample_ids, _ = align_xy(preprocessed_spectra, y_df)

            st.success("✅ Data loaded and aligned")

            # === Run models ===
            results = {}

            for model_name in selected_models:
                st.subheader(f"🧠 {model_name} Model Results")
                if model_name == "PLS":
                    result = PLS_model(X, Y.values, tmpdir, cropped_axis, analyte="ALL")
                elif model_name == "MLP":
                    result = MLPRegressor_model(X, Y.values, tmpdir, cropped_axis, analyte="ALL")
                else:
                    st.warning(f"Model {model_name} not implemented yet.")
                    continue

                results[model_name] = result

                # Show metrics
                st.markdown(f"**R²_Cal**: {result['final_r2']:.3f}")
                st.markdown(f"**R²_CV**: {result['final_r2_CV']:.3f}")
                st.markdown(f"**RMSE**: {result['final_mse']**0.5:.3f}")
                st.markdown(f"**RMSECV**: {result['final_mse_CV']**0.5:.3f}")

                # Show plots (just example: predicted vs actual)
                img_path = os.path.join(tmpdir, f"Pred_vs_Actual_{model_name}_ALL.png")
                if os.path.exists(img_path):
                    st.image(img_path, caption="Predicted vs Actual")

            st.success("✅ Modeling complete")
