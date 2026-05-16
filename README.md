# NexChem — Chemometric Modeling Platform

NexChem is an interactive, browser-based application for building and evaluating chemometric models from spectroscopy data. It is designed for analytical chemists and researchers who want to apply chemometric methods without writing code.

Built with [Streamlit](https://streamlit.io), NexChem guides you step-by-step from raw spectra to validated predictive models with publication-ready diagnostic plots.

---

## Features

**Data Loading**
- Upload Raman spectra (`.spc` files) as a `.zip` archive
   -Currently supports .spc files from Thermo Analyzers.
- Upload reference Y-block data (Excel `.xlsx`)
- Automatic sample ID matching between spectra and reference data

**Preprocessing**
- Savitzky-Golay derivative + SNV + global mean-centering
- AsLS baseline correction + Savitzky-Golay smoothing + SNV
- Replicate-averaging with Extended Multiplicative Scatter Correction (EMSC) + mean-centering
- No preprocessing (pass-through)
- Preprocessing state is saved and automatically applied to prediction sets — no data leakage

**Regression Modeling**
- Partial Least Squares (PLS) with automatic or manual component selection
- Multi-Layer Perceptron (MLP) regression with grid search cross-validation
- Standard K-Fold and Group K-Fold cross-validation
- One model built per analyte column in your Y-block

**Diagnostic Plots**
- CV predicted vs. actual (static and interactive)
- R² and RMSECV curves
- PLS: VIP scores, regression coefficients, T²/Q residuals, LV score plot
- MLP: permutation feature importance
- Analyte correlation heatmap (Y-block)
- All plots saved as PNG and PDF to your output directory

**Prediction**
- Apply a trained model to an external prediction set
- Optional: upload reference Y values to compute R²_pred and RMSEP
- Results saved as CSV

**PCA**
- PCA score plot with selectable principal components
- 95% confidence ellipses per class
- PCA loadings plot with annotated top bands
- PCA-DA: LDA applied to PCA scores with cross-validated accuracy curve
- Run on Raman spectra or Y-block (fatty acid profiles)

---

## Installation

### One-Click Launcher (Recommended)

If you downloaded NexChem from the GitHub release page, a one-click launcher is included.

1. Download the `NexChem` binary from the release page.
2. Double-click to launch — the app will open automatically in your browser.

> **Mac / Linux note:** If you see a *"cannot be opened because it is from an unidentified developer"* warning, go to Privacy & security settings and choose open anyway.


### 2. Manual Installation

```bash
conda env create -f environment.yml
conda activate nexchem_env
```

This installs all dependencies including `ramanspy`, `plotly`, and the `spc` file reader.

> **Note:** The `spc` package is installed directly from GitHub. An internet connection is required for the first install.

### 3. Launch the app

```bash
streamlit run streamlit_app.py
```

The app will open automatically in your browser at `http://localhost:8501`.

---

## Quick Start

### Preparing your files

**Raman spectra** should be Thermo `.spc` files named using this convention:

```
SampleID-Replicate_acquisition-params.spc
```

Examples:
```
188-1_450mw_10s.spc
188-2_450mw_10s.spc
188-3_450mw_10s.spc
190-1_450mw_10s.spc
```

The number before the dash (`188`, `190`) is the sample group ID. The number after the dash is the replicate number. Everything after the underscore is ignored.

Pack all `.spc` files for one dataset into a single `.zip` file before uploading.

**Reference data (Y-block)** must be an Excel file with:
- An `ID` column matching the sample group IDs in the spectra names
- One column per target analyte (e.g., `DHA`, `EPA`, `PUFA`)
- An optional `Class` column for coloring plots

Example:

| ID  | Class | DHA  | EPA  | PUFA |
|-----|-------|------|------|------|
| 188 | 2022  | 10.5 | 5.1  | 15.6 |
| 190 | 2023  | 1.5  | 1.0  | 2.5  |

### Running a model

1. **Data Loading tab** — Set an output directory, then upload your `.zip` spectra file and your Y-block Excel file. Click **Load calibration data**.
2. **Preprocessing tab** — Choose a preprocessing method, adjust parameters if needed, and click **Run Preprocessing**.
3. **Modeling tab** — Choose a model (PLS or MLP), set cross-validation options, and click **Train Model**. Diagnostic plots appear automatically.
4. **Prediction tab** — Upload a new `.zip` of spectra. Optionally upload a Y-block for that set. Click **Run Prediction**.
5. **PCA tab** — Visualize your preprocessed data in PCA space with class-colored groupings.

All results are saved to the output directory you set in step 1.

---

## Project Structure

```
Nexchem_AI/
├── streamlit_app.py          # Main application — all UI tabs
├── run_nexchem.py            # PyInstaller entry point for packaged launcher
├── config.py                 # Shared defaults (crop region, SavGol params, MLP grid)
├── environment.yml           # Conda environment specification
│
├── loaders/
│   └── raman_loader.py       # Reads .spc files, returns dict of Spectrum objects
│
├── preprocessors/
│   ├── raman_preprocess.py   # All preprocessing pipelines (compute / plot / shim)
│   ├── transforms.py         # Composable steps: GlobalMeanCenterStep, SNVStep
│   ├── aligner.py            # Aligns X and Y blocks by sample ID
│   └── labeling.py           # Bins continuous targets for classification
│
├── models/
│   ├── wrappers.py           # PLS_model, MLPRegressor_model, PCA_model, PCADA_model
│   ├── run_loops.py          # run_regression_loop, MODEL_REGISTRY
│   ├── cross_val.py          # KFold_CV, KFold_Gridsearch_CV, PCADA_CV
│   ├── prediction_eval.py    # evaluate_on_prediction_set
│   ├── vip.py                # VIP score calculation
│   ├── block_selection.py    # Spectral block/region selection utilities
│   └── selectors/            # Pluggable feature selectors (VIP, block)
│
├── plotting/
│   ├── plot_regression.py    # pred vs actual, VIP, coefficients, T²/Q, scores, correlation map
│   ├── plot_classifier.py    # confusion matrix, ROC, decision boundary
│   ├── plot_raw.py           # raw spectra overlays
│   └── plot_PCA.py           # PCA scores, loadings, PCA-DA CV curve
│
├── utils/
│   └── pdf_export.py         # PDF export helpers for figures and images
│
└── data/
    └── hubbs/
        └── Seriola/          # Example dataset (Seriola dorsalis Raman spectra)
```

---

## Adding a New Regression Model

NexChem uses a registry-based architecture. To add a new model type (e.g., SVR):

1. Add a wrapper function `SVR_model(...)` in `models/wrappers.py` following the same compute / plot / shim pattern as `PLS_model`. Populate `result['diagnostic_plots']` and `result['model_type']` in the returned dict.
2. Add an adapter in `models/run_loops.py`:
   ```python
   def _run_svr(X, y, results_dir, axis, analyte, groups, n_folds,
                sample_ids, class_labels, manual_param=None, param_grid=None):
       return SVR_model(...)
   ```
3. Register it:
   ```python
   MODEL_REGISTRY["SVR"] = _run_svr
   ```
4. Add `"SVR"` to the model selectbox in `streamlit_app.py`.

The diagnostic plot rendering in Tab 3 is generic — it reads `diagnostic_plots` from the result dict, so no other UI changes are needed.

---

## Contributing

Contributions are welcome. If you find a bug or want to add a feature:

1. Fork the repository and create a branch from `main`.
2. Make your changes with clear, focused commits.
3. Ensure the app runs without errors by testing in the Streamlit UI.
4. Open a pull request with a description of what was changed and why.

For major changes (new model types, new preprocessing pipelines, UI redesigns), please open an issue first to discuss the approach.

---

## Citation

If you use NexChem in your research, please cite the accompanying manuscript:

> Panitz, B. et al. (*in review*). NexChem: An Interactive Chemometric Modeling Platform for Raman Spectroscopy. *Food Chemistry*.

Once published, this will be updated with the full citation and DOI. A BibTeX entry will be added here at that time.

---

## Contact

Developed by **Ben Panitz**
FAU Bioanalytical Core & Aquaculture Research

For questions, bug reports, or collaboration inquiries, please open an [issue on GitHub](https://github.com/bpanitz20/Nexchem_AI/issues).
