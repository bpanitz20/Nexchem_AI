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

**Modeling**
- Regression or classification modeling modes
- Multiple model types available per mode
- Manual or automatic parameter selection
- Variable selection
- Adjustable cross validation strategies to fit your dataset
- One model built per analyte column in your Y-block

**Diagnostic Plots**
- Mode-specific diagnostic plots generated automatically for each model
- Regression: performance curves, variable importance, residual diagnostics, and more
- Classification: confusion matrices, ROC curves, score plots, and more
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
- Run on Raman spectra or Y-block 

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
- An optional `Class` column for coloring plots and cross validation

Example:

| ID  | Class | DHA  | EPA  | PUFA |
|-----|-------|------|------|------|
| 188 | 2022  | 10.5 | 5.1  | 15.6 |
| 190 | 2023  | 1.5  | 1.0  | 2.5  |

### Running a model

1. **Data Loading tab** — Set an output directory, then upload your `.zip` spectra file and your Y-block Excel file. Click **Load calibration data**.
2. **Preprocessing tab** — Choose a preprocessing method, adjust parameters if needed, and click **Run Preprocessing**.
3. **Modeling tab** — Choose regression or classification, select a model type, set cross-validation options, and click **Train Model**. Diagnostic plots appear automatically.
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
│   ├── classification_wrappers.py  # PLS-DA classifier and classification entry points
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
