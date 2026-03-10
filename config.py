# config.py
# Central configuration — single source of truth for shared defaults in NexChem_AI.

import numpy as np

# --- Spectral axis ---
DEFAULT_CROP_REGION = (800, 1800)

# --- Savitzky-Golay ---
DEFAULT_SAVGOL_WINDOW = 13
DEFAULT_SAVGOL_POLYORDER = 2
DEFAULT_DERIV_ORDER = 1

# --- AsLS baseline correction ---
DEFAULT_ASLS_LAMBDA = 1e5
DEFAULT_ASLS_P = 0.001

# --- Cross-validation ---
DEFAULT_N_FOLDS = 8

# --- MLP default hyperparameter grid ---
# Shared between models/wrappers.py (_mlp_compute) and models/run_loops.py.
DEFAULT_MLP_PARAM_GRID = {
    'pls__n_components': [4, 5, 6],
    'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
    'mlp__activation': ['relu'],
    'mlp__alpha': [0.02, 0.01, 0.0009],
    'mlp__learning_rate_init': list(np.linspace(0.0001, 0.01, 10)),
    'mlp__early_stopping': [True],
    'mlp__solver': ['adam'],
}
