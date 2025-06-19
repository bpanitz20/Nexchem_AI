#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 14:49:27 2025

@author: bp
"""

import numpy as np
from models.wrappers import (
    PLS_model,
    MLPRegressor_model,
    MLPClassifier_model
    
)

from preprocessors.labeling import bin_targets

"""
def run_regression_loop(X, Y_df, results_dir, axis, groups=None, manual_param=None, sample_ids=None):
    print("\n🚀 Starting regression model training...")
    model_results = {}

    for analyte in Y_df.columns:
        print(f"\n🔬 Regression for: {analyte}")
        y = Y_df[analyte].values.reshape(-1, 1)
        model_results[analyte] = {}

        # === Add models here ===

        # PLS
        pls_result = PLS_model(
            X, y, results_dir, axis,
            analyte=analyte,
            groups=groups,
            manual_param=manual_param,
            sample_ids=sample_ids
        )
        model_results[analyte]["PLS"] = pls_result

        # MLP
        mlp_result = MLPRegressor_model(
            X, y, results_dir, axis,
            analyte=analyte,
            groups=groups
        )
        model_results[analyte]["MLP"] = mlp_result

        # You can add more models here later (e.g., SVR, RF, Ridge)

    return model_results
"""

def run_regression_loop(
    X,
    Y_df,
    results_dir,
    axis,
    model_name="PLS",
    param_name=None,
    param_grid=None,
    param_range=None,
    manual_param=None,
    n_folds=8,
    groups=None,
    sample_ids=None
):
    """
    Runs a regression loop across all analytes in Y_df using the specified model and CV configuration.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix.
    Y_df : pd.DataFrame
        Target variable(s) dataframe.
    results_dir : str
        Output directory for results and plots.
    axis : list or np.ndarray
        Raman shift axis for feature importance plotting.
    model_name : str
        Model to use ("PLS", "MLP", etc.).
    param_name : str
        Name of hyperparameter to tune (PLS only).
    param_range : list
        Range of hyperparameter values to test (PLS only).
    manual_param : int or tuple
        Manually specified parameter value (skips grid search).
    use_group_kfold : bool
        Whether to use group-aware cross-validation (MLP only).
    n_folds : int
        Number of CV folds.
    groups : list or array
        Group labels if using group-aware CV.
    sample_ids : list
        List of sample names (used for logging or plotting).

    Returns:
    --------
    dict
        Model results dictionary.
    """

    results_all = {}

    for analyte in Y_df.columns:
        print(f"\n🔬 Regression for: {analyte}")
        y = Y_df[analyte].values.reshape(-1, 1)

        if model_name == "PLS":
            model_results = PLS_model(
                X, y, results_dir, axis,
                analyte=analyte,
                manual_param=manual_param,
                groups=groups, n_folds=n_folds,
                sample_ids=sample_ids
            )

        elif model_name == "MLP":
            if param_grid is None:
                param_grid = {
                    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
                    'activation': ['relu'],
                    'alpha': [0.02, 0.01, 0.0009],
                    'learning_rate_init': np.linspace(0.0001, 0.01, 10).tolist(),
                    'early_stopping': [True],
                    'solver': ['adam']
                    }
            model_results = MLPRegressor_model(
                x=X,
                y=y,
                directory=results_dir,
                axis=axis,
                analyte=analyte,
                param_grid=param_grid,
                groups=groups, n_folds=n_folds
            )

        else:
            raise ValueError(f"Unsupported model: {model_name}")

        results_all[analyte] = model_results

    return results_all



def run_classification_loop(X, Y_df, results_dir, axis, bins=[0, 0.4, 0.7, 1], groups=None):
    """
    Loop through each Y column (analyte) and run classification models.

    Parameters:
    -----------
    X : np.ndarray
        Feature matrix
    Y_df : pd.DataFrame
        Target table with one column per analyte
    results_dir : str
        Directory to save model outputs
    axis : list
        Spectral axis for plotting
    bins : list of float
        Bin edges for converting continuous Y to classes
    """
    print("\n🚀 Starting classification model training...")
    for analyte in Y_df.columns:
        print(f"\n🔬 Classification for: {analyte}")
        y_continuous = Y_df[analyte].values
        
        y_class = bin_targets(y_continuous, bins=bins)
        print("Binned class distribution:", np.unique(y_class, return_counts=True))

        MLPClassifier_model(X, y_class, results_dir, axis, analyte=analyte, groups=groups)
        # Add other classifiers here (SVMClassifier_model, RFClassifier_model, etc.)
