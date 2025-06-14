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
