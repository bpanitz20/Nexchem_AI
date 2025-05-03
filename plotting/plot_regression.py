#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Functions for Regression Model Evaluation

This module contains standardized plotting functions for visualizing
model performance, including predicted vs. actual plots, cross-validation
curves, and regression coefficient plots. Designed for modular use
within NexChem_AI.

Author: Ben Panitz
Created: April 26, 2025
"""


# plotting/plot_regression.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance


def plot_pred_vs_actual(y_true, y_pred, directory, title, filename):
    plt.figure(figsize=(8, 8))
    plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Data')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
             color='red', linestyle='--', label='Ideal')
    slope, intercept = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)
    plt.plot(y_true, slope*y_true + intercept, 'g--', label='Best Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(directory, filename), dpi=300, bbox_inches="tight")
    plt.show
    plt.close()

def plot_cv_performance(param_range, mean_r2_CV, mean_mse_CV, param_name, analyte, model_name, directory):
    # Plot R² vs Parameter
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, mean_r2_CV, marker='o', linestyle='-', color='b')
    plt.xlabel(param_name)
    plt.ylabel('R² Score CV')
    plt.title(f'R² Score CV vs. {param_name} for {analyte} ({model_name})')
    plt.grid(True)
    plt.savefig(os.path.join(directory, f'R²_vs_{param_name}_{model_name}_{analyte}.png'), dpi=300, bbox_inches="tight")
    plt.close()

    # Plot MSECV vs Parameter
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, mean_mse_CV, marker='o', linestyle='-', color='r')
    plt.xlabel(param_name)
    plt.ylabel('Mean Squared Error CV (MSECV)')
    plt.title(f'MSECV vs. {param_name} for {analyte} ({model_name})')
    plt.grid(True)
    plt.savefig(os.path.join(directory, f'MSE_vs_{param_name}_{model_name}_{analyte}.png'), dpi=300, bbox_inches="tight")
    plt.close()

def plot_coefficients(axis, coefficients, directory, model_name, analyte):
    plt.figure(figsize=(10, 6))
    plt.plot(axis, coefficients.flatten(), color='blue')
    plt.xlabel('Features')
    plt.ylabel('Regression Coefficients')
    plt.title(f'Regression Coefficients ({model_name})')
    plt.grid(True)
    plt.savefig(os.path.join(directory, f'Coefficients_{model_name}_{analyte}.png'), dpi=300, bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, x, y, axis, directory, model_name, analyte, top_n=20):
    """
    Plots top-N permutation feature importances for models that don't expose coefficients.
    """

    result = permutation_importance(model, x, y, n_repeats=10, random_state=42, n_jobs=-1, scoring='neg_mean_squared_error')
    importances = result.importances_mean
    sorted_idx = np.argsort(importances)[::-1]

    top_idx = sorted_idx[:top_n]
    top_wavenumbers = [f"{axis[i]:.1f}" for i in top_idx]
    top_importances = importances[top_idx]

    plt.figure(figsize=(10, 5))
    plt.bar(range(top_n), top_importances, align='center')
    plt.xticks(range(top_n), top_wavenumbers, rotation=45, ha='right')
    plt.title(f"Top {top_n} Feature Importances via Permutation ({model_name})")
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Importance")
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'Feature_Importance_{model_name}_{analyte}.png'), dpi=300)
    plt.close()

def plot_vip_scores(pls_model, x, axis, directory, model_name, analyte):
    """
    Compute and plot VIP scores for a fitted PLS model.
    """
    t = pls_model.x_scores_
    w = pls_model.x_weights_
    q = pls_model.y_loadings_

    p, h = w.shape
    s = np.square(t).sum(axis=0) * np.square(q).flatten()
    total_s = np.sum(s)

    vip = np.sqrt(p * (np.dot(np.square(w), s)) / total_s)

    # Plot VIP scores
    plt.figure(figsize=(10, 5))
    plt.plot(axis, vip, label='VIP Scores')
    plt.axhline(1.0, color='red', linestyle='--', label='VIP = 1 Threshold')
    plt.xlabel('Wavenumber (cm⁻¹)')
    plt.ylabel('VIP Score')
    plt.title(f'VIP Scores for {analyte} ({model_name})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'VIP_Scores_{model_name}_{analyte}.png'), dpi=300)
    plt.close()