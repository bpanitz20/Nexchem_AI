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
