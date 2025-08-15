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
import pandas as pd
from scipy.stats import f
from sklearn.preprocessing import LabelEncoder

def print_CV_table(param_name, param_range, r2_cv, r2_cal, mse_cv, rmse_cal,
                   model_name, analyte, directory):
    """
    Print and save cross-validation + calibration performance summary table.

    Parameters:
    - param_name (str): Name of hyperparameter (e.g., 'n_components', 'alpha')
    - param_range (list): Parameter values tested
    - r2_cv (list): R² values from cross-validation
    - r2_cal (list): R² values for calibration
    - mse_cv (list): MSE values from cross-validation
    - rmse_cal (list): RMSE values from calibration
    - model_name (str): Model name (e.g., 'PLS')
    - analyte (str): Target name
    - directory (str): Where to save the CSV
    """
   
    """
    print(f"\n📊 Cross-Validation Summary for {analyte} ({model_name}):")
    print(f"{param_name:<12} | R2_CV    | R2_Cal   | RMSE_CV  | RMSE_Cal")
    print("-" * 60)
    for val, r2cv, r2cal, mse, rmsec in zip(param_range, r2_cv, r2_cal, mse_cv, rmse_cal):
        print(f"{val:<12} | {r2cv:.4f}  | {r2cal:.4f}  | {mse**0.5:.4f}  | {rmsec:.4f}")
    """
    
    # Save to CSV
    df = pd.DataFrame({
        param_name: param_range,
        "R2_CV": r2_cv,
        "R2_Cal": r2_cal,
        "RMSE_CV": [m**0.5 for m in mse_cv],
        "RMSE_Cal": rmse_cal
    })
    filename = f"CV_Summary_{model_name}_{analyte}.csv"
    output_path = os.path.join(directory, filename)
    df.to_csv(output_path, index=False)    

def format_model_summary(model_name, analyte, final_r2, final_r2_CV, final_mse, final_mse_CV,
                         best_params=None, optimal_param=None, param_name=None):
    """
    Returns a markdown-formatted summary string of model performance.
    """
    if model_name == "MLP":
        summary = f"""**🔬 {analyte} ({model_name})**  
• **Best parameters:** `{best_params}`  
• R²_Cal: `{final_r2:.4f}`  
• R²_CV: `{final_r2_CV:.4f}`  
• RMSE: `{final_mse**0.5:.4f}`  
• RMSECV: `{final_mse_CV**0.5:.4f}`  
"""
    else:
        summary = f"""**🔬 {analyte} ({model_name})**  
• {param_name}: `{optimal_param}`  
• R²_Cal: `{final_r2:.4f}`  
• R²_CV: `{final_r2_CV:.4f}`  
• RMSE: `{final_mse**0.5:.4f}`  
• RMSECV: `{final_mse_CV**0.5:.4f}`  
"""
    return summary


def print_model_summary(model_name, analyte, final_r2, final_r2_CV, final_mse, final_mse_CV,
                         best_params=None, optimal_param=None, param_name=None):
    """
    Print model performance summary to the console.
    """
    print(f"\nFinal Model Metrics for {analyte} ({model_name}):")
    if optimal_param is not None and param_name:
        print(f"{param_name}: {optimal_param}")
    elif best_params is not None:
        print("Best parameters:", best_params)
    print(f"R²_Cal: {final_r2:.4f}")
    print(f"R²_CV: {final_r2_CV:.4f}")
    print(f"RMSE: {final_mse**0.5:.4f}")
    print(f"RMSECV: {final_mse_CV**0.5:.4f}")
    #print(f"MSE: {final_mse:.4f}")
    #print(f"MSECV: {final_mse_CV:.4f}")
    return format_model_summary(model_name, analyte, final_r2, final_r2_CV, final_mse, final_mse_CV,
                                best_params, optimal_param, param_name)



def plot_pred_vs_actual(y_true, y_pred, directory, title, filename, class_labels=None):
    plt.figure(figsize=(8, 6))

    if class_labels is not None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(class_labels)
        unique_labels = np.unique(encoded_labels)
        class_names = le.classes_

        # Marker styles cycle
        markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']
        colors = plt.cm.tab10.colors

        for idx, class_val in enumerate(unique_labels):
            indices = encoded_labels == class_val
            plt.scatter(
                y_true[indices], y_pred[indices],
                c=[colors[idx % len(colors)]],
                marker=markers[idx % len(markers)],
                alpha=0.7,
                label=str(class_names[class_val])
            )

        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Data')

    # Ideal and best-fit lines
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             color='red', linestyle='--', label='Ideal')

    slope, intercept = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)
    plt.plot(y_true, slope * y_true + intercept, 'g--', label='Best Fit')

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(directory, filename), dpi=300, bbox_inches="tight")
    plt.close()


def plot_cv_performance(param_range, r2_cv, r2_cal, mse_cv, rmse_cal, param_name, analyte, model_name, directory):
    """
    Plot R2 and RMSE curves for calibration and cross-validation.
    """
    # Plot R2
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, r2_cv, label='R² CV', marker='o', color='tab:blue')
    if r2_cal:
        plt.plot(param_range, r2_cal, label='R² Cal', marker='s', color='tab:cyan')
    plt.title(f"R² vs {param_name} for {analyte} ({model_name})")
    plt.xlabel(param_name)
    plt.ylabel("R²")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'CV_R2_{model_name}_{analyte}.png'), dpi=300)
    plt.close()

    # Plot RMSE
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, [m**0.5 for m in mse_cv], label='RMSE CV', marker='o', color='tab:red')
    if rmse_cal:
        plt.plot(param_range, rmse_cal, label='RMSE Cal', marker='s', color='tab:pink')
    plt.title(f"RMSE vs {param_name} for {analyte} ({model_name})")
    plt.xlabel(param_name)
    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f'CV_RMSE_{model_name}_{analyte}.png'), dpi=300)
    plt.close()

def plot_coefficients(axis, coefficients, directory, model_name, analyte):
    plt.figure(figsize=(10, 6))
    plt.plot(axis, coefficients.flatten(), color='blue')
    plt.xlabel('Features')
    plt.ylabel('Regression Coefficients')
    plt.title(f'Regression Coefficients ({model_name})')
    plt.grid(True)
    plt.savefig(os.path.join(directory, f"PLS_Coefficients_{analyte}.png"), dpi=300, bbox_inches="tight")
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
    
    
def plot_t2_q_residuals(pls_model, X, y, analyte, directory, model_name="PLS", sample_ids=None):
    """
    Plots Hotelling's T² vs Q-residuals for PLS models, with optional outlier labeling.

    Parameters:
    -----------
    pls_model : sklearn.cross_decomposition.PLSRegression
        Trained PLS model
    X : np.ndarray
        Training feature matrix
    y : np.ndarray
        Target vector
    analyte : str
        Target name for labeling
    directory : str
        Directory to save the plot
    model_name : str
        Name of the model (default 'PLS')
    sample_ids : list of str or None
        Sample IDs to annotate (must match order of X rows)
    """
    # Compute scores and loadings
    T = pls_model.x_scores_
    P = pls_model.x_loadings_
    n, A = T.shape  # n = samples, A = components

    # Hotelling's T²
    T2 = np.sum((T / np.std(T, axis=0))**2, axis=1)
    T2_threshold = (A * (n - 1) / (n - A)) * f.ppf(0.95, A, n - A)

    # Reconstructed X and Q residuals
    X_reconstructed = T @ P.T
    Q_residuals = np.sum((X - X_reconstructed)**2, axis=1)
    Q_threshold = np.percentile(Q_residuals, 95)

    # Outlier detection
    outlier_mask = (T2 > T2_threshold) | (Q_residuals > Q_threshold)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(T2, Q_residuals, alpha=0.7, edgecolors='k', label='Samples')
    plt.axvline(T2_threshold, color='red', linestyle='--', label="T² 95% Limit")
    plt.axhline(Q_threshold, color='blue', linestyle='--', label="Q Residual 95% Limit")

    # Annotate outliers
    if sample_ids is not None:
        for i, is_outlier in enumerate(outlier_mask):
            if is_outlier:
                plt.annotate(sample_ids[i], (T2[i], Q_residuals[i]),
                             textcoords="offset points", xytext=(5, 5), ha='left', fontsize=8)

    plt.xlabel("Hotelling's T²")
    plt.ylabel("Q Residual")
    plt.title(f"T² vs Q-Residuals for {analyte} ({model_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(directory, f"T2_vs_Q_Residuals_PLS_{analyte}.png"), dpi=300)
    plt.close()


def plot_pls_scores(model, x, directory, analyte, class_labels=None, ellipse_alpha=0.2):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse
    from sklearn.preprocessing import LabelEncoder
    import numpy as np
    import os

    T = model.transform(x)  # PLS scores
    lv1, lv2 = T[:, 0], T[:, 1]
    
    x_var = np.var(model.x_scores_, axis=0)
    explained_variance = x_var / np.sum(x_var)

    plt.figure(figsize=(8, 6))

    if class_labels is not None:
        le = LabelEncoder()
        encoded_labels = le.fit_transform(class_labels)
        classes = le.classes_
        cmap = plt.get_cmap("tab10")

        for i, cls in enumerate(np.unique(encoded_labels)):
            idx = encoded_labels == cls
            x_pts, y_pts = lv1[idx], lv2[idx]
            color = cmap(i % 10)

            # Scatter
            plt.scatter(x_pts, y_pts, label=classes[cls], color=color, alpha=0.7)

            # Ellipse
            if len(x_pts) > 2:
                cov = np.cov(x_pts, y_pts)
                vals, vecs = np.linalg.eigh(cov)
                order = vals.argsort()[::-1]
                vals, vecs = vals[order], vecs[:, order]
                theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
                width, height = 2 * 1.96 * np.sqrt(vals)
                ellipse = Ellipse(
                    xy=(np.mean(x_pts), np.mean(y_pts)),
                    width=width,
                    height=height,
                    angle=theta,
                    edgecolor=color,
                    facecolor=color,
                    alpha=ellipse_alpha,
                    linewidth=1.2
                )
                plt.gca().add_patch(ellipse)

        plt.legend(title="Class", bbox_to_anchor=(1.05, 1), loc='upper left')

    else:
        plt.scatter(lv1, lv2, color="blue", alpha=0.7)

    plt.xlabel(f"Latent Variable 1 ({explained_variance[0]*100:.1f}%)")
    plt.ylabel(f"Latent Variable 2 ({explained_variance[1]*100:.1f}%)")
    plt.title(f"PLS Score Plot (LV1 vs LV2) - {analyte}")
    plt.grid(True)

    out_path = os.path.join(directory, f"PLS_ScorePlot_LV1vsLV2_{analyte}.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path
