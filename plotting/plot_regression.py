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

def format_model_summary(model_name, analyte, final_r2, final_r2_CV, final_mse, final_rmse_CV,
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
• RMSECV: `{final_rmse_CV:.4f}`  
"""
    else:
        summary = f"""**🔬 {analyte} ({model_name})**  
• {param_name}: `{optimal_param}`  
• R²_Cal: `{final_r2:.4f}`  
• R²_CV: `{final_r2_CV:.4f}`  
• RMSE: `{final_mse**0.5:.4f}`  
• RMSECV: `{final_rmse_CV:.4f}`  
"""
    return summary


def print_model_summary(model_name, analyte, final_r2, final_r2_CV, final_mse, final_rmse_CV,
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
    print(f"RMSECV: {final_rmse_CV:.4f}")
    #print(f"MSE: {final_mse:.4f}")
    #print(f"MSECV: {final_mse_CV:.4f}")
    return format_model_summary(model_name, analyte, final_r2, final_r2_CV, final_mse, final_rmse_CV,
                                best_params, optimal_param, param_name)



def plot_pred_vs_actual_paper(y_true, y_pred, directory, title, filename, class_labels=None):
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

    _base = os.path.join(directory, os.path.splitext(filename)[0])
    plt.savefig(_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(_base + ".pdf", bbox_inches="tight")
    plt.close()





def plot_pred_vs_actual(
    y_true,
    y_pred,
    directory,
    title,
    filename,
    class_labels=None,
):

    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    fig, ax = plt.subplots(figsize=(8,6))

    # ---- style map ----
    style_map = {
        "2016":        {"marker": "D", "color": "red"},
        "2022-2023":   {"marker": "s", "color": "lightgreen"},
        "2023-2024":   {"marker": "^", "color": "blue"},
        "2018":        {"marker": "v", "color": "turquoise"},
        "2017":        {"marker": "*", "color": "pink"},
        "2022":        {"marker": "o", "color": "gold"},
        "2023":        {"marker": "D", "color": "darkgreen"},
        "2021-2022":   {"marker": "s", "color": "darkblue"},
    }

    # Desired legend order
    legend_order = [
        "Fit",
        "1:1",
        "2016",
        "2022-2023",
        "2023-2024",
        "2018",
        "2017",
        "2022",
        "2023",
        "2021-2022"
    ]

    lo = float(np.nanmin([y_true.min(), y_pred.min()]))
    hi = float(np.nanmax([y_true.max(), y_pred.max()]))

    # ---- 1:1 line ----
    line_ideal, = ax.plot(
        [lo, hi], [lo, hi],
        linestyle="--",
        linewidth=1.2,
        color="green",
        label="1:1",
        zorder=1
    )

    # ---- Fit line ----
    slope, intercept = np.polyfit(y_true, y_pred, 1)
    x_line = np.linspace(lo, hi, 200)

    line_fit, = ax.plot(
        x_line,
        slope * x_line + intercept,
        linestyle="-",
        linewidth=1.6,
        alpha=0.5,
        color="red",
        label=f"Fit (slope = {slope:.4f})",
        zorder=1
    )

    # ---- scatter points ----
    scatter_handles = {}

    if class_labels is not None:
        class_labels = np.asarray(class_labels).astype(str)
        unique_labels = np.unique(class_labels)

        for label in unique_labels:
            idx = class_labels == label

            marker = style_map.get(label, {}).get("marker", "o")
            color = style_map.get(label, {}).get("color", "black")

            size = 50
            if marker == "*":
                size = 80

            sc = ax.scatter(
                y_true[idx],
                y_pred[idx],
                marker=marker,
                s=size,
                facecolor=color,
                edgecolor="black",
                linewidth=0.5,
                alpha=0.85,
                label=label,
                zorder=3
            )

            scatter_handles[label] = sc

    else:
        ax.scatter(
            y_true,
            y_pred,
            s=55,
            facecolor="blue",
            edgecolor="black",
            zorder=3
        )

    # ---- build ordered legend ----
    legend_map = {
        "Fit": line_fit,
        "1:1": line_ideal,
        **scatter_handles
    }

    ordered_handles = []
    ordered_labels = []

    for key in legend_order:
        if key in legend_map:
            h = legend_map[key]
            ordered_handles.append(h)
    
            # Use the actual plotted label for the fit line (so slope shows)
            if key == "Fit":
                ordered_labels.append(h.get_label())  # "Fit (slope = ...)"
            else:
                ordered_labels.append(key)

    ax.legend(
    ordered_handles,
    ordered_labels,
    loc="upper left",
    frameon=True
)

    # ---- labels ----
    ax.set_xlabel("Actual Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title(title)

    # ---- grid ----
    ax.grid(True, linewidth=0.6, alpha=0.25)

    # ---- clean journal style ----
    ax.spines["top"].set_visible(True)
    ax.spines["right"].set_visible(True)

    # ---- ticks INSIDE plot ----
    ax.tick_params(
        axis='both',
        direction='in',
        length=6,
        width=1
    )

    fig.tight_layout()
    
    # ---- save ----
    os.makedirs(directory, exist_ok=True)
    
    fig.tight_layout()
    
    # Save original file (PNG or whatever filename specifies)
    fig.savefig(
        os.path.join(directory, filename),
        dpi=300,
        bbox_inches="tight"
    )
    
    # Save additional PDF copy
    base, _ = os.path.splitext(filename)
    pdf_name = base + ".pdf"
    
    fig.savefig(
        os.path.join(directory, pdf_name),
        bbox_inches="tight"
    )
    
    plt.close(fig)
    









def plot_cv_performance(param_range, r2_cv, r2_cal, rmse_cv, rmse_cal, param_name, analyte, model_name, directory):
    """
    Plot R2 and RMSE curves for calibration and cross-validation.
    """
    # Plot R2
    plt.figure(figsize=(8, 6))
    plt.plot(param_range, r2_cv, label='R² CV', marker='o', color='tab:blue')
    if r2_cal:
        plt.plot(param_range, r2_cal, label='R² Cal', marker='s', color='tab:orange')
    plt.title(f"R² vs {param_name} for {analyte} ({model_name})")
    plt.xlabel(param_name)
    plt.ylabel("R²")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _r2_base = os.path.join(directory, f'CV_R2_{model_name}_{analyte}')
    plt.savefig(_r2_base + '.png', dpi=300)
    plt.savefig(_r2_base + '.pdf')
    plt.close()

    # Plot RMSE
    fig, ax = plt.subplots(figsize=(8,6))
    plt.plot(param_range, rmse_cv, label='RMSE CV', marker='o', color='tab:blue')
    if rmse_cal:
        plt.plot(param_range, rmse_cal, label='RMSE Cal', marker='o', color='tab:orange')
    plt.title(f"RMSE vs {param_name} for {analyte} ({model_name})")
    plt.xlabel(param_name)

    ax.set_ylim(0,8)
    ax.set_yticks(range(0,9,1))
    ax.tick_params(axis="both", direction="in")
    ax.grid(True, alpha=0.2, linewidth=0.5)

    plt.ylabel("RMSE")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    _rmse_base = os.path.join(directory, f'CV_RMSE_{model_name}_{analyte}')
    plt.savefig(_rmse_base + '.png', dpi=300)
    plt.savefig(_rmse_base + '.pdf')
    plt.close()

def plot_coefficients(axis, coefficients, directory, model_name, analyte):
    plt.figure(figsize=(10, 6))
    plt.plot(axis, coefficients.flatten(), color='blue')
    plt.xlabel('Features')
    plt.ylabel('Regression Coefficients')
    plt.title(f'Regression Coefficients ({model_name})')
    plt.grid(True)
    _base = os.path.join(directory, f"PLS_Coefficients_{analyte}")
    plt.savefig(_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(_base + ".pdf", bbox_inches="tight")
    plt.close()

def plot_feature_importance(model, x, y, axis, directory, model_name, analyte, top_n=20):
    """
    Plots top-N permutation feature importances for models that don't expose coefficients.
    """

    result = permutation_importance(model, x, y, n_repeats=10, random_state=42, n_jobs=1, scoring='neg_mean_squared_error')
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
    _base = os.path.join(directory, f'Feature_Importance_{model_name}_{analyte}')
    plt.savefig(_base + '.png', dpi=300)
    plt.savefig(_base + '.pdf')
    plt.close()

def plot_vip_scores(pls_model, x, axis, directory,
                    model_name, analyte,
                    label_peaks=False, threshold=1.0, vip=None):
    """
    Compute and plot VIP scores for a fitted PLS model.

    Parameters
    ----------
    threshold : float
        Value at which to draw the horizontal significance line.
        Defaults to 1.0 (conventional cutoff).
    vip : np.ndarray or None
        Pre-computed VIP scores.  When provided, ``pls_model`` and ``x`` are
        ignored and the calculation step is skipped.  Pass pre-computed scores
        here when plotting full-spectrum VIP from a preliminary model (e.g.
        after block variable selection reduces X).
    """
    if vip is None:
        from models.vip import calculate_vip
        vip = calculate_vip(pls_model)

    # Plot VIP scores
    plt.figure(figsize=(10, 5))
    plt.plot(axis, vip, label='VIP Scores')
    plt.axhline(threshold, color='red', linestyle='--',
                label=f'Significance Threshold ({threshold:.1f})')
    
    
    if label_peaks:
        
        # ---- Label top N VIP peaks with minimum separation (cm^-1) ----
        top_n = 10
        min_sep = 10  # <-- change to 5 or 10 etc (in cm^-1)
        
        axis_arr = np.asarray(axis)
        vip_arr = np.asarray(vip)
        
        # Optional: only consider points within your plot window
        in_window = (axis_arr >= 800) & (axis_arr <= 1800)
        cand_idx = np.where(in_window)[0]
        
        # Sort candidate indices by VIP descending
        cand_sorted = cand_idx[np.argsort(vip_arr[cand_idx])[::-1]]
        
        selected = []
        for i in cand_sorted:
            if all(abs(axis_arr[i] - axis_arr[j]) >= min_sep for j in selected):
                selected.append(i)
            if len(selected) == top_n:
                break
        
        # Annotate selected peaks
        for i in selected:
            plt.scatter(axis_arr[i], vip_arr[i], color="black", s=20, zorder=5)
            plt.text(
                axis_arr[i],
                vip_arr[i] + 0.08,
                f"{int(round(axis_arr[i]))}",
                fontsize=9,
                ha="center",
                va="bottom",
                zorder=6,
            )
        
    
    plt.xlabel('Raman Shift (cm⁻¹)')
    #plt.xlim(800, 1801)
    #plt.ylim(0, 3)
    
    #plt.xticks(np.arange(800, 1801, 200))
    #plt.yticks(np.arange(0, 3.1, 0.5))
        
    plt.ylabel('VIP Scores')
    plt.title(f'VIP Scores for {analyte}')
    plt.legend()
    plt.grid(True, linestyle="-", linewidth=0.4, alpha=0.3)
    plt.tight_layout()
    
    
    os.makedirs(directory, exist_ok=True)
    out_base = os.path.join(directory, f"VIP_Scores_{model_name}_{analyte}")
    plt.savefig(out_base + ".pdf")          # vector
    plt.savefig(out_base + ".png", dpi=300) # raster backup
        
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
    _base = os.path.join(directory, f"T2_vs_Q_Residuals_PLS_{analyte}")
    plt.savefig(_base + ".png", dpi=300)
    plt.savefig(_base + ".pdf")
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

    _base = os.path.join(directory, f"PLS_ScorePlot_LV1vsLV2_{analyte}")
    out_path = _base + ".png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.savefig(_base + ".pdf", bbox_inches="tight")
    plt.close()
    return out_path
