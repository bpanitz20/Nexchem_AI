#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:20:48 2025

@author: bp
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from matplotlib.patches import Ellipse



def plot_confusion_matrix(y_true, y_pred, directory, model_name, analyte, labels=None, suffix=""):
    """
    Plots and saves the confusion matrix.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth class labels
    y_pred : array-like
        Predicted class labels
    directory : str
        Directory to save the plot
    model_name : str
        Name of the model (used in title and filename)
    analyte : str
        Name of the target analyte
    labels : list, optional
        Class label names
    """
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=labels if labels else np.unique(y_true),
        cmap='Blues', xticks_rotation=45
    )
    plt.title(f'Confusion Matrix: {model_name} ({analyte}{suffix})')
    plt.tight_layout()
    _base = os.path.join(directory, f'ConfusionMatrix_{model_name}_{analyte}{suffix}')
    plt.savefig(_base + '.png', dpi=300)
    plt.savefig(_base + '.pdf')
    plt.close()
    
    
def plot_roc_curve(y_true, y_proba, directory, model_name, analyte, suffix=""):
    """
    Plots multi-class ROC curves with optional suffix for CV vs. final model.
    
    Parameters:
    -----------
    y_true : array-like
        True class labels (integers)
    y_proba : array-like
        Predicted class probabilities (n_samples, n_classes)
    directory : str
        Output directory to save plot
    model_name : str
        Name of the model
    analyte : str
        Target analyte
    suffix : str
        Optional suffix for labeling ("", "_CV", etc.)
    """
    classes = np.unique(y_true)
    n_classes = len(classes)

    # Binarize y_true for one-vs-rest ROC computation
    y_bin = label_binarize(y_true, classes=classes)

    # Start ROC plot
    plt.figure(figsize=(7, 6))

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Class {classes[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {model_name} ({analyte}){suffix}')
    plt.legend(loc='lower right')
    plt.grid(True)

    _base = os.path.join(directory, f"ROC_{model_name}_{analyte}{suffix}")
    plt.savefig(_base + '.png', dpi=300, bbox_inches="tight")
    plt.savefig(_base + '.pdf', bbox_inches="tight")
    plt.close()
    

def plot_decision_boundary(x, y, model, directory, model_name, analyte):
    """
    Projects high-dimensional data to 2D using PCA for visualization only,
    then fits a separate visualization classifier in that 2D space to draw
    the decision boundary. The passed model is never modified.

    Note: the boundary shown is an approximation — it reflects how a copy of
    the same classifier type behaves in 2D PCA space, not in the original
    feature space. The scatter points represent the true training samples.
    """
    from sklearn.base import clone

    # Step 1: PCA to 2D for visualization only
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(x)

    # Step 2: Fit a fresh clone of the model in 2D PCA space (visualization only).
    # The original trained model is not touched.
    viz_model = clone(model)
    viz_model.fit(X_pca, y)

    # Step 3: Create mesh grid
    h = 0.02
    x_min, x_max = X_pca[:, 0].min() - 1, X_pca[:, 0].max() + 1
    y_min, y_max = X_pca[:, 1].min() - 1, X_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Step 4: Predict on mesh using the visualization clone
    Z = viz_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Step 5: Plot
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, edgecolor='k', cmap='coolwarm')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(f'Decision Boundary (2D PCA approx.): {model_name} ({analyte})')
    plt.legend(*scatter.legend_elements(), title="Class")
    plt.tight_layout()

    # Step 6: Save
    _base = os.path.join(directory, f'DecisionBoundary_{model_name}_{analyte}')
    plt.savefig(_base + '.png', dpi=300)
    plt.savefig(_base + '.pdf')
    plt.close()


# ── PLS-DA plot functions ────────────────────────────────────────────────────
# All functions return a matplotlib Figure for display in Streamlit.


def plot_plsda_cv_curve(cv_results, optimal_param, analyte="",
                        fig_width=7.0, fig_height=5.0,
                        label_fontsize=12, tick_fontsize=10):
    """
    CV accuracy vs n_components curve (analog of RMSECV/RMSEC plot in
    regression).  Shows both calibration and CV accuracy so the user can
    judge over-fitting and select the optimal number of latent variables.

    Returns
    -------
    matplotlib.figure.Figure
    """
    param_range  = list(range(1, len(cv_results['mean_acc_CV']) + 1))
    mean_acc_cv  = cv_results['mean_acc_CV']
    mean_acc_cal = cv_results['mean_acc_cal']

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.plot(param_range, mean_acc_cv,  'o-', color='steelblue',
            label='CV Accuracy',  linewidth=1.8, markersize=5)
    ax.plot(param_range, mean_acc_cal, 's--', color='tomato',
            label='Cal Accuracy', linewidth=1.5, markersize=5)
    ax.axvline(optimal_param, color='grey', linestyle=':', linewidth=1.2,
               label=f'Optimal LV = {optimal_param}')
    ax.set_xlabel('Number of Latent Variables', fontsize=label_fontsize)
    ax.set_ylabel('Accuracy', fontsize=label_fontsize)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(param_range)
    ax.tick_params(labelsize=tick_fontsize)
    title = 'PLS-DA CV Accuracy vs LVs'
    if analyte:
        title += f' — {analyte}'
    ax.set_title(title, fontsize=label_fontsize)
    ax.legend(fontsize=tick_fontsize)
    fig.tight_layout()
    return fig


def plot_plsda_confusion_matrix(y_true, y_pred, classes, suffix="",
                                analyte="", fig_width=5.0, fig_height=4.5,
                                label_fontsize=12, tick_fontsize=10,
                                cmap='Blues'):
    """
    Confusion matrix as a matplotlib Figure (not saved to disk).

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred,
        display_labels=classes,
        cmap=cmap,
        xticks_rotation=45,
        ax=ax,
    )
    title = f'Confusion Matrix{suffix}'
    if analyte:
        title += f' — {analyte}'
    ax.set_title(title, fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    fig.tight_layout()
    return fig


def plot_plsda_scores(x_scores, y_labels, classes, lv_x=1, lv_y=2,
                      show_ellipses=True, ellipse_alpha=0.18,
                      class_colors=None, analyte="",
                      fig_width=7.0, fig_height=6.0,
                      label_fontsize=12, tick_fontsize=10,
                      point_size=50, show_legend=True,
                      sample_labels=None):
    """
    LV scores plot (LVx vs LVy) colored by class with optional 95 %
    confidence ellipses.  Mirrors the PCA score plot style in the app.

    Parameters
    ----------
    x_scores      : np.ndarray  (n_samples, n_components)
    y_labels      : array-like  class labels
    classes       : array-like  ordered unique class labels
    lv_x, lv_y   : int  1-based LV indices for x/y axes
    show_ellipses : bool
    ellipse_alpha : float
    class_colors  : dict or None  {class_label: color}
    sample_labels : array-like or None
        When provided, each point is annotated with its label (sample ID).

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_labels = np.asarray(y_labels)
    cmap = plt.get_cmap('tab10')
    ix, iy = lv_x - 1, lv_y - 1

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    for i, cls in enumerate(classes):
        mask  = y_labels == cls
        x_pts = x_scores[mask, ix]
        y_pts = x_scores[mask, iy]
        color = class_colors[cls] if class_colors else cmap(i % 10)

        ax.scatter(x_pts, y_pts, label=str(cls), color=color,
                   alpha=0.8, s=point_size, edgecolors='k', linewidths=0.4)

        if show_ellipses and x_pts.size > 2:
            cov  = np.cov(x_pts, y_pts)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            w, h  = 2 * 1.96 * np.sqrt(np.abs(vals))
            ell   = Ellipse(
                xy        = (x_pts.mean(), y_pts.mean()),
                width     = w,
                height    = h,
                angle     = theta,
                edgecolor = color,
                facecolor = color,
                alpha     = ellipse_alpha,
                linewidth = 1.2,
            )
            ax.add_patch(ell)

        if sample_labels is not None:
            labels_arr = np.asarray(sample_labels)
            for xp, yp, lbl in zip(x_pts, y_pts, labels_arr[mask]):
                ax.annotate(str(lbl), (xp, yp),
                            fontsize=max(tick_fontsize - 3, 6),
                            xytext=(3, 3), textcoords='offset points',
                            color=color, clip_on=True)

    ax.set_xlabel(f'LV{lv_x}', fontsize=label_fontsize)
    ax.set_ylabel(f'LV{lv_y}', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    ax.axhline(0, color='grey', linewidth=0.6, linestyle='--')
    ax.axvline(0, color='grey', linewidth=0.6, linestyle='--')
    title = f'PLS-DA Score Plot (LV{lv_x} vs LV{lv_y})'
    if analyte:
        title += f' — {analyte}'
    ax.set_title(title, fontsize=label_fontsize)
    if show_legend:
        ax.legend(title='Class', fontsize=tick_fontsize)
    fig.tight_layout()
    return fig


def plot_plsda_loadings(x_loadings, axis, lv_indices,
                        analyte="", fig_width=8.0, fig_height=4.5,
                        label_fontsize=12, tick_fontsize=10,
                        top_n_bands=0):
    """
    Plot PLS-DA X-loadings for one or more latent variables against the
    spectral axis.  Loadings describe which spectral regions drive each LV.

    Parameters
    ----------
    x_loadings  : np.ndarray  (n_features, n_components)
                  from PLSDAClassifier.pls_.x_loadings_
    axis        : array-like  spectral axis (wavenumber / Raman shift)
    lv_indices  : list of int  1-based LV numbers to plot (e.g. [1, 2])
    analyte     : str
    top_n_bands : int  annotate the N bands with the highest |loading| per LV;
                  0 = no annotations

    Returns
    -------
    matplotlib.figure.Figure
    """
    axis = np.asarray(axis)
    cmap = plt.get_cmap('tab10')

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axhline(0, color='grey', linewidth=0.7, linestyle='--')

    for i, lv in enumerate(lv_indices):
        col_idx = lv - 1
        if col_idx >= x_loadings.shape[1]:
            continue
        load_vec = x_loadings[:, col_idx]
        color = cmap(i % 10)
        ax.plot(axis, load_vec, color=color, linewidth=1.4, label=f'LV{lv}')

        if top_n_bands and top_n_bands > 0:
            min_sep     = 10  # minimum spacing between labelled bands (axis units)
            cand_sorted = np.argsort(np.abs(load_vec))[::-1]
            selected    = []
            for idx in cand_sorted:
                if all(abs(axis[idx] - axis[j]) >= min_sep for j in selected):
                    selected.append(idx)
                if len(selected) == int(top_n_bands):
                    break
            for idx in selected:
                xval   = axis[idx]
                yval   = load_vec[idx]
                offset = 5 if yval >= 0 else -5
                va     = 'bottom' if yval >= 0 else 'top'
                ax.annotate(
                    f'{xval:.0f}',
                    xy=(xval, yval),
                    xytext=(0, offset),
                    textcoords='offset points',
                    fontsize=max(tick_fontsize - 3, 6),
                    color=color,
                    ha='center', va=va,
                    clip_on=True,
                )

    ax.set_xlabel('Raman Shift (cm⁻¹)', fontsize=label_fontsize)
    ax.set_ylabel('Loading', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    title = 'PLS-DA Loadings'
    if analyte:
        title += f' — {analyte}'
    ax.set_title(title, fontsize=label_fontsize)
    if ax.get_legend_handles_labels()[0]:
        ax.legend(fontsize=tick_fontsize)
    fig.tight_layout()
    return fig


def plot_plsda_score_distribution(y_score, y_true, classes,
                                  suffix="", analyte="",
                                  class_colors=None,
                                  fig_width=6.0, fig_height=5.0,
                                  label_fontsize=12, tick_fontsize=10,
                                  point_size=20):
    """
    Strip + box plot of the continuous PLS-DA prediction score grouped by
    true class.  For binary models the score is the raw regression output
    (0 = class[0], 1 = class[1]); the decision boundary at 0.5 is drawn
    as a dashed reference line.

    Parameters
    ----------
    y_score  : np.ndarray  output of PLSDAClassifier.decision_function()
    y_true   : array-like  true class labels
    classes  : array-like  ordered unique class labels
    suffix   : str  e.g. '' for calibration, ' (CV)' for cross-val

    Returns
    -------
    matplotlib.figure.Figure
    """
    y_true  = np.asarray(y_true)
    scores  = np.asarray(y_score)
    if scores.ndim > 1:
        scores = scores[:, 0]

    cmap = plt.get_cmap('tab10')
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    positions = np.arange(len(classes))
    box_data  = [scores[y_true == cls] for cls in classes]

    bp = ax.boxplot(box_data, positions=positions, widths=0.35,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', linewidth=1.5))

    for i, (cls, patch) in enumerate(zip(classes, bp['boxes'])):
        color = class_colors[cls] if class_colors else cmap(i % 10)
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        jitter = np.random.default_rng(0).uniform(-0.12, 0.12, size=box_data[i].size)
        ax.scatter(positions[i] + jitter, box_data[i],
                   color=color, s=point_size, alpha=0.7,
                   edgecolors='k', linewidths=0.3, zorder=3)

    if len(classes) == 2:
        ax.axhline(0.5, color='grey', linestyle='--', linewidth=1.0,
                   label='Decision boundary (0.5)')
        ax.legend(fontsize=tick_fontsize)

    ax.set_xticks(positions)
    ax.set_xticklabels([str(c) for c in classes], fontsize=tick_fontsize)
    ax.set_ylabel('PLS-DA Score', fontsize=label_fontsize)
    ax.set_xlabel('True Class', fontsize=label_fontsize)
    ax.tick_params(labelsize=tick_fontsize)
    title = f'PLS-DA Score Distribution{suffix}'
    if analyte:
        title += f' — {analyte}'
    ax.set_title(title, fontsize=label_fontsize)
    fig.tight_layout()
    return fig