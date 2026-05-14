#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Plotting Functions for Regression Model Evaluation

This module contains plotting functions for PCA Visualization, Designed for modular use
within NexChem_AI.

Author: Ben Panitz
Created: July 21, 2025
"""

import matplotlib.pyplot as plt
import numpy as np
import os

def plot_pca_loadings(pca_model, axis, directory, components=[0, 1], top_n=3, label_fontsize=12, tick_fontsize=10, fig_width=10, fig_height=5, title=None, show_legend=True):
    """
    Plot PCA loadings and label top N influential bands on each component.

    Parameters:
    -----------
    pca_model : sklearn.decomposition.PCA
        Trained PCA object.
    axis : np.ndarray
        Spectral axis (e.g., Raman shift).
    directory : str
        Output directory.
    components : list
        Which PCs to plot (default: [0, 1]).
    top_n : int
        Number of top influential bands to label per PC.
    """
    loadings = pca_model.components_
    plt.figure(figsize=(fig_width, fig_height))

    for idx in components:
        pc_load = loadings[idx]
        label = f'PC{idx + 1}'
        plt.plot(axis, pc_load, label=label)

        if top_n == 0:
            continue

        # Annotate top N influential bands (by absolute value), avoid overlap
        sorted_indices = np.argsort(np.abs(pc_load))[::-1]
        annotated = 0
        used_positions = []

        for i in sorted_indices:
            x = axis[i]
            y = pc_load[i]
            
            # Skip too-close labels
            if any(abs(x - xp) < 10 for xp in used_positions):  # Adjust spacing as needed
                continue
            
            plt.annotate(f"{int(x)}", (x, y),
                         textcoords="offset points",
                         xytext=(0, 8),
                         ha='center',
                         fontsize=11,
                         fontweight='bold',
                         color='black',
                         arrowprops=dict(arrowstyle="->", color='gray', lw=0.8))
            used_positions.append(x)
            annotated += 1
            if annotated >= top_n:
                break
            
    plt.xlabel("Wavenumber (cm⁻¹)", fontsize=label_fontsize)
    plt.ylabel("Loading Weight", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.title(title if title else "PCA Loadings")
    if show_legend:
        plt.legend()
    plt.tight_layout()

    os.makedirs(directory, exist_ok=True)
    _base = os.path.join(directory, "PCA_Loadings_Annotated")
    plt.savefig(_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(_base + ".pdf", bbox_inches="tight")
    plt.close()


def plot_yblock_pca_loadings(pca_model, feature_names, directory, components=[0, 1], top_n=3, label_fontsize=12, tick_fontsize=10, fig_width=None, fig_height=5, title=None, show_legend=True):
    """
    Plot PCA loadings for Y-block (e.g. fatty acid) data and label top N features.

    Parameters:
    -----------
    pca_model : sklearn.decomposition.PCA
        Trained PCA object.
    feature_names : list of str
        Names of Y-block features (e.g. fatty acid column names).
    directory : str
        Output directory.
    components : list
        Which PCs to plot (default: [0, 1]).
    top_n : int
        Number of top influential features to label per PC.
    """
    loadings = pca_model.components_
    n_features = loadings.shape[1]
    x_indices = np.arange(n_features)

    _fig_width = fig_width if fig_width is not None else max(10, n_features * 0.6)
    plt.figure(figsize=(_fig_width, fig_height))

    for idx in components:
        pc_load = loadings[idx]
        label = f'PC{idx + 1}'
        plt.plot(x_indices, pc_load, label=label, marker='o', markersize=4)

        if top_n == 0:
            continue

        sorted_indices = np.argsort(np.abs(pc_load))[::-1]
        annotated = 0
        used_positions = []

        for i in sorted_indices:
            x = x_indices[i]
            y = pc_load[i]

            if any(abs(x - xp) < 2 for xp in used_positions):
                continue

            plt.annotate(feature_names[i], (x, y),
                         textcoords="offset points",
                         xytext=(0, 10),
                         ha='center',
                         fontsize=9,
                         fontweight='bold',
                         color='black',
                         rotation=45,
                         arrowprops=dict(arrowstyle="->", color='gray', lw=0.8))
            used_positions.append(x)
            annotated += 1
            if annotated >= top_n:
                break

    plt.xticks(x_indices, feature_names, rotation=45, ha='right', fontsize=9)
    plt.xlabel("Feature", fontsize=label_fontsize)
    plt.ylabel("Loading Weight", fontsize=label_fontsize)
    plt.tick_params(labelsize=tick_fontsize)
    plt.title(title if title else "PCA Loadings (Y-block)")
    if show_legend:
        plt.legend()
    plt.tight_layout()

    os.makedirs(directory, exist_ok=True)
    _base = os.path.join(directory, "PCA_Loadings_YBlock_Annotated")
    plt.savefig(_base + ".png", dpi=300, bbox_inches="tight")
    plt.savefig(_base + ".pdf", bbox_inches="tight")
    plt.close()


def plot_pcada_cv_curve(accuracies, selected_n, directory):
    """Plot CV accuracy vs number of PCA components for PCA-DA.

    Parameters
    ----------
    accuracies : list of float
        Mean CV accuracy for n = 1 … len(accuracies).
    selected_n : int
        Currently selected n_components — marked with a dashed vertical line.
    directory : str or None
        If provided, saves PNG + PDF. Returns the Figure either way.
    """
    n_range = list(range(1, len(accuracies) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(n_range, accuracies, marker="o", color="#1f77b4", linewidth=1.8,
            markersize=5, label="CV Accuracy")
    ax.axvline(selected_n, color="red", linestyle="--", linewidth=1.2,
               label=f"Selected (n={selected_n})")

    ax.set_xlabel("Number of PCA Components", fontsize=12)
    ax.set_ylabel("CV Classification Accuracy", fontsize=12)
    ax.set_title("PCA-DA: CV Accuracy vs Number of PCA Components", fontsize=12)
    ax.set_xticks(n_range)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    fig.tight_layout()

    if directory:
        os.makedirs(directory, exist_ok=True)
        _base = os.path.join(directory, "PCADA_CV_Accuracy")
        fig.savefig(_base + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(_base + ".pdf", bbox_inches="tight")

    return fig
