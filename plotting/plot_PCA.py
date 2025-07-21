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

def plot_pca_loadings(pca_model, axis, directory, components=[0, 1], top_n=3):
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
    plt.figure(figsize=(10, 5))

    for idx in components:
        pc_load = loadings[idx]
        label = f'PC{idx + 1}'
        plt.plot(axis, pc_load, label=label)

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
            
    plt.xlabel("Wavenumber (cm⁻¹)")
    plt.ylabel("Loading Weight")
    plt.title("PCA Loadings")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    os.makedirs(directory, exist_ok=True)
    plt.savefig(os.path.join(directory, "PCA_Loadings_Annotated.png"), dpi=300, bbox_inches="tight")
    plt.show()
