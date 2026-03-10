#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:20:48 2025

@author: bp
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA



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
    filename = f'ConfusionMatrix_{model_name}_{analyte}{suffix}.png'
    plt.savefig(os.path.join(directory, filename), dpi=300)
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

    filename = f"ROC_{model_name}_{analyte}{suffix}.png"
    plt.savefig(os.path.join(directory, filename), dpi=300, bbox_inches="tight")
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
    filename = f'DecisionBoundary_{model_name}_{analyte}.png'
    plt.savefig(os.path.join(directory, filename), dpi=300)
    plt.close()