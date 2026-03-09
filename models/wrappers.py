#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Wrappers
"""
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from models.cross_val import KFold_CV
from models.cross_val import KFold_Gridsearch_CV
from sklearn.metrics import accuracy_score, f1_score
from plotting.plot_regression import (
    plot_pred_vs_actual,
    plot_coefficients,
    plot_feature_importance,
    plot_vip_scores,
    print_model_summary,
    print_CV_table,
    plot_t2_q_residuals,
    plot_pls_scores,
    plot_cv_performance,
)
from plotting.plot_classifier import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_decision_boundary
)

from sklearn.base import BaseEstimator, TransformerMixin

class PLSFeaturizer(BaseEstimator, TransformerMixin):
    """
    Wrapper around PLSRegression that exposes only the X-scores (T)
    so it can be safely used inside a sklearn Pipeline.
    """
    def __init__(self, n_components=2, scale=False):
        self.n_components = n_components
        self.scale = scale
        self.pls_ = None

    def fit(self, X, y=None):
        self.pls_ = PLSRegression(
            n_components=self.n_components,
            scale=self.scale
        )
        self.pls_.fit(X, y)
        return self

    def transform(self, X):
        # transform returns X_scores (T); 2D
        X_scores = self.pls_.transform(X)
        return X_scores







def _pls_compute(x, y, directory, axis, max_lv=15, analyte="", groups=None,
                 manual_param=None, sample_ids=None, n_folds=8, class_labels=None):
    """
    Pure computation layer for PLS regression.
    Runs cross-validation, fits the final model, and computes all metrics.
    No plots are generated. Use plot_pls_results() to produce diagnostic plots.

    Returns a result dict with identical keys to PLS_model(), plus:
      'param_name'  — hyperparameter name ('n_components')
      'param_range' — list of values swept during CV
    """
    param_name = 'n_components'
    param_range = list(range(1, max_lv + 1))
    model = PLSRegression()

    # Run CV
    cv_results = KFold_CV(x, y, model, param_name, param_range, analyte=analyte,
                          groups=groups, model_name='PLS', directory=directory,
                          manual_param=manual_param, n_folds=n_folds, sample_ids=sample_ids,
                          class_labels=class_labels)
    fold_df = cv_results.get("fold_df")

    # Final fit with optimal parameter
    if manual_param is not None:
        final_param = manual_param
    else:
        final_param = cv_results['optimal_param']

    # Fit model
    model.set_params(**{param_name: final_param})
    model.fit(x, y - cv_results['y_mean'])
    Y_pred = model.predict(x)

    # Metrics
    final_r2 = r2_score(y - cv_results['y_mean'], Y_pred)
    final_mse = mean_squared_error(y - cv_results['y_mean'], Y_pred)
    final_r2_CV = cv_results['pooled_r2_CV'][param_range.index(cv_results['optimal_param'])]
    final_rmse_CV = cv_results['pooled_rmse_CV'][param_range.index(cv_results['optimal_param'])]

    # Save CV summary CSV and print console summary
    print_CV_table(
        param_name=param_name,
        param_range=param_range,
        r2_cv=cv_results['pooled_r2_CV'],
        r2_cal=cv_results['mean_r2_cal'],
        mse_cv=cv_results['pooled_rmse_CV'],
        rmse_cal=cv_results['mean_rmse_cal'],
        model_name="PLS",
        analyte=analyte,
        directory=directory
    )

    cv_table_df = pd.DataFrame({
        param_name: param_range,
        "R²_Cal": cv_results['mean_r2_cal'],
        "R²_CV": cv_results['pooled_r2_CV'],
        "RMSE_CV": cv_results['pooled_rmse_CV'],
        "RMSE_Cal": cv_results['mean_rmse_cal']
    })

    summary_string = print_model_summary(
        model_name="PLS",
        analyte=analyte,
        final_r2=final_r2,
        final_r2_CV=final_r2_CV,
        final_mse=final_mse,
        final_rmse_CV=final_rmse_CV,
        optimal_param=cv_results['optimal_param'],
        param_name=param_name
    )

    return {
        'model': model,
        'final_r2': final_r2,
        'final_mse': final_mse,
        'cv_results': cv_results,
        'cv_table_df': cv_table_df,
        'cv_r2_plot_path': cv_results['cv_r2_plot_path'],
        'cv_rmse_plot_path': cv_results['cv_rmse_plot_path'],
        'cv_pred_plot_path': cv_results['cv_pred_plot_path'],
        'summary': summary_string,
        'vip_plot_path': os.path.join(directory, f"VIP_Scores_PLS_{analyte}.png"),
        'coef_plot_path': os.path.join(directory, f"PLS_Coefficients_{analyte}.png"),
        't2_plot_path': os.path.join(directory, f"T2_vs_Q_Residuals_PLS_{analyte}.png"),
        'final_pred_plot_path': os.path.join(directory, f"Final_Pred_vs_Actual_PLS_{analyte}.png"),
        'fold_df': fold_df,
        'scoreplot_path': None,       # populated by plot_pls_results()
        # extras consumed by the reporting layer
        'param_name': param_name,
        'param_range': param_range,
    }


def plot_pls_results(results, x, y, axis, directory, analyte,
                     sample_ids=None, class_labels=None):
    """
    Reporting layer for PLS. Generates all diagnostic plots from a _pls_compute()
    result dict.  Call this after _pls_compute() when plots are needed.

    Returns the scoreplot_path string (or None if n_components < 2).
    """
    model      = results['model']
    cv_results = results['cv_results']
    param_name = results['param_name']
    param_range = results['param_range']
    final_param = cv_results['optimal_param']
    y_mean      = cv_results['y_mean']

    # CV performance curves (moved here from KFold_CV)
    plot_cv_performance(
        param_range,
        cv_results['pooled_r2_CV'],
        cv_results['mean_r2_cal'],
        cv_results['pooled_rmse_CV'],
        cv_results['mean_rmse_cal'],
        param_name, analyte, "PLS", directory
    )

    # CV predicted vs actual (moved here from KFold_CV)
    plot_pred_vs_actual(
        y,
        cv_results['Y_pred_CV'] + y_mean,
        directory,
        f'CV Predicted vs. Actual for {analyte} (PLS)',
        f'CV_Pred_vs_Actual_PLS_{analyte}.png',
        class_labels=class_labels
    )

    # Final model diagnostics
    plot_t2_q_residuals(model, x, y, analyte, directory, model_name="PLS", sample_ids=sample_ids)

    Y_pred = model.predict(x)
    plot_pred_vs_actual(
        y,
        Y_pred + y_mean,
        directory,
        f"Final Predicted vs. Actual for {analyte} (PLS)",
        f"Final_Pred_vs_Actual_PLS_{analyte}.png",
        class_labels=class_labels
    )

    plot_coefficients(axis, model.coef_, directory, "PLS", analyte)
    plot_vip_scores(model, x, axis, directory, "PLS", analyte)

    scoreplot_path = None
    if final_param >= 2:
        scoreplot_path = plot_pls_scores(
            model=model,
            x=x,
            directory=directory,
            analyte=analyte,
            class_labels=class_labels
        )

    return scoreplot_path


def PLS_model(x, y, directory, axis, max_lv=15, analyte="", groups=None,
              manual_param=None, sample_ids=None, n_folds=8, class_labels=None):
    """
    Public API for PLS regression. Signature and return dict are unchanged.
    Internally delegates to _pls_compute() (pure computation) and
    plot_pls_results() (all diagnostic plots).
    """
    results = _pls_compute(x, y, directory, axis, max_lv, analyte, groups,
                           manual_param, sample_ids, n_folds, class_labels)
    scoreplot_path = plot_pls_results(results, x, y, axis, directory, analyte,
                                      sample_ids=sample_ids, class_labels=class_labels)
    results['scoreplot_path'] = scoreplot_path
    return results

def _mlp_compute(x, y, directory, axis, analyte="",
                 param_grid=None, groups=None,
                 random_state=42, n_folds=8, sample_ids=None, class_labels=None):
    """
    Pure computation layer for MLP regression.
    Runs grid-search cross-validation, selects the best model, and computes all
    metrics. No plots are generated. Use plot_mlp_results() for diagnostics.

    Returns a result dict with identical keys to MLPRegressor_model(), plus
    'Y_pred_CV' inside the nested cv_results sub-dict (needed by reporting layer).
    """
    y = np.array(y).ravel()

    if param_grid is None:
        param_grid = {
            'pls__n_components': [4, 5, 6],
            'mlp__hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'mlp__activation': ['relu'],
            'mlp__alpha': [0.02, 0.01, 0.0009],
            'mlp__learning_rate_init': np.linspace(0.0001, 0.01, 10).tolist(),
            'mlp__early_stopping': [True],
            'mlp__solver': ['adam']
        }

    base_mlp = MLPRegressor(
        max_iter=2000,
        random_state=random_state,
        verbose=False,
        n_iter_no_change=20,
        tol=1e-4,
        batch_size='auto'
    )

    # Pipeline: PLSFeaturizer -> StandardScaler -> MLP
    pipeline = Pipeline([
        ('pls', PLSFeaturizer(scale=False)),
        ('scaler', StandardScaler()),
        ('mlp', base_mlp)
    ])

    cv_results = KFold_Gridsearch_CV(
        x=x,
        y=y,
        model=pipeline,
        param_grid=param_grid,
        task="regression",
        analyte=analyte,
        model_name="MLP",
        groups=groups,
        directory=directory,
        n_folds=n_folds,
        sample_ids=sample_ids,
        class_labels=class_labels
    )
    fold_df = cv_results.get("fold_df")

    final_model = cv_results['best_estimator']
    Y_pred = final_model.predict(x) + cv_results['y_mean']

    final_r2 = r2_score(y, Y_pred)
    final_mse = mean_squared_error(y, Y_pred)

    Y_pred_CV = cv_results['Y_pred_CV'] + cv_results['y_mean']
    final_r2_CV = r2_score(y, Y_pred_CV)
    mse_CV = mean_squared_error(y, Y_pred_CV)
    rmse_CV = np.sqrt(mse_CV)

    summary_string = print_model_summary(
        model_name="MLP",
        analyte=analyte,
        final_r2=final_r2,
        final_r2_CV=final_r2_CV,
        final_mse=final_mse,
        final_rmse_CV=rmse_CV,
        best_params=cv_results['best_params']
    )

    return {
        'model': final_model,
        'final_r2': final_r2,
        'final_mse': final_mse,
        'cv_results': {
            'cv_results': cv_results['cv_results'],
            'best_params': cv_results['best_params'],
            'y_mean': cv_results['y_mean'],
            'Y_pred_CV': Y_pred_CV,   # needed by plot_mlp_results()
        },
        'best_params': cv_results['best_params'],
        'cv_pred_plot_path': cv_results['cv_pred_plot_path'],
        'cv_table_df': cv_results['cv_table_df'],
        'summary': summary_string,
        'feature_importance_path': os.path.join(directory, f"Feature_Importance_MLP_{analyte}.png"),
        'final_pred_plot_path': os.path.join(directory, f"Final_Pred_vs_Actual_MLP_{analyte}.png"),
        'fold_df': fold_df,
    }


def plot_mlp_results(results, x, y, axis, directory, analyte,
                     sample_ids=None, class_labels=None):
    """
    Reporting layer for MLP. Generates all diagnostic plots from a _mlp_compute()
    result dict. Call this after _mlp_compute() when plots are needed.
    """
    final_model = results['model']
    y_mean      = results['cv_results']['y_mean']
    Y_pred_CV   = results['cv_results']['Y_pred_CV']

    # CV predicted vs actual (moved here from KFold_Gridsearch_CV)
    plot_pred_vs_actual(
        y,
        Y_pred_CV,
        directory,
        f'CV Predicted vs. Actual for {analyte} (MLP)',
        f'CV_Pred_vs_Actual_MLP_{analyte}.png',
        class_labels=class_labels
    )

    # Final model diagnostics
    plot_feature_importance(
        model=final_model,
        x=x,
        y=y,
        axis=axis,
        directory=directory,
        model_name="MLP",
        analyte=analyte
    )

    Y_pred = final_model.predict(x) + y_mean
    plot_pred_vs_actual(
        y,
        Y_pred,
        directory,
        f"Final Predicted vs. Actual for {analyte} (MLP)",
        f"Final_Pred_vs_Actual_MLP_{analyte}.png",
        class_labels=class_labels
    )


def MLPRegressor_model(x, y, directory, axis, analyte="",
                       param_grid=None, groups=None,
                       random_state=42, n_folds=8, sample_ids=None, class_labels=None):
    """
    Public API for MLP regression. Signature and return dict are unchanged.
    Internally delegates to _mlp_compute() (pure computation) and
    plot_mlp_results() (all diagnostic plots).
    """
    results = _mlp_compute(x, y, directory, axis, analyte, param_grid, groups,
                           random_state, n_folds, sample_ids, class_labels)
    plot_mlp_results(results, x, np.array(y).ravel(), axis, directory, analyte,
                     sample_ids=sample_ids, class_labels=class_labels)
    return results


"""
Classification Model Wrppers
"""

def MLPClassifier_model(x, y, directory, axis, analyte="", param_grid=None, groups=None, random_state=42):
    y = np.array(y).ravel()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    if param_grid is None:
        param_grid = {
            'alpha': [0.01, 0.001],
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'learning_rate_init': [0.001],
            'early_stopping': [True]
        }

    base_model = MLPClassifier(
        max_iter=2000,
        random_state=random_state,
        tol=1e-4,
        verbose=False
    )

    # Run GridSearch CV
    cv_results = KFold_Gridsearch_CV(
        x=X_scaled,
        y=y,
        model=base_model,
        param_grid=param_grid,
        task="classification",
        groups=groups,
        analyte=analyte,
        model_name="MLPClassifier",
        directory=directory
    )

    # Final model and predictions
    final_model = cv_results["best_estimator"]
    y_pred = final_model.predict(X_scaled)
    y_cv_pred = cv_results["Y_pred_CV"]
    y_cv_proba = cv_results["Y_proba_CV"]

    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    acc_cv = accuracy_score(y, y_cv_pred)
    f1_cv = f1_score(y, y_cv_pred, average='macro')

    print(f"\n📊 MLPClassifier Performance for {analyte}")
    print(f"Best Params: {cv_results['best_params']}")
    print(f"Accuracy (Train): {acc:.3f} | F1 (Train): {f1:.3f}")
    print(f"Accuracy (CV):    {acc_cv:.3f} | F1 (CV): {f1_cv:.3f}")

    # Plots
    plot_confusion_matrix(y_pred, y, directory, "MLPClassifier", analyte)
    plot_confusion_matrix(y_cv_pred, y, directory, "MLPClassifier", analyte, suffix="_CV")

    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(X_scaled)
        plot_roc_curve(y, y_proba, directory, "MLPClassifier", analyte)
        if y_cv_proba is not None:
            plot_roc_curve(y, y_cv_proba, directory, "MLPClassifier", analyte, suffix="_CV")

    plot_decision_boundary(X_scaled, y, final_model, directory, "MLPClassifier", analyte)

    return {
        "model": final_model,
        "best_params": cv_results["best_params"],
        "accuracy": acc,
        "f1": f1,
        "cv_accuracy": acc_cv,
        "cv_f1": f1_cv,
    }






from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def PCA_model(
    X,
    classes,
    axis,
    directory,
    n_components=8,
    show_ellipses=True,
    ellipse_alpha=0.2,
    pc_x=1,                 # 1-based index for x-axis PC (PC1 by default)
    pc_y=2                  # 1-based index for y-axis PC (PC2 by default)
):
    """
    PCA with selectable PCs on x/y axes and optional class-colored 95% confidence ellipses.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (preprocessed).
    classes : list or np.ndarray
        Class labels for coloring points.
    axis : list or np.ndarray
        Spectral axis (kept for parity; not used here).
    directory : str
        Output directory for saved figure.
    n_components : int
        Number of PCA components to compute (will be increased if needed by pc_x/pc_y).
    show_ellipses : bool
        Whether to show 95% confidence ellipses for each class.
    ellipse_alpha : float
        Transparency for ellipses (0–1).
    pc_x : int
        1-based PC index to plot on x-axis (PC1 -> 1).
    pc_y : int
        1-based PC index to plot on y-axis (PC2 -> 2).

    Returns
    -------
    dict with:
        - pca_model : fitted PCA object
        - X_pca : PCA scores (n_samples, n_components)
        - explained_variance : explained variance ratio per component
        - used_components : (pc_x, pc_y) as 1-based integers
    """
    if pc_x < 1 or pc_y < 1:
        raise ValueError("pc_x and pc_y must be 1-based indices (>= 1).")

    # Ensure we compute enough components for the requested axes
    required = max(pc_x, pc_y)
    n_components = max(n_components, required)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Run PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    var_ratio = pca.explained_variance_ratio_

    # Convert to 0-based for indexing
    ix = pc_x - 1
    iy = pc_y - 1

    # Plot setup
    plt.figure(figsize=(8, 6))
    unique_classes = np.unique(classes)
    color_map = plt.get_cmap('tab10')

    for i, cls in enumerate(unique_classes):
        mask = np.array(classes) == cls
        x_pts = X_pca[mask, ix]
        y_pts = X_pca[mask, iy]
        color = color_map(i % 10)

        # Scatter plot
        plt.scatter(x_pts, y_pts, label=str(cls), alpha=0.7, color=color)

        # 95% ellipse per class
        if show_ellipses and x_pts.size > 2:
            cov = np.cov(x_pts, y_pts)
            vals, vecs = np.linalg.eigh(cov)
            order = vals.argsort()[::-1]
            vals, vecs = vals[order], vecs[:, order]
            theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
            width, height = 2 * 1.96 * np.sqrt(vals)  # 95% CI
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

    # Final plot formatting
    x_label = f'PC{pc_x} ({var_ratio[ix]*100:.1f}%)'
    y_label = f'PC{pc_y} ({var_ratio[iy]*100:.1f}%)'
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(f'PCA Score Plot (PC{pc_x} vs PC{pc_y})')
    plt.legend(title="Class")
    plt.grid(True)

    os.makedirs(directory, exist_ok=True)
    outname = f"PCA_PC{pc_x}_vs_PC{pc_y}.png"
    plt.savefig(os.path.join(directory, outname), dpi=300, bbox_inches="tight")
    plt.show()

    return {
        "pca_model": pca,
        "X_pca": X_pca,
        "explained_variance": var_ratio,
        "used_components": (pc_x, pc_y),
    }
