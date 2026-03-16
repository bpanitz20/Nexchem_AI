#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from models.wrappers import PLS_model, MLPRegressor_model, MLPClassifier_model
from preprocessors.labeling import bin_targets
from config import DEFAULT_MLP_PARAM_GRID


# ---------------------------------------------------------------------------
# Model registry — add new regression models here; no other file needs to
# change to support a new entry.
# ---------------------------------------------------------------------------

def _run_pls(X, y, results_dir, axis, analyte, groups, n_folds, sample_ids,
             class_labels, manual_param=None, param_grid=None, vip_threshold=None,
             block_params=None):
    return PLS_model(
        X, y, results_dir, axis,
        analyte=analyte, manual_param=manual_param,
        groups=groups, n_folds=n_folds,
        sample_ids=sample_ids, class_labels=class_labels,
        vip_threshold=vip_threshold,
        block_params=block_params,
    )


def _run_mlp(X, y, results_dir, axis, analyte, groups, n_folds, sample_ids,
             class_labels, manual_param=None, param_grid=None):
    return MLPRegressor_model(
        x=X, y=y, directory=results_dir, axis=axis,
        analyte=analyte, param_grid=param_grid,
        groups=groups, n_folds=n_folds,
        sample_ids=sample_ids, class_labels=class_labels
    )


MODEL_REGISTRY = {
    "PLS": _run_pls,
    "MLP": _run_mlp,
}


def run_regression_loop(
    X,
    Y_df,
    results_dir,
    axis,
    model_name="MLP",
    param_name=None,
    param_grid=None,
    param_range=None,
    manual_param=None,
    n_folds=8,
    groups=None,
    sample_ids=None,
    class_labels=None,
    vip_threshold=None,
    block_params=None,
):
    """
    Runs a regression loop across all analytes in Y_df using the specified model.

    Parameters
    ----------
    X : np.ndarray
    Y_df : pd.DataFrame
    results_dir : str
    axis : array-like
    model_name : str
        Key into MODEL_REGISTRY ("PLS", "MLP", …).
    param_grid : dict or None
        MLP hyperparameter grid; defaults to DEFAULT_MLP_PARAM_GRID.
    manual_param : int or None
        Fixed n_components for PLS (skips grid search).
    n_folds : int
    groups : array-like or None
    sample_ids : list or None
    class_labels : array-like or None

    Returns
    -------
    dict  {analyte: result_dict}
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unsupported model: '{model_name}'. "
            f"Available: {list(MODEL_REGISTRY)}"
        )

    if model_name == "MLP" and param_grid is None:
        param_grid = DEFAULT_MLP_PARAM_GRID

    run_fn = MODEL_REGISTRY[model_name]
    results_all = {}

    # vip_threshold and block_params are only meaningful for PLS; never forwarded to MLP.
    call_kwargs = {'manual_param': manual_param, 'param_grid': param_grid}
    if model_name == "PLS":
        if vip_threshold is not None:
            call_kwargs['vip_threshold'] = vip_threshold
        if block_params is not None:
            call_kwargs['block_params'] = block_params

    for analyte in Y_df.columns:
        print(f"\n🔬 Regression for: {analyte}")
        y = Y_df[analyte].values.reshape(-1, 1)
        results_all[analyte] = run_fn(
            X, y, results_dir, axis, analyte, groups, n_folds,
            sample_ids, class_labels,
            **call_kwargs
        )

    return results_all


def run_classification_loop(X, Y_df, results_dir, axis, bins=[0, 0.4, 0.7, 1], groups=None):
    """
    Loop through each Y column (analyte) and run classification models.

    Parameters
    ----------
    X : np.ndarray
    Y_df : pd.DataFrame
    results_dir : str
    axis : array-like
    bins : list of float
        Bin edges for converting continuous Y to class labels.
    groups : array-like or None
    """
    print("\n🚀 Starting classification model training...")
    for analyte in Y_df.columns:
        print(f"\n🔬 Classification for: {analyte}")
        y_continuous = Y_df[analyte].values
        y_class = bin_targets(y_continuous, bins=bins)
        print("Binned class distribution:", np.unique(y_class, return_counts=True))
        MLPClassifier_model(X, y_class, results_dir, axis, analyte=analyte, groups=groups)
