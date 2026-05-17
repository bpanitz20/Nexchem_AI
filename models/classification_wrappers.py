#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classification Model Wrappers (PLS-DA and future classifiers).

Kept entirely separate from wrappers.py so the regression pipeline
is not touched.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from models.cross_val import KFold_CV_class


# ── sklearn-compatible PLS-DA estimator ─────────────────────────────────────

class PLSDAClassifier(BaseEstimator, ClassifierMixin):
    """
    Thin sklearn wrapper around PLSRegression that implements a
    PLS Discriminant Analysis classifier.

    Binary:     Y is encoded as a single 0/1 dummy column; prediction
                threshold is 0.5 on the continuous regression score.
    Multiclass: Y is one-hot encoded; prediction is argmax over columns.

    Parameters
    ----------
    n_components : int
    scale : bool  passed through to PLSRegression
    """

    def __init__(self, n_components=2, scale=False):
        self.n_components = n_components
        self.scale = scale

    def fit(self, X, y):
        self.classes_ = np.sort(np.unique(y))
        self._n_classes = len(self.classes_)

        if self._n_classes == 2:
            y_dummy = (y == self.classes_[1]).astype(float)
        else:
            y_dummy = (y[:, None] == self.classes_[None, :]).astype(float)

        self.pls_ = PLSRegression(
            n_components=self.n_components, scale=self.scale
        )
        self.pls_.fit(X, y_dummy)
        return self

    def decision_function(self, X):
        """Continuous PLS-DA score. Binary: 1-D array. Multiclass: 2-D."""
        return self.pls_.predict(X).squeeze()

    def predict(self, X):
        scores = self.decision_function(X)
        if self._n_classes == 2:
            idx = (scores >= 0.5).astype(int)
        else:
            idx = np.argmax(scores, axis=1)
        return self.classes_[idx]


# ── Computation layer ────────────────────────────────────────────────────────

def _plsda_compute(x, y_labels, directory, axis, max_lv=15,
                   analyte="", groups=None, manual_param=None,
                   sample_ids=None, n_folds=8):
    """
    Pure computation layer for PLS-DA classification.

    Mirrors the structure of _pls_compute in wrappers.py:
      1. Cap max_lv to safe range given CV fold sizes and feature count.
      2. Run KFold_CV_class sweep to select optimal n_components.
      3. Fit final PLSDAClassifier on full data.
      4. Collect calibration and CV predictions, metrics, LV scores.

    Parameters
    ----------
    x           : np.ndarray  (n_samples, n_features)
    y_labels    : array-like  class label strings/ints
    directory   : str   output directory (may be empty string)
    axis        : array-like  spectral axis for coefficient plot
    max_lv      : int   upper bound for n_components sweep
    analyte     : str   used in filenames / console logs
    groups      : array-like or None  replicate group labels
    manual_param : int or None  fix n_components (skip optimisation)
    sample_ids  : array-like or None
    n_folds     : int

    Returns
    -------
    dict — all keys needed by plot_plsda_results and the Streamlit UI
    """
    from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold

    y_labels   = np.asarray(y_labels)
    classes    = np.sort(np.unique(y_labels))
    n_classes  = len(classes)
    param_name = 'n_components'

    # ── Cap max_lv: must not exceed min training-fold size or feature count ──
    _n_samples, _n_features = x.shape
    if groups is not None:
        _cv_cap = StratifiedGroupKFold(n_splits=n_folds)
        _cap_kw = {"groups": np.asarray(groups)}
    else:
        _cv_cap = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
        _cap_kw = {}
    _min_train = min(
        len(tr) for tr, _ in _cv_cap.split(x, y_labels, **_cap_kw)
    )
    _safe_max = min(max_lv, _min_train, _n_features)
    if _safe_max < max_lv:
        print(
            f"[PLS-DA] max_lv capped {max_lv} → {_safe_max} "
            f"(min CV train fold: {_min_train}, features: {_n_features})"
        )
    param_range = list(range(1, _safe_max + 1))

    model = PLSDAClassifier(scale=False)

    # ── CV sweep ─────────────────────────────────────────────────────────────
    cv_results = KFold_CV_class(
        x, y_labels, model,
        param_name  = param_name,
        param_range = param_range,
        n_folds     = n_folds,
        groups      = groups,
        analyte     = analyte,
        model_name  = 'PLSDA',
        directory   = directory,
        manual_param = manual_param,
        sample_ids  = sample_ids,
    )
    fold_df       = cv_results.get('fold_df')
    optimal_param = cv_results['optimal_param']

    # ── Final model on full data ──────────────────────────────────────────────
    final_model = PLSDAClassifier(
        n_components=optimal_param, scale=False
    )
    final_model.fit(x, y_labels)

    # Calibration predictions
    y_pred_cal   = final_model.predict(x)
    y_score_cal  = final_model.decision_function(x)   # continuous
    cal_accuracy = float(accuracy_score(y_labels, y_pred_cal))
    cal_cm       = confusion_matrix(y_labels, y_pred_cal, labels=classes)

    # CV predictions
    y_pred_cv    = cv_results['Y_pred_CV']
    y_score_cv   = cv_results['Y_score_CV']           # continuous
    cv_accuracy  = cv_results['pooled_acc_CV']
    cv_cm        = confusion_matrix(y_labels, y_pred_cv, labels=classes)

    # LV scores (T) for scores plot — from the fitted PLS inside the wrapper
    x_scores = final_model.pls_.transform(x)  # (n_samples, optimal_param)

    # PLS-DA regression coefficients
    coefficients = final_model.pls_.coef_.ravel()

    # CV summary table (mirrors cv_table_df in regression)
    cv_table_df = pd.DataFrame({
        'n_components':   param_range,
        'CV Accuracy':    [round(a, 4) for a in cv_results['mean_acc_CV']],
        'Cal Accuracy':   [round(a, 4) for a in cv_results['mean_acc_cal']],
    })

    return {
        'model':          final_model,
        'classes':        classes,
        'n_classes':      n_classes,
        'param_name':     param_name,
        'param_range':    param_range,
        'optimal_param':  optimal_param,
        'cv_results':     cv_results,
        'cv_table_df':    cv_table_df,
        'y_true':         y_labels,
        'y_pred_cal':     y_pred_cal,
        'y_pred_cv':      y_pred_cv,
        'y_score_cal':    y_score_cal,
        'y_score_cv':     y_score_cv,
        'cal_accuracy':   cal_accuracy,
        'cv_accuracy':    cv_accuracy,
        'cal_cm':         cal_cm,
        'cv_cm':          cv_cm,
        'x_scores':       x_scores,
        'coefficients':   coefficients,
        'axis':           np.asarray(axis),
        'fold_df':        fold_df,
        'sample_ids':     sample_ids,
    }


# ── Public API ───────────────────────────────────────────────────────────────

def PLSDA_model(x, y_labels, directory, axis, max_lv=15,
                analyte="", groups=None, manual_param=None,
                sample_ids=None, n_folds=8):
    """
    Public entry point for PLS-DA classification.

    Calls _plsda_compute (computation) and returns the result dict.
    Plotting is handled separately by the Streamlit UI using the
    functions in plotting/plot_classifier.py.

    Parameters match _plsda_compute.
    """
    return _plsda_compute(
        x           = x,
        y_labels    = y_labels,
        directory   = directory,
        axis        = axis,
        max_lv      = max_lv,
        analyte     = analyte,
        groups      = groups,
        manual_param = manual_param,
        sample_ids  = sample_ids,
        n_folds     = n_folds,
    )
