#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIP (Variable Importance in Projection) helpers for PLS models.

Provides a standalone calculate_vip() so the formula is defined once and
shared by both the modeling layer (_pls_compute) and the plotting layer
(plot_vip_scores).
"""

import numpy as np


def calculate_vip(pls_model) -> np.ndarray:
    """
    Compute VIP scores from a fitted sklearn PLSRegression model.

    Formula (Wold 1993):
        VIP_j = sqrt( p * sum_h( (W_jh / ||W_h||)^2 * SS_h ) / SS_total )

    where:
        p      = number of features
        W      = x_weights_            shape (p, n_components)
        T      = x_scores_             shape (n_samples, n_components)
        Q      = y_loadings_           shape (n_targets, n_components)
        SS_h   = ||T_h||^2 * Q_h^2    variance of y explained by component h

    Parameters
    ----------
    pls_model : fitted sklearn.cross_decomposition.PLSRegression

    Returns
    -------
    np.ndarray, shape (n_features,), all values >= 0.
    """
    t = pls_model.x_scores_    # (n_samples,  n_components)
    w = pls_model.x_weights_   # (n_features, n_components)
    q = pls_model.y_loadings_  # (n_targets,  n_components)

    p, h = w.shape
    s        = np.square(t).sum(axis=0) * np.square(q).flatten()
    total_s  = np.sum(s)
    return np.sqrt(p * (np.dot(np.square(w), s)) / total_s)
