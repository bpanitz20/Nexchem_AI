#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VIP-based variable selection.

VIPSelector uses the full-spectrum VIP scores computed during Stage 1 of
_pls_compute to retain only variables with VIP >= threshold.  No additional
cross-validation is needed — the mask is derived directly from the pre-computed
scores, making this much faster than iPLS block selection.

Conventional threshold: 1.0 (variables that contribute above average).
"""

import numpy as np
from models.selectors.base import SelectionResult


class VIPSelector:
    """
    Retain spectral variables whose VIP score meets or exceeds a threshold.

    Parameters
    ----------
    threshold : float
        Minimum VIP score for a variable to be retained.
        Default 1.0 — the conventional chemometrics cutoff.
    """

    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold

    def fit(
        self,
        x,
        y_centered,
        axis,
        n_folds,
        groups,
        vip_scores_full,
        scoring_n_components,
    ) -> SelectionResult:
        """
        Build a selection mask from pre-computed VIP scores.

        Parameters
        ----------
        x, y_centered, n_folds, groups, scoring_n_components
            Accepted for API compatibility with BlockSelector; not used.
        axis : np.ndarray
            Full spectral axis.
        vip_scores_full : np.ndarray
            VIP scores from the Stage 1 preliminary PLS model.

        Returns
        -------
        SelectionResult
        """
        axis_arr = np.asarray(axis)
        vip = np.asarray(vip_scores_full)

        selected_mask = vip >= self.threshold

        # Safety guard: if threshold is too strict and removes everything,
        # fall back to the top 50% of variables by VIP score.
        if selected_mask.sum() == 0:
            cutoff = np.percentile(vip, 50)
            selected_mask = vip >= cutoff

        return SelectionResult(
            selected_mask=selected_mask,
            axis_reduced=axis_arr[selected_mask],
            axis_full=axis_arr,
            vip_scores_full=vip,
            method="vip",
            metadata={
                "threshold": self.threshold,
                "n_selected": int(selected_mask.sum()),
                "n_total": len(selected_mask),
            },
        )
