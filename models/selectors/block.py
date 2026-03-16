#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BlockSelector — thin façade around ``models/block_selection.py``.

The selection algorithm lives entirely in ``block_selection.select_blocks``.
This class provides a clean, uniform interface that:
  * accepts configuration at construction time (block_size, n_components),
  * exposes a single ``fit()`` method that returns a ``SelectionResult``,
  * lets ``_pls_compute`` swap in any other Selector without touching the
    rest of the pipeline.

Usage in _pls_compute
---------------------
    selector = BlockSelector(block_size=100, n_components=5)
    selection = selector.fit(x, y_centered, axis=axis_arr,
                             n_folds=n_folds, groups=groups,
                             vip_scores_full=vip_scores_full)
    x = x[:, selection.selected_mask]
"""

import numpy as np
from models.selectors.base import SelectionResult
from models.block_selection import select_blocks


class BlockSelector:
    """Bottom-up greedy block variable selection.

    Parameters
    ----------
    block_size : int
        Number of contiguous spectral variables per block.  Default 100.
    n_components : int
        Fixed PLS component count used when scoring candidate block sets.
        Clamped automatically when a block is narrower than this value.
        Default 5.
    """

    def __init__(self, block_size: int = 100, n_components: int = 5):
        self.block_size   = block_size
        self.n_components = n_components

    def fit(
        self,
        x: np.ndarray,
        y_centered: np.ndarray,
        axis: np.ndarray,
        n_folds: int = 8,
        groups=None,
        vip_scores_full: np.ndarray | None = None,
    ) -> SelectionResult:
        """Run block selection and return a ``SelectionResult``.

        Parameters
        ----------
        x : np.ndarray, shape (n_samples, n_features)
            Feature matrix.
        y_centered : np.ndarray
            Mean-centred target vector (``y - y_mean`` from Stage 1 CV).
        axis : np.ndarray
            Full spectral axis, shape (n_features,).
        n_folds : int
            CV fold count — must match the main PLS pipeline setting.
        groups : array-like or None
            Group labels for GroupKFold; ``None`` uses standard KFold.
        vip_scores_full : np.ndarray or None
            Pre-computed full-spectrum VIP scores from the preliminary model.
            Forwarded into ``SelectionResult`` for diagnostic plotting.

        Returns
        -------
        SelectionResult
        """
        axis_arr = np.asarray(axis)

        raw = select_blocks(
            x, y_centered,
            n_components=self.n_components,
            block_size=self.block_size,
            n_folds=n_folds,
            groups=groups,
        )

        return SelectionResult(
            selected_mask   = raw["selected_mask"],
            axis_reduced    = axis_arr[raw["selected_mask"]],
            axis_full       = axis_arr,
            vip_scores_full = vip_scores_full,
            method          = "block",
            metadata        = {
                "selected_block_indices":     raw["selected_block_indices"],
                "block_scores":               raw["block_scores"],
                "block_size_used":            raw["block_size_used"],
                "block_scoring_n_components": raw["block_scoring_n_components"],
                "n_blocks_total":             raw["n_blocks_total"],
                "n_blocks_selected":          raw["n_blocks_selected"],
            },
        )
