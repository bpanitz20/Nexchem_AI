#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bottom-up greedy block variable selection for PLS.

Algorithm
---------
Phase 1 : Score every contiguous block individually by RMSECV.
Phase 2 : Greedily add the next best block as long as RMSECV improves.
Stop    : When no remaining block lowers the current RMSECV.

The returned ``selected_mask`` is a boolean array of shape (n_features,)
that plugs directly into the existing VIP-style ``selected_mask`` pipeline
in ``models/wrappers.py`` and ``models/prediction_eval.py``.

Notes
-----
* ``y_centered`` passed to ``select_blocks`` must already be mean-centred
  (subtract the training-set mean computed by ``KFold_CV``).  This matches
  the convention used inside ``KFold_CV``.
* CV splitting mirrors the main PLS pipeline: same KFold / GroupKFold
  choice, same fold count, same group labels.
* ``cross_val_predict`` only — no sweep, no print output, no file I/O.
"""

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict
from sklearn.metrics import mean_squared_error


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_blocks(n_features: int, block_size: int) -> list:
    """Return list of (start, end) index pairs for contiguous blocks.

    The final block is a partial block when n_features % block_size != 0.
    """
    blocks = []
    start = 0
    while start < n_features:
        end = min(start + block_size, n_features)
        blocks.append((start, end))
        start = end
    return blocks


def _build_mask(indices: list, blocks: list, n_features: int) -> np.ndarray:
    """Boolean mask of shape (n_features,) covering the given block indices."""
    mask = np.zeros(n_features, dtype=bool)
    for i in indices:
        s, e = blocks[i]
        mask[s:e] = True
    return mask


def _rmsecv_for_block_set(
    X_sub: np.ndarray,
    y_centered: np.ndarray,
    n_components: int,
    cv,
    cv_kwargs: dict,
) -> float:
    """Single RMSECV score for a variable subset.

    Uses ``cross_val_predict`` once with a fixed n_components.  n_components
    is clamped so it never exceeds the block width or the training-set size.
    Returns ``np.inf`` on any error (e.g., degenerate block).
    """
    n_comp = min(n_components, X_sub.shape[1], X_sub.shape[0] - 1)
    if n_comp < 1:
        return np.inf
    try:
        model = PLSRegression(n_components=n_comp)
        yhat = cross_val_predict(model, X_sub, y_centered, cv=cv, **cv_kwargs)
        return float(np.sqrt(mean_squared_error(y_centered, yhat)))
    except Exception:
        return np.inf


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def select_blocks(
    x: np.ndarray,
    y_centered: np.ndarray,
    n_components: int = 5,
    block_size: int = 100,
    n_folds: int = 8,
    groups=None,
) -> dict:
    """Bottom-up greedy block variable selection.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y_centered : np.ndarray, shape (n_samples,) or (n_samples, 1)
        Target vector already mean-centred (``y - y_mean`` from Stage 1
        KFold_CV).
    n_components : int
        Fixed PLS component count used for block scoring.  Clamped per
        block to avoid errors when a block is narrower than this value.
    block_size : int
        Number of contiguous spectral variables per block.
    n_folds : int
        CV fold count — mirrors the main PLS pipeline setting.
    groups : array-like or None
        Group labels for GroupKFold.  ``None`` uses standard KFold.

    Returns
    -------
    dict
        selected_mask           : np.ndarray[bool], shape (n_features,)
        selected_block_indices  : list[int]  (sorted)
        block_scores            : list[float]  Phase-1 individual RMSECV per block
        n_blocks_total          : int
        n_blocks_selected       : int
        block_size_used         : int
        block_scoring_n_components : int
    """
    n_samples, n_features = x.shape
    y_flat = np.array(y_centered).ravel()

    # Build CV splitter — exactly mirrors KFold_CV logic
    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        cv_kwargs = {"groups": groups}
    else:
        cv = KFold(n_splits=n_folds, shuffle=False)
        cv_kwargs = {}

    blocks = _build_blocks(n_features, block_size)
    n_blocks = len(blocks)

    # Guard: only one block — return full mask without any selection
    if n_blocks <= 1:
        return {
            "selected_mask": np.ones(n_features, dtype=bool),
            "selected_block_indices": list(range(n_blocks)),
            "block_scores": [np.inf],
            "n_blocks_total": n_blocks,
            "n_blocks_selected": n_blocks,
            "block_size_used": block_size,
            "block_scoring_n_components": n_components,
        }

    # ── Phase 1: score every block individually ───────────────────────────
    block_scores = []
    for start, end in blocks:
        score = _rmsecv_for_block_set(
            x[:, start:end], y_flat, n_components, cv, cv_kwargs
        )
        block_scores.append(score)

    # Start with the single best-scoring block
    best_single = int(np.argmin(block_scores))
    selected_indices = [best_single]

    current_mask = _build_mask(selected_indices, blocks, n_features)
    current_score = _rmsecv_for_block_set(
        x[:, current_mask], y_flat, n_components, cv, cv_kwargs
    )

    # ── Phase 2: greedily add blocks while RMSECV improves ────────────────
    remaining = [i for i in range(n_blocks) if i not in selected_indices]
    improved = True
    while improved and remaining:
        improved = False
        candidate_scores = []
        for i in remaining:
            trial_mask = _build_mask(selected_indices + [i], blocks, n_features)
            score = _rmsecv_for_block_set(
                x[:, trial_mask], y_flat, n_components, cv, cv_kwargs
            )
            candidate_scores.append((score, i))

        best_score, best_idx = min(candidate_scores, key=lambda t: t[0])
        if best_score < current_score:
            selected_indices.append(best_idx)
            remaining.remove(best_idx)
            current_score = best_score
            improved = True

    selected_mask = _build_mask(selected_indices, blocks, n_features)

    return {
        "selected_mask": selected_mask,
        "selected_block_indices": sorted(selected_indices),
        "block_scores": block_scores,
        "n_blocks_total": n_blocks,
        "n_blocks_selected": len(selected_indices),
        "block_size_used": block_size,
        "block_scoring_n_components": n_components,
    }
