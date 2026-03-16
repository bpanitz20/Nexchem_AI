#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Variable-selection result container.

``SelectionResult`` is the single, uniform output type for every variable-
selection method in NexChem.  All consumers (wrappers.py, plot_pls_results,
streamlit_app.py, prediction_eval.py) read from this object rather than from
a collection of flat dict keys.

Adding a new selection method requires only:
  1. A new Selector class that returns a ``SelectionResult`` from its fit()
     method.
  2. A hook in ``_pls_compute`` that calls that class.
No downstream file needs to change.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class SelectionResult:
    """Uniform output container for all variable-selection methods.

    Attributes
    ----------
    selected_mask : np.ndarray[bool], shape (n_features,)
        True for every feature retained by the selection step.
    axis_reduced : np.ndarray
        Spectral axis values corresponding to the retained features.
    axis_full : np.ndarray
        Original full spectral axis before any reduction.
    vip_scores_full : np.ndarray or None
        VIP scores computed from the preliminary full-X PLS model.
        Stored here so ``plot_pls_results`` can draw the VIP plot on the
        full axis without gap artefacts from the reduced feature space.
        ``None`` when the selection method does not compute a preliminary model.
    method : str
        Short identifier for the selection algorithm, e.g. ``"block"``.
    metadata : dict
        Method-specific auxiliary information (block indices, scores, etc.).
        Consumers should treat this as read-only.

    Properties
    ----------
    n_selected : int
        Number of features retained (derived from selected_mask).
    n_total_features : int
        Total number of features before selection (derived from axis_full).
    """

    selected_mask:   np.ndarray
    axis_reduced:    np.ndarray
    axis_full:       np.ndarray
    vip_scores_full: np.ndarray | None
    method:          str
    metadata:        dict = field(default_factory=dict)

    @property
    def n_selected(self) -> int:
        """Number of features retained after selection."""
        return int(self.selected_mask.sum())

    @property
    def n_total_features(self) -> int:
        """Total number of features before selection."""
        return int(self.axis_full.shape[0])
