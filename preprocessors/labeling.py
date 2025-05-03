#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  3 11:38:50 2025

@author: bp
"""

import pandas as pd


def bin_targets(y, bins, labels=None, return_codes=True):
    """
    Bin continuous numerical targets into class labels or integer codes.

    Parameters:
    -----------
    y : array-like
        Numeric values (e.g., hatch rates)
    bins : list of float
        Bin edges (e.g., [0, 40, 70, 100])
    labels : list of str, optional
        Labels like ['Low', 'Mid', 'High']
    return_codes : bool
        If True, return integer codes (0, 1, 2); otherwise return categorical labels

    Returns:
    --------
    y_binned : np.ndarray
        Integer codes or class labels
    """
    if labels is None:
        labels = [f"Class_{i+1}" for i in range(len(bins)-1)]

    y_binned = pd.cut(y, bins=bins, labels=labels, include_lowest=True)

    if return_codes:
        return y_binned.codes  # returns integers
    else:
        return y_binned        # returns pandas.Categorical
