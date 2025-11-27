#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 27 08:38:43 2025

@author: bp
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_spectra_colored_by_analyte(
        sample_spectra,
        y_df,
        analyte_col,
        id_col="ID",
        use_first_replicate=False,
        cmap_name="viridis",
        vmin=None,
        vmax=None,
        title=None):
    """
    Create an overlay plot of spectra colored by analyte value.

    Parameters
    ----------
    sample_spectra : dict
        {ID: Spectrum or [Spectrum, ...]}.
    y_df : pandas.DataFrame
        Y-block with at least [id_col, analyte_col] columns.
        For you: columns = ["ID", "Class", "EPA+DHA", "PUFA", ...]
    analyte_col : str
        Column to color by, e.g. "EPA+DHA" or "PUFA".
    id_col : str, default "ID"
        Column in y_df holding the sample IDs matching sample_spectra keys.
    use_first_replicate : bool, default False
        If True, only the first spectrum per ID is plotted.
    cmap_name : str, default "viridis"
        Matplotlib colormap.
    vmin, vmax : float or None
        Color scale limits; if None, inferred from data.
    title : str or None
        Plot title.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    # Ensure IDs are the index
    if id_col in y_df.columns:
        y_idx = y_df.set_index(id_col)
    else:
        y_idx = y_df  # assume already indexed by ID

    if analyte_col not in y_idx.columns:
        raise ValueError(f"analyte_col '{analyte_col}' not found in y-block columns: {list(y_idx.columns)}")

    # IDs that exist in BOTH spectra and Y-block
    common_ids = [sid for sid in sample_spectra.keys() if sid in y_idx.index]
    if not common_ids:
        raise ValueError("No overlapping IDs between sample_spectra keys and Y-block IDs.")

    analyte_vals = y_idx.loc[common_ids, analyte_col].astype(float).values

    if vmin is None:
        vmin = np.nanmin(analyte_vals)
    if vmax is None:
        vmax = np.nanmax(analyte_vals)

    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    cmap = plt.get_cmap(cmap_name)

    fig, ax = plt.subplots(figsize=(8, 5))

    for sid, bundle in sample_spectra.items():
        if sid not in y_idx.index:
            continue

        y_val = float(y_idx.loc[sid, analyte_col])
        color = cmap(norm(y_val))

        # Normalize bundle to list of spectra (raw or preprocessed)
        if isinstance(bundle, list):
            spectra_list = bundle
        else:
            spectra_list = [bundle]

        if use_first_replicate:
            spectra_list = spectra_list[:1]

        for sp in spectra_list:
            ax.plot(sp.spectral_axis, sp.spectral_data, color=color, alpha=0.7)

    ax.set_xlabel("Raman Shift (cm⁻¹)")
    ax.set_ylabel("Intensity (a.u.)")
    if title is None:
        title = f"Overlay Colored by {analyte_col}"
    ax.set_title(title)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax)
    cbar.set_label(analyte_col)

    fig.tight_layout()
    return fig, ax
