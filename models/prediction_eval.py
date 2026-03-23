#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 17:07:47 2025

@author: bp
"""

# models/prediction_eval.py

import numpy as np
import pandas as pd
import os
from sklearn.metrics import r2_score, mean_squared_error
from plotting.plot_regression import plot_pred_vs_actual

def evaluate_on_prediction_set(
    model,
    X_pred,
    y_mean=None,
    axis=None,
    analyte="",
    directory="",
    Y_pred_true=None,
    model_name="Model",
    sample_ids=None,
    selected_mask=None,
):
    """
    Evaluate trained model on external prediction set.

    Parameters:
    -----------
    model : sklearn estimator
        Trained regression model
    X_pred : np.ndarray
        Prediction feature matrix
    y_mean : float or np.ndarray
        Mean of Y from training (for mean-centering recovery)
    axis : list or np.ndarray
        Spectral axis (optional; not used yet)
    analyte : str
        Target name (e.g., 'DHA')
    directory : str
        Where to save results
    Y_pred_true : np.ndarray or None
        True Y values for prediction set, if available
    model_name : str
        Name of model (e.g., 'PLS', 'MLP')
    selected_mask : np.ndarray of bool or None
        Feature mask from VIP variable selection.  When provided, X_pred is
        reduced to the same variable subset used during training before
        prediction.  None (default) leaves X_pred unchanged.
    """
    # Apply VIP feature mask before prediction
    if selected_mask is not None:
        assert X_pred.shape[1] == len(selected_mask), (
            f"Prediction feature count ({X_pred.shape[1]}) does not match "
            f"training mask length ({len(selected_mask)}). "
            "Ensure prediction data uses the same preprocessing as training."
        )
        X_pred = X_pred[:, selected_mask]

    # Step 1: Predict
    Y_pred = model.predict(X_pred)
    if y_mean is not None:
        Y_pred += y_mean

    # Step 2: Save CSV
    result_dict = {
    "Sample ID": sample_ids if sample_ids is not None else [f"sample_{i+1}" for i in range(len(Y_pred))],
    "Predicted": Y_pred.ravel()
        }
    if Y_pred_true is not None:
        result_dict["Actual"] = Y_pred_true.ravel()

    pred_df = pd.DataFrame(result_dict)
    csv_path = os.path.join(directory, f"Prediction_Scores_For_{model_name}_{analyte}.csv")
    pred_df.to_csv(csv_path, index=False)
    #print(f"✅ External predictions saved to: {csv_path}")

    # Step 3: Metrics + Plot
    if Y_pred_true is not None:
        r2_pred = r2_score(Y_pred_true, Y_pred)
        rmsep = np.sqrt(mean_squared_error(Y_pred_true, Y_pred))
        
        print(f"\n📈 External Prediction Metrics for {analyte} ({model_name}):")
        print(f"R²_pred  = {r2_pred:.4f}")
        print(f"RMSEP    = {rmsep:.4f}")

        plot_pred_vs_actual(
            y_true=Y_pred_true,
            y_pred=Y_pred,
            directory=directory,
            title=f"External Predicted vs. Actual for {analyte} ({model_name})",
            filename=f"External_Pred_vs_Actual_{model_name}_{analyte}.png"
        )
    if Y_pred_true is not None:
        return {
            "analyte": analyte,
            "model_name": model_name,
            "r2_pred": r2_pred,
            "rmsep": rmsep,
            "pred_plot_path": os.path.join(directory, f"External_Pred_vs_Actual_{model_name}_{analyte}.png"),
            "csv_path": csv_path,
            # Raw arrays for interactive plots in the Streamlit app
            "y_true": np.asarray(Y_pred_true).ravel(),
            "y_pred": np.asarray(Y_pred).ravel(),
            "sample_ids": sample_ids,
        }
    else:
        return {
            "analyte": analyte,
            "model_name": model_name,
            "csv_path": csv_path,
            "y_pred": np.asarray(Y_pred).ravel(),
            "sample_ids": sample_ids,
        }
