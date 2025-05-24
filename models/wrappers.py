#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Wrappers
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from models.cross_val import KFold_CV
from models.cross_val import KFold_Gridsearch_CV
from plotting.plot_regression import (
    plot_pred_vs_actual,
    plot_coefficients,
    plot_feature_importance,
    plot_vip_scores,
    print_model_summary
) 

def PLS_model(x, y, directory, axis, max_lv=10, analyte=""):
    param_name = 'n_components'
    param_range = list(range(1, max_lv + 1))
    model = PLSRegression()

    # Run CV
    cv_results = KFold_CV(x, y, model, param_name, param_range, analyte=analyte, model_name='PLS', directory=directory)

    # Final fit with optimal parameter
    model.set_params(**{param_name: cv_results['optimal_param']})
    model.fit(x, y - cv_results['y_mean'])
    Y_pred = model.predict(x)

    # Metrics
    final_r2 = r2_score(y - cv_results['y_mean'], Y_pred)
    final_mse = mean_squared_error(y - cv_results['y_mean'], Y_pred)
    final_r2_CV = cv_results['mean_r2_CV'][param_range.index(cv_results['optimal_param'])]
    final_mse_CV = cv_results['mean_mse_CV'][param_range.index(cv_results['optimal_param'])]

    # Print summary
    print_model_summary(
        model_name="PLS",
        analyte=analyte,
        final_r2=final_r2,
        final_r2_CV=final_r2_CV,
        final_mse=final_mse,
        final_mse_CV=final_mse_CV,
        optimal_param=cv_results['optimal_param'],
        param_name=param_name
    )

    # Plots
    plot_pred_vs_actual(
        y,
        Y_pred + cv_results['y_mean'],
        directory,
        f"Final Predicted vs. Actual for {analyte} (PLS)",
        f"Final_Pred_vs_Actual_PLS_{analyte}.png"
    )
    plot_coefficients(axis, model.coef_, directory, "PLS", analyte)
    plot_vip_scores(model, x, axis, directory, "PLS", analyte)

    return {
        'model': model,
        'final_r2': final_r2,
        'final_mse': final_mse,
        'cv_results': cv_results
    }

def MLPRegressor_model(x, y, directory, axis, analyte="", param_grid=None, random_state=42):
    y = np.array(y).ravel()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'activation': ['relu'],
            'alpha': [0.02, 0.01, 0.0009],
            'learning_rate_init': np.linspace(0.0001, 0.01, 10).tolist(),
            'early_stopping': [True],
            'solver': ['adam']
        }

    base_model = MLPRegressor(
        max_iter=2000,
        random_state=random_state,
        verbose=False,
        n_iter_no_change=20,
        tol=1e-4,
        batch_size='auto'
    )

    cv_results = KFold_Gridsearch_CV(
        x=X_scaled,
        y=y,
        model=base_model,
        param_grid=param_grid,
        task="regression",
        analyte=analyte,
        model_name="MLP",
        directory=directory
    )

    final_model = cv_results['best_estimator']
    Y_pred = final_model.predict(X_scaled) + cv_results['y_mean']

    final_r2 = r2_score(y, Y_pred)
    final_mse = mean_squared_error(y, Y_pred)

    Y_pred_CV = cv_results['Y_pred_CV'] + cv_results['y_mean']
    final_r2_CV = r2_score(y, Y_pred_CV)
    final_mse_CV = mean_squared_error(y, Y_pred_CV)

    print_model_summary(
        model_name="MLP",
        analyte=analyte,
        final_r2=final_r2,
        final_r2_CV=final_r2_CV,
        final_mse=final_mse,
        final_mse_CV=final_mse_CV,
        best_params=cv_results['best_params']
        )
    
    plot_feature_importance(
    model=final_model,
    x=X_scaled,
    y=y,
    axis=axis,
    directory=directory,
    model_name="MLP",
    analyte=analyte
    )
    
    plot_pred_vs_actual(
        y,
        Y_pred,
        directory,
        f"Final Predicted vs. Actual for {analyte} (MLP)",
        f"Final_Pred_vs_Actual_MLP_{analyte}.png"
    )

    return {
        'model': final_model,
        'final_r2': final_r2,
        'final_mse': final_mse,
        'cv_results': cv_results['cv_results'],
        'best_params': cv_results['best_params']
    }


"""
Classification Model Wrppers
"""

def MLPClassifier_model(x, y, directory, axis, analyte="", param_grid=None, random_state=42):
    """
    Wrapper for MLPClassifier using evaluate_classifier()

    Parameters:
    -----------
    x : np.ndarray
        Feature matrix
    y : array-like
        Class labels (already binned/coded)
    directory : str
        Output directory to save results
    axis : list
        Spectral axis
    analyte : str
        Target name for labeling
    param_grid : dict, optional
        Grid search parameters
    random_state : int
        Random seed
    """

    # Ensure 1D label array
    y = np.array(y).ravel()

    # Scale X for neural network
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    # Default hyperparameter grid
    if param_grid is None:
        param_grid = {
            'alpha': [0.01, 0.001],
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'learning_rate_init': [0.001],
            'early_stopping': [True]
        }

    # Base model
    base_model = MLPClassifier(
        max_iter=2000,
        random_state=random_state,
        tol=1e-4,
        verbose=False
    )
"""
    return evaluate_classifier(
        x=X_scaled,
        y=y,
        directory=directory,
        axis=axis,
        model=base_model,
        model_name='MLPClassifier',
        analyte=analyte,
        param_grid=param_grid
    )
"""