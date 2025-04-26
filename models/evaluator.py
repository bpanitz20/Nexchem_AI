#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:43:18 2025

@author: bp
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from models.cross_val import KFold_CV, KFold_Gridsearch_CV

from sklearn.metrics import r2_score, mean_squared_error

def evaluate_model(x, y, directory, axis, model, model_name, analyte, 
                  param_name=None, param_range=None, param_grid=None, 
                  cv=KFold_CV):
    """
    Enhanced model evaluation function that handles both single-parameter optimization
    and grid search, with consistent plotting outputs.
    
    Parameters:
    -----------
    x : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    directory : str
        Directory to save results
    axis : list
        Spectral axis for plotting
    model : sklearn estimator
        Model to evaluate
    model_name : str
        Name of model (for plots)
    analyte : str
        Name of analyte (for plots)
    param_name : str, optional
        Name of single parameter to optimize
    param_range : list, optional
        Range of parameter values to test (single parameter)
    param_grid : dict, optional
        Parameter grid for grid search
    cv : callable
        Cross-validation function to use
    n_folds : int
        Number of CV folds
        
    Returns:
    --------
    dict with model evaluation results
    """
    # Determine evaluation mode
    if param_grid is not None:
        mode = 'grid_search'
        cv_results = KFold_Gridsearch_CV(x, y, model, param_grid)
    else:
        mode = 'single_param'
        cv_results = KFold_CV(x, y, model, param_name, param_range)
    
    # Common result processing
    if mode == 'single_param':
        # Train final model with optimal parameter
        final_model = model.set_params(**{param_name: cv_results['optimal_param']})
        final_model.fit(x, y - cv_results['y_mean'])
        Y_pred = final_model.predict(x)
        
        # Calculate metrics
        final_r2 = r2_score(y - cv_results['y_mean'], Y_pred)
        final_mse = mean_squared_error(y - cv_results['y_mean'], Y_pred)
        optimal_idx = param_range.index(cv_results['optimal_param'])
        final_r2_CV = cv_results['mean_r2_CV'][optimal_idx]
        final_mse_CV = cv_results['mean_mse_CV'][optimal_idx]
    else:  # grid_search
        final_model = cv_results['best_estimator']
        Y_pred = final_model.predict(x)
        
        # Calculate metrics
        final_r2 = r2_score(y - cv_results['y_mean'], Y_pred)
        final_mse = mean_squared_error(y - cv_results['y_mean'], Y_pred)
        final_r2_CV = cv_results['best_score']
        final_mse_CV = -cv_results['best_score']  # Assuming scoring was 'neg_mean_squared_error'
    
    # Print metrics
    print(f"\nFinal Model Metrics for {analyte} ({model_name}):")
    if mode == 'single_param':
        print(f"Optimal {param_name}: {cv_results['optimal_param']}")
    else:
        print("Best parameters:", cv_results['best_params'])
    print(f"R²_Cal: {final_r2:.4f}")
    print(f"R²_CV: {final_r2_CV:.4f}")
    print(f"MSE: {final_mse:.4f}")
    print(f"MSECV: {final_mse_CV:.4f}")
    
    # Plotting for single parameter optimization
    if mode == 'single_param':
        # Parameter vs R² plot
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, cv_results['mean_r2_CV'], marker='o', linestyle='-', color='b')
        plt.xlabel(param_name)
        plt.ylabel('R² Score CV')
        plt.title(f'R² Score CV vs. {param_name} for {analyte} ({model_name})')
        plt.grid(True)
        plt.savefig(os.path.join(directory, f'R²_vs_{param_name}_{model_name}_{analyte}.png'), 
                    dpi=300, bbox_inches="tight")
        plt.close()
        
        # Parameter vs MSE plot
        plt.figure(figsize=(10, 6))
        plt.plot(param_range, cv_results['mean_mse_CV'], marker='o', linestyle='-', color='r')
        plt.xlabel(param_name)
        plt.ylabel('Mean Squared Error CV (MSECV)')
        plt.title(f'MSECV vs. {param_name} for {analyte} ({model_name})')
        plt.grid(True)
        plt.savefig(os.path.join(directory, f'MSE_vs_{param_name}_{model_name}_{analyte}.png'), 
                    dpi=300, bbox_inches="tight")
        plt.close()
    else:
        # Save parameter search results as DataFrame
        cv_df = pd.DataFrame(cv_results['cv_results'])
        cv_df.to_csv(os.path.join(directory, f'GridSearch_results_{model_name}_{analyte}.csv'), index=False)
    
    # Plot regression coefficients if available
    if hasattr(final_model, 'coef_'):
        plt.figure(figsize=(10, 6))
        plt.plot(axis, final_model.coef_.flatten(), color='blue')
        plt.xlabel('Features')
        plt.ylabel('Regression Coefficients')
        plt.title(f'Regression Coefficients ({model_name})')
        plt.grid(True)
        plt.savefig(os.path.join(directory, f'Coefficients_{model_name}_{analyte}.png'), 
                    dpi=300, bbox_inches="tight")
        plt.close()
    
    # Common plotting - Predicted vs Actual (always make these plots)
    def plot_pred_vs_actual(y_true, y_pred, title, filename):
        plt.figure(figsize=(8, 8))
        plt.scatter(y_true, y_pred, color='blue', alpha=0.6, label='Data')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 
                 color='red', linestyle='--', label='Ideal')
        slope, intercept = np.polyfit(y_true.ravel(), y_pred.ravel(), 1)
        plt.plot(y_true, slope*y_true + intercept, 'g--', label='Best Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(title)
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(directory, filename), dpi=300, bbox_inches="tight")
        plt.show()
    
    # Final model predictions
    plot_pred_vs_actual(
        y, 
        Y_pred + cv_results['y_mean'], 
        f'Predicted vs. Actual for {analyte} ({model_name})',
        f'Pred_vs_Actual_{model_name}_{analyte}.png'
    )
    
    # CV predictions
    plot_pred_vs_actual(
        y, 
        cv_results['Y_pred_CV'] + cv_results['y_mean'], 
        f'CV Predicted vs. Actual for {analyte} ({model_name})',
        f'CV_Pred_vs_Actual_{model_name}_{analyte}.png'
    )
    
    return {
        'final_r2': final_r2,
        'final_mse': final_mse,
        'optimal_param': cv_results.get('optimal_param', None),
        'best_params': cv_results.get('best_params', None),
        'final_r2_CV': final_r2_CV,
        'final_mse_CV': final_mse_CV,
        'model': final_model,
        'cv_results': cv_results
    }
