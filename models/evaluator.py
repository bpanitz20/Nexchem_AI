#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluator Module for Regression Models

This module provides a flexible `evaluate_model` function for training,
cross-validating, and assessing regression models. It supports both
single-parameter sweeps and full grid searches. Key performance metrics
and plots are automatically generated and saved.

Author: Ben Panitz
Created: April 26, 2025
"""
import os
import pandas as pd
import numpy as np
from models.cross_val import KFold_CV, KFold_Gridsearch_CV
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from plotting.plot_regression import (
    plot_pred_vs_actual,
    plot_cv_performance,
    plot_coefficients,
    plot_feature_importance,
    plot_vip_scores
) 
from plotting.plot_classifier import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_decision_boundary,

)

def evaluate_regression_model(x, y, directory, axis, model, model_name, analyte, 
                  param_name=None, param_range=None, param_grid=None, 
                  cv=KFold_CV):
    """
    Enhanced regression model evaluation function that handles both single-parameter optimization
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
        cv_results = KFold_Gridsearch_CV(
            x=x,
            y=y,
            model=model,
            param_grid=param_grid,
            task="regression")
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
      plot_cv_performance(
      param_range, 
      cv_results['mean_r2_CV'], 
      cv_results['mean_mse_CV'], 
      param_name, 
      analyte, 
      model_name, 
      directory
  )
    else:
        # Save parameter search results as DataFrame
        cv_df = pd.DataFrame(cv_results['cv_results'])
        cv_df.to_csv(os.path.join(directory, f'GridSearch_results_{model_name}_{analyte}.csv'), index=False)
    
    # Plot regression coefficients if available
    if hasattr(final_model, 'coef_'):
        plot_coefficients(axis, final_model.coef_, directory, model_name, analyte)
        if model_name.lower() == 'pls':
            plot_vip_scores(final_model, x, axis, directory, model_name, analyte)
    elif any(m in model_name.lower() for m in ['mlp','gbr', 'svm', 'random forest', 'knn']):
        plot_feature_importance(final_model, x, y - cv_results['y_mean'], axis, directory, model_name, analyte)
    
   # Final model predictions
    plot_pred_vs_actual(
        y, 
        Y_pred + cv_results['y_mean'], 
        directory,
        f'Predicted vs. Actual for {analyte} ({model_name})',
        f'Pred_vs_Actual_{model_name}_{analyte}.png'
    )
    
    # CV predictions
    plot_pred_vs_actual(
        y, 
        cv_results['Y_pred_CV'] + cv_results['y_mean'], 
        directory,
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


def evaluate_classifier(x, y, directory, axis, model, model_name, analyte,
                        param_grid, cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=42)):
    """
    Evaluates a classifier with grid search CV. Outputs metrics, confusion matrix, ROC, and decision boundary.
    """
    # Step 1: Run grid search cross-validation
    cv_results = KFold_Gridsearch_CV(x, y, model, param_grid, task="classification", n_folds=cv.n_splits)

    # Step 2: Extract relevant values
    final_model = cv_results["best_estimator"]
    best_params = cv_results["best_params"]
    mean_cv_acc = cv_results["best_score"]
    std_cv_acc = np.std(cv_results["cv_results"]["mean_test_score"])
    y_cv_pred = cv_results["Y_pred_CV"]
    y_cv_proba = cv_results["Y_proba_CV"]

    # Step 3: Predict on full train set
    y_pred = final_model.predict(x)
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')

    # Step 4: Print results
    print(f"\n📊 Classification Metrics for {analyte} ({model_name})")
    print(f"Best Params: {best_params}")
    print(f"Accuracy (Full Train Set): {acc:.3f}")
    print(f"F1 Score (Full Train Set): {f1:.3f}")
    print(f"CV Accuracy (Mean ± SD): {mean_cv_acc:.3f} ± {std_cv_acc:.3f}")

    # Step 5: Plot final model outputs
    plot_confusion_matrix(y_pred, y, directory, model_name, analyte)
    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(x)
        plot_roc_curve(y, y_proba, directory, model_name, analyte)
    plot_decision_boundary(x, y, final_model, directory, model_name, analyte)

    # Step 6: Plot CV outputs
    plot_confusion_matrix(y_cv_pred, y, directory, model_name, analyte, suffix="_CV")
    if y_cv_proba is not None:
        plot_roc_curve(y, y_cv_proba, directory, model_name, analyte, suffix="_CV")

    # Step 7: Return all results
    return {
        'accuracy': acc,
        'f1': f1,
        'cv_accuracy_mean': mean_cv_acc,
        'cv_accuracy_std': std_cv_acc,
        'model': final_model,
        'best_params': best_params,
        'y_true': y,
        'y_pred': y_pred,
        'y_pred_cv': y_cv_pred
    }
