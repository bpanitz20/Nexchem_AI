#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:43:18 2025

@author: bp
"""
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV

def KFold_CV(x, y, model, param_name, param_range, n_folds=10, groups=None):
    """
    Generic Venetian Blind Cross-Validation
    
    Parameters:
    -----------
    x : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model : sklearn estimator
        Model object to evaluate
    param_name : str
        Name of parameter to optimize (e.g., 'n_components', 'alpha')
    param_range : list
        Range of parameter values to test
    n_folds : int
        Number of cross-validation folds
        
    Returns:
    --------
    dict with:
        - mean_r2_CV: list of mean R2 scores for each parameter value
        - mean_mse_CV: list of mean MSE scores for each parameter value
        - optimal_param: best parameter value
        - Y_pred_CV: cross-validated predictions for optimal parameter
    """
    # Mean center y
    y_mean = np.mean(y, axis=0)
    y_centered = y - y_mean
    
    # Initialize CV strategy
    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        #print("Using GroupKFold CV")
    else:
        cv = KFold(n_splits=n_folds, shuffle=False, random_state=None)
        #print("Using standard KFold CV")
    
    # Initialize KFold and scorers
    r2_scorer = make_scorer(r2_score)
    mse_scorer = make_scorer(mean_squared_error)
    
    # Initialize results storage
    mean_r2_CV = []
    mean_mse_CV = []
    all_predictions = {}
    
    for param_value in param_range:
       # Set parameter value
       model.set_params(**{param_name: param_value})
       
       # Get cross-validated scores
       r2_scores = cross_val_score(model, x, y_centered, cv=cv, scoring=r2_scorer)
       mse_scores = cross_val_score(model, x, y_centered, cv=cv, scoring=mse_scorer)
       
       # Store mean scores
       mean_r2_CV.append(np.mean(r2_scores))
       mean_mse_CV.append(np.mean(mse_scores))
       
       # Get cross-validated predictions
       all_predictions[param_value] = cross_val_predict(model, x, y_centered, cv=cv)
   
    # Find optimal parameter
    optimal_param = param_range[np.argmax(mean_r2_CV)]
    Y_pred_CV = all_predictions[optimal_param]
   
    return {
      'mean_r2_CV': mean_r2_CV,
      'mean_mse_CV': mean_mse_CV,
      'optimal_param': optimal_param,
      'Y_pred_CV': Y_pred_CV,
      'y_mean': y_mean
  }



def KFold_Gridsearch_CV(x, y, model, param_grid, task="regression", n_folds=5, groups=None, scoring=None):
    """
    Generic Grid Search with CV for both regression and classification.

    Parameters:
    -----------
    x : np.ndarray
        Feature matrix
    y : np.ndarray
        Target vector
    model : sklearn estimator
        Model object to evaluate
    param_grid : dict
        Dictionary of parameters to optimize
    task : str
        'regression' or 'classification'
    n_folds : int
        Number of cross-validation folds
    groups : array-like, optional
        Group labels for GroupKFold
    scoring : str or callable
        Scoring metric ('r2', 'neg_mean_squared_error', 'accuracy', etc.)

    Returns:
    --------
    dict with:
        - best_params: dictionary of best parameters
        - best_score: best cross-validation score
        - cv_results: full CV results
        - Y_pred_CV: cross-validated predictions for best model
        - Y_proba_CV: cross-validated probabilities (classification only)
        - best_estimator: fitted model with best parameters
    """
    y_centered = y
    y_mean = None

    if task == 'regression':
        # Mean center y
        y_mean = np.mean(y, axis=0)
        y_centered = y - y_mean

    # CV splitter
    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
    else:
        cv = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    # Grid search setup
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=-1,
        return_train_score=False
    )

    # Fit model
    grid_search.fit(x, y_centered)

    # Get CV predictions
    if task == 'regression':
        Y_pred_CV = cross_val_predict(grid_search.best_estimator_, x, y_centered, cv=cv, method='predict')
        Y_proba_CV = None
    else:  # classification
        Y_pred_CV = cross_val_predict(grid_search.best_estimator_, x, y, cv=cv, method='predict')
        Y_proba_CV = None
        if hasattr(grid_search.best_estimator_, "predict_proba"):
            Y_proba_CV = cross_val_predict(grid_search.best_estimator_, x, y, cv=cv, method='predict_proba')

    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'Y_pred_CV': Y_pred_CV,
        'Y_proba_CV': Y_proba_CV,
        'y_mean': y_mean,
        'best_estimator': grid_search.best_estimator_
    }
