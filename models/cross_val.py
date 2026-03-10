#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 18:43:18 2025

@author: bp
"""
import numpy as np
from sklearn.model_selection import KFold, GroupKFold, cross_val_score, cross_val_predict
from sklearn.metrics import mean_squared_error, root_mean_squared_error, r2_score, make_scorer
from sklearn.model_selection import GridSearchCV
import pandas as pd
import os

def KFold_CV(x, y, model, param_name, 
             param_range, n_folds=8, groups=None, 
             analyte="", model_name="", 
             directory="", manual_param=None, sample_ids=None, class_labels=None):
    
    """
    Kfold Cross-Validation
    
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
        cv_kwargs = {"groups": groups}
        print("Using GroupKFold CV")
    else:
        cv = KFold(n_splits=n_folds, shuffle=False, random_state=None)
        cv_kwargs = {}
        print("\nUsing standard KFold CV")
    print(f"n_folds = {n_folds}")
   
    # Initialize KFold and scorers
    r2_scorer = make_scorer(r2_score)
    mse_scorer = make_scorer(mean_squared_error)
    
    # Initialize results storage
    mean_r2_CV = []
    mean_r2_cal = []
    mean_mse_CV = []
    mean_rmse_cal = []
    all_predictions = {}
    pooled_r2_CV = []       
    pooled_rmse_CV = []
    
    # CV fold tracking
    fold_assignments = {} if sample_ids is not None else None

    
    for param_value in param_range:
       # Set parameter value
       model.set_params(**{param_name: param_value}) 
       
       # Get cross-validated scores
       r2_scores = cross_val_score(model, x, y_centered, cv=cv, scoring=r2_scorer, **cv_kwargs)
       mse_scores = cross_val_score(model, x, y_centered, cv=cv, scoring=mse_scorer, **cv_kwargs)      
       mean_r2_CV.append(np.mean(r2_scores))
       mean_mse_CV.append(np.mean(mse_scores))     
       
        # Get cross-validated predictions (existing)
       yhat_cv = cross_val_predict(model, x, y_centered, cv=cv, **cv_kwargs)
       all_predictions[param_value] = yhat_cv 
       
       #pooled metrics from stacked CV predictions (added)
       pooled_r2_CV.append(r2_score(y_centered, yhat_cv))
       pooled_rmse_CV.append(np.sqrt(mean_squared_error(y_centered, yhat_cv)))
        

       # Fit and calculate calibration errors
       model.fit(x, y_centered)
       Y_fit = model.predict(x)
       mse_fit = mean_squared_error(y_centered, Y_fit)
       r2_fit = r2_score(y_centered, Y_fit)
       mean_rmse_cal.append(np.sqrt(mse_fit))
       mean_r2_cal.append(r2_fit) 
   
       # Capture fold assignments
       if param_value == param_range[0] and fold_assignments is not None:
            for fold_idx, (_, test_idx) in enumerate(cv.split(x, y_centered, **cv_kwargs)):
                for i in test_idx:
                    fold_assignments[sample_ids[i]] = fold_idx

    # Build fold assignment DataFrame
    if fold_assignments:
        fold_df = pd.DataFrame(list(fold_assignments.items()), columns=["Sample ID", "CV Fold"])
        fold_df = fold_df.sort_values(by="CV Fold").reset_index(drop=True)
    else:
        fold_df = None

   
    # Select optimal parameter 
    pooled_rmscev = np.array(pooled_rmse_CV)             # already √MSE (RMSECV)
    rmsec          = np.array(mean_rmse_cal)             # already √MSE (RMSEC)
    rmse_gap       = np.abs(rmsec - pooled_rmscev)
    
    if manual_param is not None:
        optimal_param = manual_param
    else:
        optimal_idx   = int(np.argmin(rmse_gap))
        optimal_param = list(param_range)[optimal_idx]

    Y_pred_CV = all_predictions[optimal_param]
    cv_r2_plot_path = os.path.join(directory, f'CV_R2_{model_name}_{analyte}.png')
    cv_rmse_plot_path = os.path.join(directory, f'CV_RMSE_{model_name}_{analyte}.png')
    cv_pred_path = os.path.join(directory, f'CV_Pred_vs_Actual_{model_name}_{analyte}.png')

    return {
      'mean_r2_CV': mean_r2_CV,
      'mean_mse_CV': mean_mse_CV,
      'mean_r2_cal': mean_r2_cal,
      'mean_rmse_cal': mean_rmse_cal,
      'optimal_param': optimal_param,
      'Y_pred_CV': Y_pred_CV,
      'y_mean': y_mean,
      'cv_r2_plot_path': cv_r2_plot_path,
      'cv_rmse_plot_path': cv_rmse_plot_path,
      'cv_pred_plot_path': cv_pred_path,
      'fold_df': fold_df,
      'pooled_r2_CV': pooled_r2_CV,
      'pooled_rmse_CV': pooled_rmse_CV,
  }




def KFold_Gridsearch_CV(x, y, model, param_grid, task="regression", 
                        n_folds=10, groups=None, scoring=None, 
                        analyte="", model_name="", directory="",
                        sample_ids=None, class_labels=None):
   
    y_centered = y
    y_mean = None

    if task == 'regression':
        y_mean = np.mean(y, axis=0)
        y_centered = y - y_mean
  
   # Initialize CV strategy
    if groups is not None:
       cv = GroupKFold(n_splits=n_folds)
       cv_kwargs = {"groups": groups}
       print("✅ Using GroupKFold")
    else:
       cv = KFold(n_splits=n_folds, shuffle=False, random_state=None)
       cv_kwargs = {}
       print("Using standard KFold")
    print(f"n_folds = {n_folds}")
    
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        refit=True,
        n_jobs=1,
        return_train_score=False
    )

    grid_search.fit(x, y_centered, groups=groups)
    
    # Track fold assignment per sample
    fold_assignments = []
    for fold_idx, (_, test_idx) in enumerate(cv.split(x, y, **cv_kwargs)):
        for i in test_idx:
            sample_id = sample_ids[i] if sample_ids is not None else i
            fold_assignments.append((sample_id, fold_idx))
    fold_df = pd.DataFrame(fold_assignments, columns=["Sample ID", "CV Fold"])

    cv_r2_pooled = None
    cv_rmse_pooled = None
    cv_table_df = None
    if task == 'regression':
        Y_pred_CV = cross_val_predict(grid_search.best_estimator_, x, y_centered, cv=cv, method='predict', **cv_kwargs)
        Y_proba_CV = None
        cv_r2_pooled = r2_score(y_centered, Y_pred_CV)
        cv_rmse_pooled = np.sqrt(mean_squared_error(y_centered, Y_pred_CV))
        
    else:
        Y_pred_CV = cross_val_predict(grid_search.best_estimator_, x, y, cv=cv, method='predict', **cv_kwargs)
        Y_proba_CV = None
        if hasattr(grid_search.best_estimator_, "predict_proba"):
            Y_proba_CV = cross_val_predict(grid_search.best_estimator_, x, y, cv=cv, method='predict_proba', **cv_kwargs)
    
    cv_pred_path = os.path.join(directory, f'CV_Pred_vs_Actual_{model_name}_{analyte}.png')

    # Save results
    if directory:
        cv_df = pd.DataFrame(grid_search.cv_results_)
        out_file = os.path.join(directory, f'GridSearch_results_{model_name}_{analyte}.csv')
        cv_df.to_csv(out_file, index=False)
        cv_table_df = cv_df.sort_values(by="mean_test_score", ascending=False)


    return {
        'best_params': grid_search.best_params_,
        'best_score': grid_search.best_score_,
        'cv_results': grid_search.cv_results_,
        'Y_pred_CV': Y_pred_CV,
        'Y_proba_CV': Y_proba_CV,
        'y_mean': y_mean,
        'cv_pred_plot_path': cv_pred_path,
        'cv_table_df': cv_table_df,
        'best_estimator': grid_search.best_estimator_,
        'fold_df': fold_df,
        'cv_r2_pooled': cv_r2_pooled,
        'cv_rmse_pooled': cv_rmse_pooled
    }

