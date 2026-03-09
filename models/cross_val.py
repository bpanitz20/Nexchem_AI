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



"""
import os
import numpy as np
import pandas as pd

from scipy.stats import t
from sklearn.model_selection import (
    KFold,
    GroupKFold,
    ShuffleSplit,
    GroupShuffleSplit,
    cross_val_score,
    cross_val_predict,
)
from sklearn.metrics import r2_score, mean_squared_error, make_scorer


def KFold_CV(
    x, y, model, param_name, param_range,
    n_folds=8, groups=None,
    analyte="", model_name="",
    directory="", manual_param=None,
    sample_ids=None, class_labels=None,
    compute_ci=True,
    compute_model_ci=True,
    n_model_ci_repeats=100,
    model_ci_test_size=None,
    model_ci_random_state=42,
):
    
    K-fold cross-validation with:

    1) fold-based 95% CI printed for each parameter value
    2) repeated holdout model-level 95% CI for the final selected model

    Notes
    -----
    - Fold CI is based on variability across CV folds for each parameter value.
    - Model CI is based on repeated random holdout splits for the final selected model.
    - If groups is provided, GroupKFold is used for the main CV, and GroupShuffleSplit
      is used for repeated grouped holdout CI of the final model.
    

    def mean_ci(vals, alpha=0.95):
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]

        if len(vals) == 0:
            return np.nan, np.nan, np.nan
        if len(vals) == 1:
            return float(vals[0]), np.nan, np.nan

        mean_val = np.mean(vals)
        sem = np.std(vals, ddof=1) / np.sqrt(len(vals))
        tval = t.ppf((1 + alpha) / 2, df=len(vals) - 1)
        low = mean_val - tval * sem
        high = mean_val + tval * sem
        return float(mean_val), float(low), float(high)

    x = np.asarray(x)
    y = np.asarray(y)

    if groups is not None:
        groups = np.asarray(groups)

    # mean-center y
    y_mean = np.mean(y, axis=0)
    y_centered = y - y_mean

    # main CV strategy
    if groups is not None:
        cv = GroupKFold(n_splits=n_folds)
        cv_kwargs = {"groups": groups}
        print("Using GroupKFold CV")
    else:
        cv = KFold(n_splits=n_folds, shuffle=False, random_state=None)
        cv_kwargs = {}
        print("\nUsing standard KFold CV")
    print(f"n_folds = {n_folds}")

    r2_scorer = make_scorer(r2_score)
    mse_scorer = make_scorer(mean_squared_error)

    mean_r2_CV = []
    mean_r2_cal = []
    mean_mse_CV = []
    mean_rmse_cal = []
    all_predictions = {}
    pooled_r2_CV = []
    pooled_rmse_CV = []

    # fold-based CI outputs
    r2_ci_lower = []
    r2_ci_upper = []
    rmse_ci_lower = []
    rmse_ci_upper = []
    fold_r2_scores_all = {}
    fold_rmse_scores_all = {}

    fold_assignments = {} if sample_ids is not None else None

    # -------------------------------------------------
    # main parameter loop
    # -------------------------------------------------
    for param_value in param_range:
        model.set_params(**{param_name: param_value})

        r2_scores = cross_val_score(
            model, x, y_centered, cv=cv, scoring=r2_scorer, **cv_kwargs
        )
        mse_scores = cross_val_score(
            model, x, y_centered, cv=cv, scoring=mse_scorer, **cv_kwargs
        )
        rmse_scores = np.sqrt(np.abs(mse_scores))

        mean_r2_CV.append(np.mean(r2_scores))
        mean_mse_CV.append(np.mean(mse_scores))

        fold_r2_scores_all[param_value] = r2_scores
        fold_rmse_scores_all[param_value] = rmse_scores

        # fold-based CI per param value
        if compute_ci and len(r2_scores) > 1:
            r2_mean, r2_low, r2_high = mean_ci(r2_scores)
            rmse_mean, rmse_low, rmse_high = mean_ci(rmse_scores)

            r2_ci_lower.append(r2_low)
            r2_ci_upper.append(r2_high)
            rmse_ci_lower.append(rmse_low)
            rmse_ci_upper.append(rmse_high)

            print(
                f"{param_name}={param_value} | "
                f"R2CV = {r2_mean:.3f} "
                f"(95% CI: {r2_low:.3f} – {r2_high:.3f}) | "
                f"RMSECV = {rmse_mean:.3f} "
                f"(95% CI: {rmse_low:.3f} – {rmse_high:.3f})"
            )
        else:
            r2_ci_lower.append(np.nan)
            r2_ci_upper.append(np.nan)
            rmse_ci_lower.append(np.nan)
            rmse_ci_upper.append(np.nan)

        # pooled CV predictions
        yhat_cv = cross_val_predict(model, x, y_centered, cv=cv, **cv_kwargs)
        all_predictions[param_value] = yhat_cv

        pooled_r2_CV.append(r2_score(y_centered, yhat_cv))
        pooled_rmse_CV.append(np.sqrt(mean_squared_error(y_centered, yhat_cv)))

        # calibration
        model.fit(x, y_centered)
        y_fit = model.predict(x)
        mse_fit = mean_squared_error(y_centered, y_fit)
        r2_fit = r2_score(y_centered, y_fit)

        mean_rmse_cal.append(np.sqrt(mse_fit))
        mean_r2_cal.append(r2_fit)

        # fold assignments
        if param_value == list(param_range)[0] and fold_assignments is not None:
            for fold_idx, (_, test_idx) in enumerate(cv.split(x, y_centered, **cv_kwargs)):
                for i in test_idx:
                    fold_assignments[sample_ids[i]] = fold_idx

    # -------------------------------------------------
    # fold assignment table
    # -------------------------------------------------
    if fold_assignments:
        fold_df = pd.DataFrame(
            list(fold_assignments.items()),
            columns=["Sample ID", "CV Fold"]
        )
        fold_df = fold_df.sort_values(by="CV Fold").reset_index(drop=True)
    else:
        fold_df = None

    # -------------------------------------------------
    # choose optimal parameter
    # -------------------------------------------------
    pooled_rmscev = np.array(pooled_rmse_CV)
    rmsec = np.array(mean_rmse_cal)
    rmse_gap = np.abs(rmsec - pooled_rmscev)

    if manual_param is not None:
        optimal_param = manual_param
    else:
        optimal_idx = int(np.argmin(rmse_gap))
        optimal_param = list(param_range)[optimal_idx]

    opt_idx = list(param_range).index(optimal_param)

    Y_pred_CV = all_predictions[optimal_param]

    cv_r2_plot_path = os.path.join(directory, f"CV_R2_{model_name}_{analyte}.png")
    cv_rmse_plot_path = os.path.join(directory, f"CV_RMSE_{model_name}_{analyte}.png")
    cv_pred_path = os.path.join(directory, f"CV_Pred_vs_Actual_{model_name}_{analyte}.png")

    # -------------------------------------------------
    # final model metrics from main CV
    # -------------------------------------------------
    final_r2_cal = mean_r2_cal[opt_idx]
    final_r2_cv = pooled_r2_CV[opt_idx]
    final_rmse = mean_rmse_cal[opt_idx]
    final_rmsecv = pooled_rmse_CV[opt_idx]

    print(f"\nFinal Model Metrics for {analyte} ({model_name}):")
    print(f"{param_name}: {optimal_param}")
    print(f"R²_Cal: {final_r2_cal:.4f}")
    print(f"R²_CV: {final_r2_cv:.4f}")
    print(f"RMSE: {final_rmse:.4f}")
    print(f"RMSECV: {final_rmsecv:.4f}")

    # -------------------------------------------------
    # repeated holdout MODEL CI for final selected model
    # -------------------------------------------------
    model_ci_summary = {
        "R2_mean": np.nan,
        "R2_ci_low": np.nan,
        "R2_ci_high": np.nan,
        "RMSE_mean": np.nan,
        "RMSE_ci_low": np.nan,
        "RMSE_ci_high": np.nan,
        "R2_values": [],
        "RMSE_values": [],
        "n_repeats": 0,
    }

    if compute_model_ci:
        final_model = model.__class__(**model.get_params())
        final_model.set_params(**{param_name: optimal_param})

        # sensible default test size
        if model_ci_test_size is None:
            model_ci_test_size = max(1 / n_folds, 0.40)

        # choose repeated splitter
        if groups is not None:
            splitter = GroupShuffleSplit(
                n_splits=n_model_ci_repeats,
                test_size=model_ci_test_size,
                random_state=model_ci_random_state,
            )
            split_iter = splitter.split(x, y_centered, groups=groups)
            print(
                f"\nRunning repeated grouped holdout CI "
                f"({n_model_ci_repeats} repeats, test_size={model_ci_test_size:.3f})"
            )
        else:
            splitter = ShuffleSplit(
                n_splits=n_model_ci_repeats,
                test_size=model_ci_test_size,
                random_state=model_ci_random_state,
            )
            split_iter = splitter.split(x, y_centered)
            print(
                f"\nRunning repeated holdout CI "
                f"({n_model_ci_repeats} repeats, test_size={model_ci_test_size:.3f})"
            )

        model_ci_r2 = []
        model_ci_rmse = []

        for i, (train_idx, test_idx) in enumerate(split_iter, start=1):
            x_train, x_test = x[train_idx], x[test_idx]
            y_train, y_test = y_centered[train_idx], y_centered[test_idx]

            final_model.fit(x_train, y_train)
            y_pred_test = final_model.predict(x_test)

            rep_r2 = r2_score(y_test, y_pred_test)
            rep_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

            model_ci_r2.append(rep_r2)
            model_ci_rmse.append(rep_rmse)

        r2_mean_m, r2_low_m, r2_high_m = mean_ci(model_ci_r2)
        rmse_mean_m, rmse_low_m, rmse_high_m = mean_ci(model_ci_rmse)

        model_ci_summary = {
            "R2_mean": r2_mean_m,
            "R2_ci_low": r2_low_m,
            "R2_ci_high": r2_high_m,
            "RMSE_mean": rmse_mean_m,
            "RMSE_ci_low": rmse_low_m,
            "RMSE_ci_high": rmse_high_m,
            "R2_values": model_ci_r2,
            "RMSE_values": model_ci_rmse,
            "n_repeats": len(model_ci_r2),
        }

        print(f"\nRepeated-holdout model CI for final {analyte} model:")
        print(
            f"R² = {r2_mean_m:.4f} "
            f"(95% CI: {r2_low_m:.4f} – {r2_high_m:.4f})"
        )
        print(
            f"RMSE = {rmse_mean_m:.4f} "
            f"(95% CI: {rmse_low_m:.4f} – {rmse_high_m:.4f})"
        )

    # -------------------------------------------------
    # optional plotting
    # -------------------------------------------------
    if directory:
        plot_cv_performance(
            param_range,
            pooled_r2_CV,
            mean_r2_cal,
            pooled_rmse_CV,
            mean_rmse_cal,
            param_name,
            analyte,
            model_name,
            directory
        )
        plot_pred_vs_actual(
            y,
            Y_pred_CV + y_mean,
            directory,
            f"CV Predicted vs. Actual for {analyte} ({model_name})",
            f"CV_Pred_vs_Actual_{model_name}_{analyte}.png",
            class_labels=class_labels
        )

    return {
        "mean_r2_CV": mean_r2_CV,
        "mean_mse_CV": mean_mse_CV,
        "mean_r2_cal": mean_r2_cal,
        "mean_rmse_cal": mean_rmse_cal,
        "optimal_param": optimal_param,
        "Y_pred_CV": Y_pred_CV,
        "y_mean": y_mean,
        "cv_r2_plot_path": cv_r2_plot_path,
        "cv_rmse_plot_path": cv_rmse_plot_path,
        "cv_pred_plot_path": cv_pred_path,
        "fold_df": fold_df,
        "pooled_r2_CV": pooled_r2_CV,
        "pooled_rmse_CV": pooled_rmse_CV,

        # fold-based per-parameter CI
        "r2_ci_lower": r2_ci_lower,
        "r2_ci_upper": r2_ci_upper,
        "rmse_ci_lower": rmse_ci_lower,
        "rmse_ci_upper": rmse_ci_upper,
        "fold_r2_scores_all": fold_r2_scores_all,
        "fold_rmse_scores_all": fold_rmse_scores_all,

        # final selected model metrics
        "final_r2_cal": final_r2_cal,
        "final_r2_cv": final_r2_cv,
        "final_rmse": final_rmse,
        "final_rmsecv": final_rmsecv,

        # repeated-holdout model CI
        "model_ci": model_ci_summary,
    }
"""








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
        n_jobs=-1,
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
        cv_table_df = cv_df.sort_values(by="mean_test_score", ascending=False)  # Optional sort


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

