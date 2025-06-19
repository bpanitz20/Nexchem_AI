#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Wrappers
"""
import numpy as np
import pandas as pd
import os
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
from sklearn.metrics import accuracy_score, f1_score
from plotting.plot_regression import (
    plot_pred_vs_actual,
    plot_coefficients,
    plot_feature_importance,
    plot_vip_scores,
    print_model_summary,
    print_CV_table,
    plot_t2_q_residuals,
)
from plotting.plot_classifier import (
    plot_confusion_matrix,
    plot_roc_curve,
    plot_decision_boundary
)

def PLS_model(x, y, directory, axis, max_lv=10, analyte="", groups=None, manual_param=None, sample_ids=None, n_folds=8):
    param_name = 'n_components'
    param_range = list(range(1, max_lv + 1))
    model = PLSRegression()

    # Run CV
    cv_results = KFold_CV(x, y, model, param_name, param_range, analyte=analyte, 
                          groups=groups, model_name='PLS', directory=directory,
                          manual_param=manual_param, n_folds=n_folds)

    # Final fit with optimal parameter
    if manual_param is not None:
        #print(f"Manual-selected {param_name}: {manual_param}")
        final_param = manual_param
    else:
        final_param = cv_results['optimal_param']
        #print(f"Auto-selected {param_name}: {final_param}")

    # Fit model
    model.set_params(**{param_name: final_param})
    model.fit(x, y - cv_results['y_mean'])
    Y_pred = model.predict(x)

    # Metrics
    final_r2 = r2_score(y - cv_results['y_mean'], Y_pred)
    final_mse = mean_squared_error(y - cv_results['y_mean'], Y_pred)
    final_r2_CV = cv_results['mean_r2_CV'][param_range.index(cv_results['optimal_param'])]
    final_mse_CV = cv_results['mean_mse_CV'][param_range.index(cv_results['optimal_param'])]

    # Print summary
    print_CV_table(
        param_name=param_name,
        param_range=param_range,
        r2_cv=cv_results['mean_r2_CV'],
        r2_cal=cv_results['mean_r2_cal'],
        mse_cv=cv_results['mean_mse_CV'],
        rmse_cal=cv_results['mean_rmse_cal'],
        model_name="PLS",
        analyte=analyte,
        directory=directory
        )
    rmse_cv = [mse**0.5 for mse in cv_results['mean_mse_CV']]
    cv_table_df = pd.DataFrame({
    param_name: param_range,
    "R²_Cal": cv_results['mean_r2_cal'],
    "R²_CV": cv_results['mean_r2_CV'],
    "RMSE_CV": rmse_cv,
    "RMSE_Cal": cv_results['mean_rmse_cal']
    })
    
    summary_string=print_model_summary(
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
    plot_t2_q_residuals(model, x, y, analyte, directory, model_name="PLS", sample_ids=sample_ids)
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
        'cv_results': cv_results,
        'cv_table_df': cv_table_df,
        'cv_r2_plot_path': cv_results['cv_r2_plot_path'],
        'cv_rmse_plot_path': cv_results['cv_rmse_plot_path'],
        'cv_pred_plot_path': cv_results['cv_pred_plot_path'],
        'summary': summary_string,
        'vip_plot_path': os.path.join(directory, f"VIP_Scores_PLS_{analyte}.png"),
        'coef_plot_path': os.path.join(directory, f"PLS_Coefficients_{analyte}.png"),
        't2_plot_path': os.path.join(directory, f"T2_vs_Q_Residuals_PLS_{analyte}.png"),
        'final_pred_plot_path': os.path.join(directory, f"Final_Pred_vs_Actual_PLS_{analyte}.png"),
            
    }

def MLPRegressor_model(x, y, directory, axis, analyte="", param_grid=None, groups=None, random_state=42, n_folds=8):
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
        groups=groups,
        directory=directory,
        n_folds=n_folds
    )

    final_model = cv_results['best_estimator']
    Y_pred = final_model.predict(X_scaled) + cv_results['y_mean']

    final_r2 = r2_score(y, Y_pred)
    final_mse = mean_squared_error(y, Y_pred)

    Y_pred_CV = cv_results['Y_pred_CV'] + cv_results['y_mean']
    final_r2_CV = r2_score(y, Y_pred_CV)
    final_mse_CV = mean_squared_error(y, Y_pred_CV)

    summary_string=print_model_summary(
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
        'cv_results': {
        'cv_results': cv_results['cv_results'],
        'best_params': cv_results['best_params'],
        'y_mean': cv_results['y_mean']  # <- this is missing!
        },
        'best_params': cv_results['best_params'],
        'cv_pred_plot_path': cv_results['cv_pred_plot_path'],
        'cv_table_df': cv_results['cv_table_df'],
        'summary': summary_string,
        'feature_importance_path': os.path.join(directory, f"Feature_Importance_MLP_{analyte}.png"),
        'final_pred_plot_path': os.path.join(directory, f"Final_Pred_vs_Actual_MLP_{analyte}.png")
    }


"""
Classification Model Wrppers
"""

def MLPClassifier_model(x, y, directory, axis, analyte="", param_grid=None, groups=None, random_state=42):
    y = np.array(y).ravel()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)

    if param_grid is None:
        param_grid = {
            'alpha': [0.01, 0.001],
            'hidden_layer_sizes': [(50,), (100,), (50, 50)],
            'learning_rate_init': [0.001],
            'early_stopping': [True]
        }

    base_model = MLPClassifier(
        max_iter=2000,
        random_state=random_state,
        tol=1e-4,
        verbose=False
    )

    # Run GridSearch CV
    cv_results = KFold_Gridsearch_CV(
        x=X_scaled,
        y=y,
        model=base_model,
        param_grid=param_grid,
        task="classification",
        groups=groups,
        analyte=analyte,
        model_name="MLPClassifier",
        directory=directory
    )

    # Final model and predictions
    final_model = cv_results["best_estimator"]
    y_pred = final_model.predict(X_scaled)
    y_cv_pred = cv_results["Y_pred_CV"]
    y_cv_proba = cv_results["Y_proba_CV"]

    # Metrics
    acc = accuracy_score(y, y_pred)
    f1 = f1_score(y, y_pred, average='macro')
    acc_cv = accuracy_score(y, y_cv_pred)
    f1_cv = f1_score(y, y_cv_pred, average='macro')

    print(f"\n📊 MLPClassifier Performance for {analyte}")
    print(f"Best Params: {cv_results['best_params']}")
    print(f"Accuracy (Train): {acc:.3f} | F1 (Train): {f1:.3f}")
    print(f"Accuracy (CV):    {acc_cv:.3f} | F1 (CV): {f1_cv:.3f}")

    # Plots
    plot_confusion_matrix(y_pred, y, directory, "MLPClassifier", analyte)
    plot_confusion_matrix(y_cv_pred, y, directory, "MLPClassifier", analyte, suffix="_CV")

    if hasattr(final_model, "predict_proba"):
        y_proba = final_model.predict_proba(X_scaled)
        plot_roc_curve(y, y_proba, directory, "MLPClassifier", analyte)
        if y_cv_proba is not None:
            plot_roc_curve(y, y_cv_proba, directory, "MLPClassifier", analyte, suffix="_CV")

    plot_decision_boundary(X_scaled, y, final_model, directory, "MLPClassifier", analyte)

    return {
        "model": final_model,
        "best_params": cv_results["best_params"],
        "accuracy": acc,
        "f1": f1,
        "cv_accuracy": acc_cv,
        "cv_f1": f1_cv
    }

