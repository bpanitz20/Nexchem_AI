#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Regression Model Wrappers
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from models.evaluator import evaluate_regression_model
from sklearn.neural_network import MLPClassifier
from models.evaluator import evaluate_classifier

def PLS_model(x, y, directory, axis, max_lv=10, analyte=""):
    """Wrapper for PLS using the generic evaluator"""
    model = PLSRegression()
    return evaluate_regression_model(
        x=x, y=y, directory=directory, axis=axis,
        model=model, param_name='n_components', 
        param_range=range(1, max_lv+1), model_name='PLS',
        analyte=analyte
    )

def Ridge_model(x, y, directory, axis, alpha_range=np.logspace(-2, 0, 10), analyte=""):
    """Wrapper for Ridge using the generic evaluator"""
    model = Ridge()
    
    # Ensure alpha_range is a list for .index() method
    alpha_range = list(alpha_range) if isinstance(alpha_range, np.ndarray) else alpha_range
        
    return evaluate_regression_model(
        x=x, y=y, directory=directory, axis=axis,
        model=model, param_name='alpha',
        param_range=alpha_range, model_name='Ridge',
        analyte=analyte
    )

def Lasso_model(x, y, directory, axis, alpha_range = np.linspace(0.01, 0.2, 20), analyte=""):
    """
    Lasso regression wrapper with identical interface to Ridge_model
    
    """
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    
    model = Lasso(max_iter=50000, tol=1e-4)  
    
    # Convert alpha_range to list if it's a numpy array
    alpha_range = list(alpha_range) if isinstance(alpha_range, np.ndarray) else alpha_range
        
    return evaluate_regression_model(
        x= X_scaled, y=y, directory=directory, axis=axis,
        model=model, param_name='alpha',
        param_range=alpha_range, model_name='Lasso',
        analyte=analyte
    )

def RF_model(x, y, directory, axis, n_estimators_range=None, analyte=""):
    """
    Random Forest wrapper matching PLS_model interface
    
    Parameters:
        x: Features (n_samples, n_features)
        y: Target values - will be automatically raveled if needed
        directory: Output directory
        axis: Plot axis
        n_estimators_range: Range of tree counts to test 
                           (default [50, 100, 200, 300, 500])
        analyte: Name of the target analyte (for labeling)
    """
    # Ensure y is 1D array to avoid warnings
    y = np.array(y).ravel()
    
    if n_estimators_range is None:
        n_estimators_range = [60, 75, 90, 100, 110, 125, 140]  # Common RF range
    
    model = RandomForestRegressor(
        random_state=42,          # For reproducibility
        min_samples_leaf=5,       # Regularization
        max_features='sqrt',      # Better for high-dim data like spectra
        max_depth=None,           # Let it find optimal depth
        n_jobs=-1,               # Use all cores
        oob_score=True           # Enable out-of-bag estimates
    )
    
    return evaluate_regression_model(
        x=x, y=y, directory=directory, axis=axis,
        model=model, param_name='n_estimators',
        param_range=n_estimators_range, model_name='RandomForest',
        analyte=analyte
    )

def MLPRegressor_model(x, y, directory, axis, analyte="", param_grid=None, random_state=42):
    """
    MLP Regressor wrapper with grid search capability
    
    Parameters:
        x: Features (n_samples, n_features)
        y: Target values
        directory: Output directory
        axis: Plot axis
        analyte: Analyte name for labeling
        param_grid: Custom parameter grid (optional)
        n_folds: Number of CV folds
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary containing model results (same format as evaluate_model)
    """
    # Ensure y is 1D
    y = np.array(y).ravel()
    
    # Scale features (critical for neural networks)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'hidden_layer_sizes': [(50,), (100,), (50,50)],
            'activation': ['relu'],
            'alpha': [0.02, 0.009, 0.01],  # L2 regularization
            'learning_rate_init': np.linspace(0.0001, 0.01, 10).tolist(),
            'early_stopping': [True],
            'solver': ['adam']
        }
    
    # Base model with fixed parameters
    base_model = MLPRegressor(
        max_iter=2000,
        random_state=random_state,
        verbose=False,
        n_iter_no_change=20,
        tol=1e-4,
        batch_size='auto'
    )
    
    # Run evaluation with grid search
    return evaluate_regression_model(
        x=X_scaled, 
        y=y, 
        directory=directory, 
        axis=axis,
        model=base_model,
        model_name='MLP',
        analyte=analyte,
        param_grid=param_grid,
    )

def KNNRegressor_model(x, y, directory, axis, analyte="", 
                      param_grid=None):
    """
    Enhanced KNN Regressor with grid search capability
    
    Parameters:
        x: Features (n_samples, n_features)
        y: Target values
        directory: Output directory
        axis: Plot axis
        analyte: Analyte name for labeling
        param_grid: Custom parameter grid (optional)
                    If None, uses default grid
        n_folds: Number of CV folds
    """
    # Ensure y is 1D
    y = np.array(y).ravel()
    
    # Scale features (critical for distance-based models)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'n_neighbors': np.linspace(3, 30, 10, dtype=int).tolist(),
            'weights': ['uniform', 'distance'],
            'p': [1, 2]  # 1: Manhattan, 2: Euclidean
        }
    
    # Base model
    base_model = KNeighborsRegressor(
        algorithm='auto',  # auto-select best algorithm
        n_jobs=-1         # use all cores
    )
    
    # Run evaluation with grid search
    return evaluate_regression_model(
        x=X_scaled, 
        y=y, 
        directory=directory, 
        axis=axis,
        model=base_model,
        model_name='KNN',
        analyte=analyte,
        param_grid=param_grid,
    )

def SVMRegressor_model(x, y, directory, axis, analyte="", 
                      param_grid=None):
    """
    Enhanced SVR model with grid search capability
    
    Parameters:
        x: Features (n_samples, n_features)
        y: Target values
        directory: Output directory
        axis: Plot axis
        analyte: Analyte name for labeling
        param_grid: Custom parameter grid (optional)
        n_folds: Number of CV folds
    """
    # Ensure y is 1D
    y = np.array(y).ravel()
    
    # Scale features (critical for SVR)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    
    # Default parameter grid if none provided
    if param_grid is None:
        param_grid = {
            'kernel': ['rbf'],
            'C': np.linspace(75, 90, 20).tolist(),  # Focus around 60 (40-80 range)
            'gamma': np.linspace(0.00025, 0.0003, 20).tolist(),  # Tight range around 0.0003
            'epsilon':  np.linspace(0.1, 0.15, 20).tolist()  # Range around 1.5 (1.0-2.0)
    }
        
    # Base model with fixed parameters
    base_model = SVR(
        tol=1e-3,
        max_iter=10000,
        cache_size=500
    )
    
    # Run evaluation with grid search
    return evaluate_regression_model(
        x=X_scaled, 
        y=y, 
        directory=directory, 
        axis=axis,
        model=base_model,
        model_name='SVR',
        analyte=analyte,
        param_grid=param_grid,
    )

def GBR_model(x, y, directory, axis, analyte="", param_grid=None):
    """
    Gradient Boosting Regressor wrapper with grid search support
    
    Parameters:
        x: Features (n_samples, n_features)
        y: Target values
        directory: Output directory
        axis: Spectral axis
        analyte: Name of the target analyte (for labeling)
        param_grid: Custom parameter grid (optional)
    """
    # Ensure y is 1D
    y = np.array(y).ravel()
    
    # Scale features (optional but often improves performance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(x)
    
    # Default grid if none provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 150],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 4, 5],
            'subsample': [0.8, 1.0]
        }
    
    base_model = GradientBoostingRegressor(
        random_state=42,
        loss='squared_error'
    )
    
    return evaluate_regression_model(
        x=X_scaled,
        y=y,
        directory=directory,
        axis=axis,
        model=base_model,
        model_name='GradientBoosting',
        analyte=analyte,
        param_grid=param_grid,
    )


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