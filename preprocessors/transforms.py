"""
preprocessors/transforms.py
----------------------------
Composable preprocessing step classes for Raman spectroscopy data.

Each class follows the fit / transform protocol:

    fit(X)           -- compute and store any state from training data.
                        Returns self so calls can be chained.
    transform(X)     -- apply the transform using stored state.
                        Returns the transformed array.
    fit_transform(X) -- fit then transform in a single call.

Steps that are stateless have trivial fit() methods that simply return self.

X convention throughout: 2D numpy array, shape (n_spectra, n_features).
"""

import numpy as np


class GlobalMeanCenterStep:
    """
    Global mean-centering across all spectra in a dataset.

    fit()       computes the column-wise mean across the training matrix and
                stores it as self.mean_  (shape: n_features,).
    transform() subtracts self.mean_ from every row of X.

    This replaces the ad-hoc use_state / return_state dict pattern previously
    scattered across preprocess_savgol_snv_mc and group_preprocess_savgol_snv_mc.

    Training usage
    --------------
        step = GlobalMeanCenterStep()
        X_mc = step.fit_transform(X_train)

    Prediction usage (reuse training mean — no data leakage)
    --------------------------------------------------------
        X_mc_pred = step.transform(X_pred)

    Restoring state from a legacy dict (backward compat)
    -----------------------------------------------------
        step = GlobalMeanCenterStep()
        step.mean_ = np.asarray(legacy_state["mean_spectrum"])
        X_mc_pred  = step.transform(X_pred)
    """

    def __init__(self):
        self.mean_ = None

    def fit(self, X):
        """
        Compute and store the column-wise mean of X.

        Parameters
        ----------
        X : array-like, shape (n_spectra, n_features)

        Returns
        -------
        self
        """
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        return self

    def transform(self, X):
        """
        Subtract the fitted mean from every row of X.

        Parameters
        ----------
        X : array-like, shape (n_spectra, n_features)

        Returns
        -------
        X_mc : np.ndarray, same shape as X

        Raises
        ------
        RuntimeError if fit() has not been called and mean_ is not set.
        """
        if self.mean_ is None:
            raise RuntimeError(
                "GlobalMeanCenterStep: call fit() or assign mean_ "
                "before calling transform()."
            )
        return np.asarray(X, dtype=float) - self.mean_

    def fit_transform(self, X):
        """
        Fit to X, then return the mean-centered version of X.

        Parameters
        ----------
        X : array-like, shape (n_spectra, n_features)

        Returns
        -------
        X_mc : np.ndarray, same shape as X
        """
        return self.fit(X).transform(X)


class SNVStep:
    """
    Standard Normal Variate (SNV) normalization, applied row-wise.

    Stateless — fit() is a no-op that returns self.

    SNV formula per spectrum:
        x_snv = (x - mean(x)) / std(x)

    X convention: 2D numpy array, shape (n_spectra, n_features).

    Usage
    -----
        step = SNVStep()
        X_snv = step.transform(X)          # or fit_transform — identical result

    Single-spectrum usage (preserves existing call patterns)
    ---------------------------------------------------------
        y_snv = step.transform(y[np.newaxis, :]).squeeze(0)
    """

    def fit(self, X):
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        mean = X.mean(axis=1, keepdims=True)
        std = X.std(axis=1, keepdims=True)
        return (X - mean) / std

    def fit_transform(self, X):
        return self.transform(X)
