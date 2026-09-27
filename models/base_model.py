"""
models/base_model.py
--------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Abstract base class that every forecasting model must implement.
Enforces a common interface: fit() → predict() → evaluate().

Design principles
-----------------
- All models receive a plain pandas Series of Close prices (chronological).
- fit() performs chronological train/validation split internally — never
  random shuffling, which would leak future data into training.
- predict() returns a plain Python list of float prices in natural scale.
- evaluate() returns a dict with keys: MAE, RMSE, MAPE, R2, n_test.
- Models must be importable even when optional dependencies (tensorflow,
  xgboost, statsmodels) are not installed; they raise ImportError at
  instantiation time, not at import time.
"""

from __future__ import annotations

import abc
from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster(abc.ABC):
    """
    Abstract base class for all StockIntel AI forecasting models.

    Subclasses must implement :meth:`fit`, :meth:`predict`, and
    :meth:`evaluate`.  The :meth:`evaluate` default implementation calls
    :meth:`fit` and then computes MAE / RMSE / MAPE / R² on the chronological
    validation split, so subclasses only need to override it if they have a
    more efficient approach.

    Attributes
    ----------
    name : str
        Human-readable model name used in the UI and comparison table.
    is_fitted : bool
        Set to True after :meth:`fit` completes successfully.
    """

    #: Override in each subclass with a short display name.
    name: str = "BaseForecaster"

    def __init__(self) -> None:
        self.is_fitted: bool = False
        self._val_y_true: np.ndarray | None = None
        self._val_y_pred: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Abstract interface — every subclass must implement these three
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def fit(self, series: pd.Series, val_size: int = 30) -> "BaseForecaster":
        """
        Train the model on a chronological Close-price series.

        The last ``val_size`` observations must be withheld as a validation
        set and never used for parameter fitting.  Implementations should
        store the validation predictions so that :meth:`evaluate` can use
        them without re-running inference.

        Args:
            series   : pd.Series of Close prices, DatetimeIndex, chronological.
            val_size : Number of tail observations to reserve for validation.

        Returns:
            self  (allows chaining: model.fit(series).predict(7))
        """

    @abc.abstractmethod
    def predict(self, steps: int) -> list[float]:
        """
        Forecast ``steps`` future close prices beyond the training window.

        The model must have been fitted first (:attr:`is_fitted` == True).
        Predictions are in the original price scale (not scaled/log space).

        Args:
            steps : Number of future trading days to forecast.

        Returns:
            list[float] of length ``steps``.

        Raises:
            RuntimeError : If called before :meth:`fit`.
        """

    @abc.abstractmethod
    def evaluate(self) -> dict[str, Any]:
        """
        Return validation-set performance metrics.

        Must be called after :meth:`fit`.  Uses only the held-out
        validation observations — never the training set.

        Returns
        -------
        dict with keys:
            MAE    : float  — Mean Absolute Error (price units)
            RMSE   : float  — Root Mean Squared Error (price units)
            MAPE   : float  — Mean Absolute Percentage Error (%)
            R2     : float  — Coefficient of Determination
            n_test : int    — Number of validation samples used
        """

    # ------------------------------------------------------------------
    # Shared metric helper — subclasses may call this from evaluate()
    # ------------------------------------------------------------------

    @staticmethod
    def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        """
        Compute MAE, RMSE, MAPE, R² from two equal-length arrays.

        Args:
            y_true : Ground-truth prices (original scale).
            y_pred : Predicted prices (original scale).

        Returns:
            dict with MAE, RMSE, MAPE, R2, n_test.
        """
        y_true = np.asarray(y_true, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mae  = float(np.mean(np.abs(y_true - y_pred)))
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

        nonzero = y_true != 0
        mape = (
            float(np.mean(
                np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])
            ) * 100)
            if nonzero.any() else float("nan")
        )

        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
        r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0

        return {
            "MAE":    mae,
            "RMSE":   rmse,
            "MAPE":   mape,
            "R2":     r2,
            "n_test": int(len(y_true)),
        }

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def _require_fitted(self) -> None:
        """Raise RuntimeError if :meth:`fit` has not been called."""
        if not self.is_fitted:
            raise RuntimeError(
                f"{self.name}.predict() called before fit(). "
                "Call fit(series) first."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted else "unfitted"
        return f"{self.__class__.__name__}(name={self.name!r}, {status})"
