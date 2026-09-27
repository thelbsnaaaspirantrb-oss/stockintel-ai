"""
models/xgboost_model.py
-----------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

XGBoost gradient-boosting forecasting model.

Design
------
- Identical feature pipeline to RandomForestModel (lag + rolling features).
- Train/validation split is always chronological.
- Recursive multi-step prediction same as the RF model.
- xgboost import is deferred to __init__ so the module is importable when
  xgboost is not installed; ImportError is raised at instantiation time.

Hyperparameters
---------------
Sensible defaults for financial time-series regression are used.
Early stopping on a chronological eval set is applied during fit to
avoid overfitting.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from models.base_model import BaseForecaster
from models.feature_engineering import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    build_features,
    chronological_split,
    get_X_y,
)

# ---------------------------------------------------------------------------
# Deferred import — fail at instantiation, not at import time
# ---------------------------------------------------------------------------
_XGB_OK: bool = True
try:
    import xgboost as xgb
except ImportError:
    _XGB_OK = False
    xgb = None  # type: ignore[assignment]


class XGBoostModel(BaseForecaster):
    """
    XGBoost gradient-boosting regressor with lag-based tabular features.

    Args:
        n_estimators  : Number of boosting rounds (default 300).
        learning_rate : Step-size shrinkage (default 0.05).
        max_depth     : Max tree depth (default 4).
        subsample     : Row subsampling ratio (default 0.8).
        random_state  : RNG seed (default 42).
        lags          : Lag depths to use as features.
        windows       : Rolling window sizes.

    Raises:
        ImportError : If xgboost is not installed.

    Example
    -------
    >>> model = XGBoostModel()
    >>> model.fit(close_series)
    >>> preds = model.predict(steps=7)
    """

    name: str = "XGBoost"

    def __init__(
        self,
        n_estimators:  int   = 300,
        learning_rate: float = 0.05,
        max_depth:     int   = 4,
        subsample:     float = 0.8,
        random_state:  int   = 42,
        lags:   list[int] = DEFAULT_LAGS,
        windows: list[int] = DEFAULT_WINDOWS,
    ) -> None:
        if not _XGB_OK:
            raise ImportError(
                "xgboost is required for XGBoostModel. "
                "Install it with: pip install xgboost"
            )
        super().__init__()
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            random_state=random_state,
            tree_method="hist",
            objective="reg:squarederror",
            verbosity=0,
        )
        self._lags    = lags
        self._windows = windows
        self._model: Any = None
        self._train_series: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, val_size: int = 30) -> "XGBoostModel":
        """
        Build features, chronologically split, and fit XGBoost.

        Args:
            series   : Chronological Close-price Series.
            val_size : Tail observations withheld for validation.

        Returns:
            self
        """
        feat_df = build_features(series, lags=self._lags, windows=self._windows)
        X, y = get_X_y(feat_df)

        if len(X) <= val_size:
            raise ValueError(
                f"Feature matrix too short ({len(X)}) for val_size={val_size}."
            )

        X_train, X_val, y_train, y_val = chronological_split(X, y, val_size)

        self._model = xgb.XGBRegressor(**self._params)

        # Use validation set as eval set for early stopping
        self._model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        self._train_series = series.iloc[: len(series) - val_size]
        self._val_y_true   = y_val
        self._val_y_pred   = self._model.predict(X_val).astype(float)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, steps: int) -> list[float]:
        """
        Recursively forecast ``steps`` future closing prices.

        Args:
            steps : Number of future trading days.

        Returns:
            list[float] of length ``steps``.
        """
        self._require_fitted()

        max_lag   = max(self._lags + self._windows)
        price_buf = list(self._train_series.iloc[-max_lag:].to_numpy(dtype=float))
        predictions: list[float] = []

        for _ in range(steps):
            tmp_series = pd.Series(
                price_buf,
                index=pd.date_range(end="2100-01-01", periods=len(price_buf), freq="B"),
            )
            feat_df = build_features(
                tmp_series, lags=self._lags, windows=self._windows
            )
            if feat_df.empty:
                predictions.append(float(price_buf[-1]))
                price_buf.append(float(price_buf[-1]))
                continue

            X_last = feat_df.drop(columns=["target"]).iloc[[-1]].to_numpy(dtype=float)
            pred   = float(self._model.predict(X_last)[0])
            predictions.append(pred)
            price_buf.append(pred)

        return predictions

    # ------------------------------------------------------------------
    def evaluate(self) -> dict[str, Any]:
        """Return chronological validation-set metrics."""
        self._require_fitted()
        return self._metrics(self._val_y_true, self._val_y_pred)
