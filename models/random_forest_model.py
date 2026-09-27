"""
models/random_forest_model.py
------------------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Random Forest Regressor forecasting model.

Design
------
- Input features are lag values, rolling statistics, and calendar features
  built by ``models.feature_engineering.build_features``.
- Train/validation split is always chronological (tail slice — no shuffling).
- For multi-step forecasting (steps > 1) we use a recursive strategy:
  each predicted value is fed back as the next lag, rolling features are
  updated accordingly.
- sklearn is a hard dependency (already installed).

Recursive multi-step forecasting
---------------------------------
At predict time we maintain a rolling buffer of the last ``max_lag``
prices.  For each step:
  1. Compute the feature vector from the buffer.
  2. Predict the next price.
  3. Append the prediction to the buffer and slide the window.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from models.base_model import BaseForecaster
from models.feature_engineering import (
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
    build_features,
    chronological_split,
    get_X_y,
)


class RandomForestModel(BaseForecaster):
    """
    Random Forest Regressor with lag-based tabular features.

    Args:
        n_estimators : Number of trees (default 200).
        max_depth    : Max depth per tree (default None — unlimited).
        random_state : RNG seed for reproducibility (default 42).
        lags         : Lag depths to use as features.
        windows      : Rolling window sizes for mean/std features.

    Example
    -------
    >>> model = RandomForestModel()
    >>> model.fit(close_series)
    >>> preds = model.predict(steps=7)
    """

    name: str = "Random Forest"

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int | None = None,
        random_state: int = 42,
        lags: list[int] = DEFAULT_LAGS,
        windows: list[int] = DEFAULT_WINDOWS,
    ) -> None:
        super().__init__()
        self._rf = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self._scaler = StandardScaler()
        self._lags    = lags
        self._windows = windows
        # Buffers for recursive prediction
        self._train_series: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, val_size: int = 30) -> "RandomForestModel":
        """
        Build features, chronologically split, and fit the RF.

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
                f"Feature matrix too short ({len(X)}) for val_size={val_size}. "
                "Use a longer historical period."
            )

        X_train, X_val, y_train, y_val = chronological_split(X, y, val_size)

        X_train_s = self._scaler.fit_transform(X_train)
        X_val_s   = self._scaler.transform(X_val)

        self._rf.fit(X_train_s, y_train)

        # Store the full training series for recursive prediction
        self._train_series = series.iloc[: len(series) - val_size]

        self._val_y_true = y_val
        self._val_y_pred = self._rf.predict(X_val_s)
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

        max_lag    = max(self._lags + self._windows)
        price_buf  = list(self._train_series.iloc[-max_lag:].to_numpy(dtype=float))
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
            X_last_s = self._scaler.transform(X_last)
            pred = float(self._rf.predict(X_last_s)[0])
            predictions.append(pred)
            price_buf.append(pred)

        return predictions

    # ------------------------------------------------------------------
    def evaluate(self) -> dict[str, Any]:
        """Return chronological validation-set metrics."""
        self._require_fitted()
        return self._metrics(self._val_y_true, self._val_y_pred)
