"""
models/naive_model.py
---------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Naive Persistence Baseline model.

Strategy
--------
Predict that tomorrow's price equals today's price (last observed value).
This is the simplest meaningful baseline for financial time series.
Any useful model should beat this benchmark.

Validation
----------
For each step i in the validation window, the prediction is the price
immediately preceding that window (one-step persistence).  We use the
last training value to seed the chain and compare against actuals.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.base_model import BaseForecaster


class NaiveModel(BaseForecaster):
    """
    Naive persistence forecaster.

    For horizon h, every forecast equals the last known close price.

    Example
    -------
    >>> model = NaiveModel()
    >>> model.fit(close_series)
    >>> preds = model.predict(steps=7)
    """

    name: str = "Baseline (Naive)"

    def __init__(self) -> None:
        super().__init__()
        self._last_value: float | None = None
        self._val_series: pd.Series | None = None

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, val_size: int = 30) -> "NaiveModel":
        """
        Fit the naive model — simply records the last training value.

        Args:
            series   : pd.Series of Close prices, chronological.
            val_size : Number of tail observations held out for validation.

        Returns:
            self
        """
        if len(series) <= val_size:
            raise ValueError(
                f"Series length ({len(series)}) must exceed val_size ({val_size})."
            )

        train = series.iloc[:-val_size]
        val   = series.iloc[-val_size:]

        self._last_value = float(train.iloc[-1])
        self._val_series = val

        # Validation predictions: persist each previous value one step ahead.
        # For i-th val step, prediction = series[-(val_size + 1 - i)],
        # i.e. shift the entire series by 1 and slice the val window.
        shifted = series.shift(1)
        val_preds = shifted.iloc[-val_size:].to_numpy(dtype=float)

        self._val_y_true = val.to_numpy(dtype=float)
        self._val_y_pred = val_preds

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, steps: int) -> list[float]:
        """
        Return ``steps`` copies of the last observed close price.

        Args:
            steps : Forecast horizon in trading days.

        Returns:
            list[float] of length ``steps``, all equal to last close.
        """
        self._require_fitted()
        return [self._last_value] * steps  # type: ignore[return-value]

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        """
        Compute validation metrics for the naive persistence forecast.

        Returns:
            dict with MAE, RMSE, MAPE, R2, n_test.
        """
        self._require_fitted()
        return self._metrics(self._val_y_true, self._val_y_pred)
