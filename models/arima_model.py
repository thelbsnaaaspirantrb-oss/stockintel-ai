"""
models/arima_model.py
---------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

ARIMA forecasting model via statsmodels.

Design
------
- Uses auto_arima-style order selection (grid search over p,d,q within
  safe limits) to pick the best ARIMA(p,d,q) by AIC.
- Falls back to ARIMA(1,1,0) if order selection fails.
- Validation: the model is fit on the training portion; one-step-ahead
  rolling predictions are generated for the validation window using
  ``get_forecast()``.
- statsmodels import is deferred to __init__ so the module is importable
  even without statsmodels installed (ImportError at instantiation time).

Note on stationarity
---------------------
Financial price series are non-stationary.  d=1 (first differencing) is
applied by default and enforced by restricting grid search to d ∈ {1, 2}.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd

from models.base_model import BaseForecaster

# ---------------------------------------------------------------------------
# Dependency check (deferred — only fail at instantiation)
# ---------------------------------------------------------------------------
_STATSMODELS_OK: bool = True
try:
    from statsmodels.tsa.arima.model import ARIMA as _ARIMA
    from statsmodels.tools.sm_exceptions import ConvergenceWarning as _CW
except ImportError:
    _STATSMODELS_OK = False
    _ARIMA = None           # type: ignore[assignment]
    _CW    = Warning        # type: ignore[assignment,misc]


def _best_arima_order(
    train: np.ndarray,
    p_range: range = range(0, 4),
    d_range: range = range(1, 3),
    q_range: range = range(0, 4),
) -> tuple[int, int, int]:
    """
    Select ARIMA(p,d,q) order by minimising AIC over a restricted grid.

    Skips any combination that fails to converge or raises an exception.

    Args:
        train   : 1-D numpy array of training prices.
        p_range : AR orders to try.
        d_range : Differencing orders to try.
        q_range : MA orders to try.

    Returns:
        (p, d, q) tuple with lowest AIC, or (1, 1, 0) as fallback.
    """
    best_aic   = np.inf
    best_order = (1, 1, 0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for d in d_range:
            for p in p_range:
                for q in q_range:
                    if p == 0 and q == 0:
                        continue  # degenerate model
                    try:
                        res = _ARIMA(train, order=(p, d, q)).fit()
                        if res.aic < best_aic:
                            best_aic   = res.aic
                            best_order = (p, d, q)
                    except Exception:  # noqa: BLE001
                        continue

    return best_order


class ARIMAModel(BaseForecaster):
    """
    ARIMA(p,d,q) forecaster backed by statsmodels.

    The best order (p,d,q) is selected via AIC grid search on the training
    set.  Validation uses rolling one-step-ahead forecasts.

    Args:
        order : Optional fixed (p,d,q) tuple.  If None (default), order is
                selected automatically on each call to :meth:`fit`.

    Raises:
        ImportError : If statsmodels is not installed.

    Example
    -------
    >>> model = ARIMAModel()
    >>> model.fit(close_series)
    >>> preds = model.predict(steps=7)
    """

    name: str = "ARIMA"

    def __init__(self, order: tuple[int, int, int] | None = None) -> None:
        if not _STATSMODELS_OK:
            raise ImportError(
                "statsmodels is required for ARIMAModel. "
                "Install it with: pip install statsmodels"
            )
        super().__init__()
        self._fixed_order = order
        self._fitted_model: Any = None
        self._order_used: tuple[int, int, int] | None = None

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, val_size: int = 30) -> "ARIMAModel":
        """
        Select ARIMA order and fit on the training portion.

        Args:
            series   : Chronological Close-price Series.
            val_size : Tail observations withheld for validation.

        Returns:
            self
        """
        if len(series) <= val_size + 10:
            raise ValueError(
                f"Series too short ({len(series)}) for ARIMA "
                f"with val_size={val_size}."
            )

        train_vals = series.iloc[:-val_size].to_numpy(dtype=float)
        val_vals   = series.iloc[-val_size:].to_numpy(dtype=float)

        # Select order
        if self._fixed_order is not None:
            order = self._fixed_order
        else:
            order = _best_arima_order(train_vals)

        self._order_used = order

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", _CW)
            warnings.simplefilter("ignore", UserWarning)
            self._fitted_model = _ARIMA(train_vals, order=order).fit()

        # Validation: rolling one-step-ahead predictions
        val_preds: list[float] = []
        history = list(train_vals)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for actual in val_vals:
                try:
                    tmp = _ARIMA(np.array(history), order=order).fit()
                    fc  = float(tmp.forecast(steps=1)[0])
                except Exception:  # noqa: BLE001
                    # Fall back to last observed value on failure
                    fc = float(history[-1])
                val_preds.append(fc)
                history.append(float(actual))

        self._val_y_true = val_vals
        self._val_y_pred = np.array(val_preds)
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, steps: int) -> list[float]:
        """
        Forecast ``steps`` future values from the fitted model.

        Args:
            steps : Forecast horizon.

        Returns:
            list[float] of length ``steps``.
        """
        self._require_fitted()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fc = self._fitted_model.forecast(steps=steps)
        return [float(v) for v in fc]

    # ------------------------------------------------------------------
    def evaluate(self) -> dict:
        """Return validation-set metrics."""
        self._require_fitted()
        return self._metrics(self._val_y_true, self._val_y_pred)

    @property
    def order(self) -> tuple[int, int, int] | None:
        """The (p,d,q) order selected / used during the last fit."""
        return self._order_used
