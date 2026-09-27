"""
models/model_registry.py
------------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Central model factory and multi-model runner.

Public API
----------
    REGISTRY        : dict mapping display name → class.
    get_model(name) : Instantiate a model by display name.
    available_models() : List names of models whose dependencies are met.
    run_all_models(series, steps, val_size) : Fit+evaluate all available
        models and return (comparison_df, predictions_dict).
    auto_select(series, steps, val_size)    : Run all models, pick the
        one with the lowest validation RMSE, return its predictions.

Model display names (match config.MODEL_OPTIONS)
-------------------------------------------------
    "Auto"              — evaluate all, pick best by RMSE
    "LSTM"              — pre-trained 2-layer LSTM
    "Random Forest"     — RandomForestRegressor with lag features
    "XGBoost"           — XGBRegressor with lag features
    "ARIMA"             — ARIMA(p,d,q) via statsmodels
    "Baseline (Naive)"  — persistence model
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from models.base_model import BaseForecaster
from models.evaluator import evaluate_all, best_model_name
from models.naive_model import NaiveModel
from models.random_forest_model import RandomForestModel

# ---------------------------------------------------------------------------
# Optional dependency models — import classes only (not instantiate)
# ---------------------------------------------------------------------------
try:
    from models.lstm_model import LSTMModel as _LSTMClass
    _LSTM_AVAILABLE = True
except ImportError:
    _LSTMClass = None       # type: ignore[assignment,misc]
    _LSTM_AVAILABLE = False

try:
    from models.arima_model import ARIMAModel as _ARIMAClass
    _ARIMA_AVAILABLE = True
except ImportError:
    _ARIMAClass = None      # type: ignore[assignment,misc]
    _ARIMA_AVAILABLE = False

try:
    from models.xgboost_model import XGBoostModel as _XGBClass
    _XGB_AVAILABLE = True
except ImportError:
    _XGBClass = None        # type: ignore[assignment,misc]
    _XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Registry — maps display name → factory function
# ---------------------------------------------------------------------------

def _make_naive()  -> BaseForecaster: return NaiveModel()
def _make_rf()     -> BaseForecaster: return RandomForestModel()
def _make_arima()  -> BaseForecaster: return _ARIMAClass()   # type: ignore[misc]
def _make_xgb()    -> BaseForecaster: return _XGBClass()     # type: ignore[misc]
def _make_lstm()   -> BaseForecaster: return _LSTMClass()    # type: ignore[misc]


#: Maps every selectable model name to a zero-argument factory.
REGISTRY: dict[str, Any] = {
    "Baseline (Naive)": _make_naive,
    "ARIMA":            _make_arima  if _ARIMA_AVAILABLE  else None,
    "Random Forest":    _make_rf,
    "XGBoost":          _make_xgb    if _XGB_AVAILABLE    else None,
    "LSTM":             _make_lstm   if _LSTM_AVAILABLE    else None,
}


def get_model(name: str) -> BaseForecaster:
    """
    Instantiate a model by its display name.

    Args:
        name : One of the keys in :data:`REGISTRY` (not 'Auto').

    Returns:
        An unfitted :class:`BaseForecaster` subclass instance.

    Raises:
        KeyError       : Unknown model name.
        RuntimeError   : Model's optional dependency not installed.
    """
    if name not in REGISTRY:
        raise KeyError(
            f"Unknown model: '{name}'. "
            f"Available: {list(REGISTRY.keys())}"
        )
    factory = REGISTRY[name]
    if factory is None:
        raise RuntimeError(
            f"Model '{name}' is not available — required dependency missing."
        )
    return factory()


def available_models() -> list[str]:
    """
    Return names of all models whose optional dependencies are installed.

    Returns:
        list[str] — subset of REGISTRY keys where factory is not None.
    """
    return [name for name, factory in REGISTRY.items() if factory is not None]


# ---------------------------------------------------------------------------
# Multi-model runner
# ---------------------------------------------------------------------------

def run_all_models(
    series: pd.Series,
    steps: int,
    val_size: int = 30,
    exclude: list[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    """
    Fit and evaluate every available model; collect predictions from each.

    Args:
        series   : Chronological Close-price Series.
        steps    : Forecast horizon for predictions.
        val_size : Tail observations reserved for validation metrics.
        exclude  : Optional list of model names to skip.

    Returns:
        comparison_df : pd.DataFrame sorted by RMSE (best first) with
                        columns [Model, MAE, RMSE, MAPE, R2, n_test, Status].
        predictions   : dict mapping model_name → list[float] of length
                        ``steps``.  Only models that fitted successfully
                        appear here.
    """
    exclude = exclude or []
    models: list[BaseForecaster] = []

    for name, factory in REGISTRY.items():
        if name in exclude or factory is None:
            continue
        try:
            models.append(factory())
        except Exception:  # noqa: BLE001
            pass

    # evaluate_all handles per-model errors gracefully
    comparison_df = evaluate_all(models, series, val_size=val_size)

    # Collect predictions from fitted models
    predictions: dict[str, list[float]] = {}
    for model in models:
        if model.is_fitted:
            try:
                predictions[model.name] = model.predict(steps)
            except Exception:  # noqa: BLE001
                pass

    return comparison_df, predictions


# ---------------------------------------------------------------------------
# Auto selector
# ---------------------------------------------------------------------------

def auto_select(
    series: pd.Series,
    steps: int,
    val_size: int = 30,
) -> tuple[str, list[float], pd.DataFrame]:
    """
    Evaluate all available models and return the best one's predictions.

    "Best" is defined as lowest validation RMSE among models that
    completed without error.

    Args:
        series   : Chronological Close-price Series.
        steps    : Forecast horizon.
        val_size : Validation split size.

    Returns:
        best_name     : str — display name of the selected model.
        predictions   : list[float] of length ``steps``.
        comparison_df : Full model comparison DataFrame.

    Raises:
        RuntimeError : If no model succeeds.
    """
    comparison_df, predictions = run_all_models(series, steps, val_size)

    chosen = best_model_name(comparison_df)
    if chosen is None or chosen not in predictions:
        raise RuntimeError(
            "Auto selection failed: no model completed successfully. "
            "Check that required dependencies are installed."
        )

    return chosen, predictions[chosen], comparison_df
