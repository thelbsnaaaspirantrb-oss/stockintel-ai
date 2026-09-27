"""
models
------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Multi-model forecasting engine package.

Modules
-------
base_model          — Abstract BaseForecaster interface (fit/predict/evaluate).
feature_engineering — Tabular feature builder for ML models.
evaluator           — Metric computation + multi-model evaluation runner.
naive_model         — Persistence baseline.
arima_model         — ARIMA(p,d,q) via statsmodels.
random_forest_model — Random Forest Regressor.
xgboost_model       — XGBoost Regressor.
lstm_model          — Pre-trained 2-layer LSTM wrapper.
model_registry      — Factory, auto-selection, and multi-model runner.

Quick start
-----------
>>> from models.model_registry import get_model, run_all_models, auto_select
>>> model = get_model("Random Forest")
>>> model.fit(close_series)
>>> predictions = model.predict(steps=7)
>>> metrics = model.evaluate()

>>> # Run all models and compare:
>>> comparison_df, preds = run_all_models(close_series, steps=7)

>>> # Auto-select best model:
>>> name, preds, comparison_df = auto_select(close_series, steps=7)
"""

from models.base_model          import BaseForecaster
from models.evaluator           import compute_metrics, evaluate_all, best_model_name
from models.model_registry      import (
    REGISTRY,
    get_model,
    available_models,
    run_all_models,
    auto_select,
)

__all__ = [
    "BaseForecaster",
    "compute_metrics",
    "evaluate_all",
    "best_model_name",
    "REGISTRY",
    "get_model",
    "available_models",
    "run_all_models",
    "auto_select",
]
