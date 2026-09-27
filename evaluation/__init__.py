"""
evaluation
----------
StockIntel AI — Professional Model Evaluation Framework
Author : Rohan Bondre

Packages
--------
metrics.py
    Regression metrics (MAE, MSE, RMSE, MAPE, R²) and the
    split-aware ``EvaluationResult`` dataclass.

time_series_validation.py
    Walk-forward validation and expanding-window validation
    for time-series forecasting models.

model_comparison.py
    Multi-model comparison table and four diagnostic Plotly charts:
    Actual vs Predicted, Residuals over time, Error distribution,
    and a multi-model RMSE bar chart.

Quick start
-----------
>>> from evaluation.metrics import compute_metrics, EvaluationResult
>>> from evaluation.time_series_validation import walk_forward_validation
>>> from evaluation.model_comparison import ModelComparisonReport

>>> result = compute_metrics(y_true, y_pred, split="validation")
>>> wf     = walk_forward_validation(model_factory, series)
>>> report = ModelComparisonReport(results).comparison_table()
"""

from evaluation.metrics import (
    compute_metrics,
    EvaluationResult,
    MetricSplit,
)
from evaluation.time_series_validation import (
    walk_forward_validation,
    expanding_window_validation,
    WalkForwardResult,
)
from evaluation.model_comparison import (
    ModelComparisonReport,
    ModelEvaluationResult,
    EvaluationType,
    compare_models,
    evaluate_model,
)

__all__ = [
    "compute_metrics",
    "EvaluationResult",
    "MetricSplit",
    "walk_forward_validation",
    "expanding_window_validation",
    "WalkForwardResult",
    "ModelComparisonReport",
    "ModelEvaluationResult",
    "EvaluationType",
    "compare_models",
    "evaluate_model",
]
