"""
models/evaluator.py
-------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Model evaluation utilities.

Provides
--------
- TimeSeriesSplit helper (chronological, never random).
- compute_metrics()  — MAE, RMSE, MAPE, R².
- evaluate_all()     — runs every registered model, returns ranked DataFrame.
- format_metrics()   — pretty-prints a metrics dict for display.

Rules
-----
- Validation always uses the chronological tail of the series.
- Training data never contains observations that come after any validation
  observation (no look-ahead bias).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from models.base_model import BaseForecaster


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray | list,
    y_pred: np.ndarray | list,
) -> dict[str, float]:
    """
    Compute MAE, RMSE, MAPE, and R² between two equal-length arrays.

    All metrics are computed in original price scale (no log transform).

    Args:
        y_true : Actual prices.
        y_pred : Predicted prices.

    Returns:
        dict with keys MAE, RMSE, MAPE, R2, n_test.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if len(y_true) != len(y_pred):
        raise ValueError(
            f"y_true length {len(y_true)} != y_pred length {len(y_pred)}"
        )

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero = y_true != 0
    mape = (
        float(
            np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]))
            * 100
        )
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


# ---------------------------------------------------------------------------
# Multi-model evaluation
# ---------------------------------------------------------------------------

def evaluate_all(
    models: list["BaseForecaster"],
    series: pd.Series,
    val_size: int = 30,
) -> pd.DataFrame:
    """
    Fit and evaluate every model in ``models`` on the same series.

    Uses chronological train/validation split: the last ``val_size``
    observations are the held-out set for every model.

    Args:
        models   : List of BaseForecaster instances (unfitted).
        series   : pd.Series of Close prices, DatetimeIndex, chronological.
        val_size : Number of tail observations reserved for validation.

    Returns:
        pd.DataFrame with columns [Model, MAE, RMSE, MAPE, R2, n_test,
        Status].  Rows are sorted by RMSE ascending (best first).
        If a model raises an exception, Status contains the error message
        and metric columns contain NaN.
    """
    rows: list[dict[str, Any]] = []

    for model in models:
        row: dict[str, Any] = {"Model": model.name}
        try:
            model.fit(series, val_size=val_size)
            metrics = model.evaluate()
            row.update(metrics)
            row["Status"] = "OK"
        except Exception as exc:  # noqa: BLE001
            row.update({"MAE": np.nan, "RMSE": np.nan,
                        "MAPE": np.nan, "R2": np.nan, "n_test": 0})
            row["Status"] = f"Error: {exc}"

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by RMSE ascending; models with errors sink to the bottom
    df = df.sort_values("RMSE", ascending=True, na_position="last")
    df = df.reset_index(drop=True)
    return df


def best_model_name(comparison_df: pd.DataFrame) -> str | None:
    """
    Return the name of the model with the lowest RMSE that completed without error.

    Args:
        comparison_df : Output of :func:`evaluate_all`.

    Returns:
        str model name, or None if no model succeeded.
    """
    ok = comparison_df[comparison_df["Status"] == "OK"]
    if ok.empty:
        return None
    return str(ok.iloc[0]["Model"])


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_metrics(metrics: dict[str, Any], currency: str = "$") -> dict[str, str]:
    """
    Format a metrics dict for human-readable display.

    Args:
        metrics  : Dict with MAE, RMSE, MAPE, R2, n_test keys.
        currency : Currency symbol prefix for MAE / RMSE.

    Returns:
        Dict with the same keys but formatted string values.
    """
    def _f(v: Any, fmt: str = ".4f") -> str:
        try:
            return format(float(v), fmt)
        except (TypeError, ValueError):
            return "N/A"

    return {
        "MAE":    f"{currency}{_f(metrics.get('MAE'), '.4f')}",
        "RMSE":   f"{currency}{_f(metrics.get('RMSE'), '.4f')}",
        "MAPE":   f"{_f(metrics.get('MAPE'), '.2f')}%",
        "R2":     _f(metrics.get("R2"), ".4f"),
        "n_test": str(metrics.get("n_test", "N/A")),
    }
