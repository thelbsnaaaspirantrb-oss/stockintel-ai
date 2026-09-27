"""
evaluation/metrics.py
---------------------
StockIntel AI — Professional Model Evaluation Framework
Author : Rohan Bondre

Provides regression metrics and a split-aware result container.

Metrics
-------
MAE   — Mean Absolute Error
MSE   — Mean Squared Error
RMSE  — Root Mean Squared Error
MAPE  — Mean Absolute Percentage Error  (%)
R²    — Coefficient of Determination

Design
------
- ``compute_metrics()`` is the single authoritative metric function.
  All values are derived purely from the supplied y_true / y_pred arrays.
  No hardcoding, no defaults, no fabrication.
- ``EvaluationResult`` carries the split label (train / validation / test)
  alongside the metric values, so callers always know which data partition
  the numbers come from.
- Every metric guards against numerical edge-cases:
    - Zero denominator in MAPE  → NaN (not ∞).
    - Zero total variance in R² → 0.0 (constant target edge-case).
    - Empty arrays              → raise ValueError immediately.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Split label enum
# ---------------------------------------------------------------------------

class MetricSplit(str, Enum):
    """Labels identifying which data partition metrics were computed on."""
    TRAIN      = "train"
    VALIDATION = "validation"
    TEST       = "test"
    UNKNOWN    = "unknown"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class EvaluationResult:
    """
    Immutable container for one model's metrics on one data split.

    Attributes
    ----------
    model_name : str
        Human-readable model identifier.
    split : MetricSplit
        Which partition (train / validation / test) produced these metrics.
    MAE : float
        Mean Absolute Error (same units as the target).
    MSE : float
        Mean Squared Error.
    RMSE : float
        Root Mean Squared Error (same units as the target).
    MAPE : float
        Mean Absolute Percentage Error (%).  NaN if any y_true == 0.
    R2 : float
        Coefficient of Determination.  1.0 = perfect, 0.0 = mean predictor.
    n_samples : int
        Number of observations used.
    y_true : np.ndarray
        Actual target values (retained for chart generation).
    y_pred : np.ndarray
        Predicted values (retained for chart generation).
    extra : dict
        Optional extra metadata (e.g. fold indices, ARIMA order).
    """
    model_name : str
    split      : MetricSplit
    MAE        : float
    MSE        : float
    RMSE       : float
    MAPE       : float
    R2         : float
    n_samples  : int
    y_true     : np.ndarray
    y_pred     : np.ndarray
    extra      : dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    def residuals(self) -> np.ndarray:
        """Return y_true − y_pred for every observation."""
        return self.y_true - self.y_pred

    def pct_errors(self) -> np.ndarray:
        """
        Return (y_true − y_pred) / y_true × 100 for non-zero actuals.
        Zero-actual rows are replaced with NaN.
        """
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(
                self.y_true != 0,
                (self.y_true - self.y_pred) / self.y_true * 100.0,
                np.nan,
            )
        return out.astype(float)

    def to_dict(self) -> dict[str, Any]:
        """Serialise scalar fields to a plain dict (excludes arrays)."""
        return {
            "model_name": self.model_name,
            "split":      self.split.value,
            "MAE":        self.MAE,
            "MSE":        self.MSE,
            "RMSE":       self.RMSE,
            "MAPE":       self.MAPE,
            "R2":         self.R2,
            "n_samples":  self.n_samples,
        }

    def __repr__(self) -> str:
        return (
            f"EvaluationResult(model={self.model_name!r}, "
            f"split={self.split.value!r}, "
            f"MAE={self.MAE:.4f}, RMSE={self.RMSE:.4f}, "
            f"MAPE={self.MAPE:.2f}%, R²={self.R2:.4f}, "
            f"n={self.n_samples})"
        )


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray | list | pd.Series,
    y_pred: np.ndarray | list | pd.Series,
    *,
    model_name: str = "model",
    split: MetricSplit | str = MetricSplit.UNKNOWN,
    extra: dict | None = None,
) -> EvaluationResult:
    """
    Compute all regression metrics from actual and predicted arrays.

    All metric values are computed exclusively from the supplied data —
    no defaults, estimates, or hardcoded numbers are used.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values (original price scale).
    y_pred : array-like
        Model predictions (original price scale).
    model_name : str
        Identifier for the model (used in reports and chart labels).
    split : MetricSplit | str
        Which data partition produced these predictions.
        Accepts 'train', 'validation', 'test', or a MetricSplit member.
    extra : dict | None
        Optional metadata to attach (e.g. ARIMA order, training epochs).

    Returns
    -------
    EvaluationResult
        Dataclass carrying all five metrics + the raw arrays.

    Raises
    ------
    ValueError
        If ``y_true`` and ``y_pred`` have different lengths, or are empty.

    Notes
    -----
    **MAE**   = mean |y_true − y_pred|

    **MSE**   = mean (y_true − y_pred)²

    **RMSE**  = √MSE

    **MAPE**  = mean |y_true − y_pred| / |y_true| × 100
               (rows where y_true == 0 are excluded from the mean;
               returns NaN if *all* y_true are zero)

    **R²**    = 1 − SS_res / SS_tot
               (returns 0.0 when SS_tot == 0, i.e. constant target)

    Examples
    --------
    >>> result = compute_metrics([100, 110, 90], [102, 108, 92],
    ...                          model_name="RF", split="validation")
    >>> result.RMSE
    2.160246899...
    """
    # ── Input normalisation ───────────────────────────────────────────────
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_pred = np.asarray(y_pred, dtype=float).ravel()

    if len(y_true) == 0:
        raise ValueError("y_true is empty — cannot compute metrics.")
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}."
        )

    # ── Normalise split label ─────────────────────────────────────────────
    if isinstance(split, str):
        try:
            split = MetricSplit(split.lower())
        except ValueError:
            split = MetricSplit.UNKNOWN

    # ── Metrics ───────────────────────────────────────────────────────────
    errors  = y_true - y_pred
    abs_err = np.abs(errors)

    mae  = float(np.mean(abs_err))
    mse  = float(np.mean(errors ** 2))
    rmse = float(math.sqrt(mse))

    # MAPE: exclude zero-actual rows to avoid division by zero
    nonzero_mask = y_true != 0.0
    if nonzero_mask.any():
        mape = float(
            np.mean(abs_err[nonzero_mask] / np.abs(y_true[nonzero_mask])) * 100.0
        )
    else:
        mape = float("nan")

    # R²
    ss_res = float(np.sum(errors ** 2))
    ss_tot = float(np.sum((y_true - float(np.mean(y_true))) ** 2))
    r2     = (1.0 - ss_res / ss_tot) if ss_tot > 0.0 else 0.0

    return EvaluationResult(
        model_name = model_name,
        split      = split,
        MAE        = mae,
        MSE        = mse,
        RMSE       = rmse,
        MAPE       = mape,
        R2         = r2,
        n_samples  = int(len(y_true)),
        y_true     = y_true.copy(),
        y_pred     = y_pred.copy(),
        extra      = extra or {},
    )


# ---------------------------------------------------------------------------
# Convenience: aggregate multiple fold results
# ---------------------------------------------------------------------------

def aggregate_fold_results(results: list[EvaluationResult]) -> dict[str, float]:
    """
    Average metrics across a list of fold results (walk-forward folds).

    Parameters
    ----------
    results : list[EvaluationResult]
        One entry per fold, all from the same model and split.

    Returns
    -------
    dict with keys MAE, MSE, RMSE, MAPE, R2 — each the mean across folds.
    NaN fold values are excluded from the mean.

    Raises
    ------
    ValueError : If ``results`` is empty.
    """
    if not results:
        raise ValueError("Cannot aggregate an empty list of results.")

    def _mean(key: str) -> float:
        vals = [getattr(r, key) for r in results if not math.isnan(getattr(r, key))]
        return float(np.mean(vals)) if vals else float("nan")

    return {
        "MAE":  _mean("MAE"),
        "MSE":  _mean("MSE"),
        "RMSE": _mean("RMSE"),
        "MAPE": _mean("MAPE"),
        "R2":   _mean("R2"),
    }
