"""
evaluation/time_series_validation.py
------------------------------------
Walk-forward and expanding-window validation for chronological time series.

All splits preserve temporal order.  Training folds never include observations
that occur after the validation window being scored.

Models are supplied via a zero-argument factory returning a
:class:`models.base_model.BaseForecaster` instance.  Each fold calls
``fit`` on the observations available up to the end of that validation
window, passing the validation size so the model withholds the tail
internally.  Validation predictions are read from the model's stored
``_val_y_true`` / ``_val_y_pred`` arrays — no training logic is duplicated
here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np
import pandas as pd

from evaluation.metrics import EvaluationResult, MetricSplit, compute_metrics

if TYPE_CHECKING:
    from models.base_model import BaseForecaster

ModelFactory = Callable[[], "BaseForecaster"]


# ---------------------------------------------------------------------------
# Fold result
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardResult:
    """
    Outcome of one chronological validation fold.

    Attributes
    ----------
    fold_number : int
        1-based fold index.
    train_start, train_end : Any
        Index labels (usually timestamps) bounding the training region.
        ``train_end`` is inclusive.
    validation_start, validation_end : Any
        Index labels bounding the validation region (inclusive).
    y_true, y_pred : np.ndarray
        Actual and predicted values on the validation window (price scale).
    metrics : EvaluationResult
        Metrics computed via :func:`evaluation.metrics.compute_metrics`.
    timestamps : np.ndarray | None
        Validation index values when the input series has an index.
    extra : dict
        Optional metadata (errors, timing, etc.).
    """
    fold_number: int
    train_start: Any
    train_end: Any
    validation_start: Any
    validation_end: Any
    y_true: np.ndarray
    y_pred: np.ndarray
    metrics: EvaluationResult
    timestamps: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Input helpers
# ---------------------------------------------------------------------------

def _as_series(series: pd.Series | np.ndarray | list) -> pd.Series:
    """Normalise input to a chronological ``pd.Series``."""
    if isinstance(series, pd.Series):
        s = series.copy()
    else:
        arr = np.asarray(series, dtype=float).ravel()
        s = pd.Series(arr, index=pd.RangeIndex(len(arr)))
    if not np.issubdtype(s.dtype, np.number):
        try:
            s = s.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError("Series must contain numeric observations.") from exc
    if s.isna().any():
        raise ValueError("Series contains NaN values — drop them before validation.")
    if not np.isfinite(s.to_numpy(dtype=float)).all():
        raise ValueError("Series contains non-finite values.")
    if len(s) < 2:
        raise ValueError("Series must contain at least 2 observations.")
    return s


def _index_label(series: pd.Series, pos: int) -> Any:
    """Safe index label for integer position ``pos``."""
    return series.index[pos]


def _validation_timestamps(series: pd.Series, v_start: int, v_end: int) -> np.ndarray:
    return np.asarray(series.index[v_start : v_end + 1])


def _iter_expanding_folds(
    n: int,
    *,
    initial_train_size: int,
    validation_window_size: int,
    step_size: int,
) -> Iterator[tuple[int, int, int]]:
    """
    Yield ``(train_end_pos, v_start, v_end)`` for expanding-window folds.

    Training always starts at index 0.  ``train_end_pos`` is the last training
    index (inclusive).  Validation spans ``[v_start, v_end]`` inclusive.
    """
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be >= 1.")
    if validation_window_size < 1:
        raise ValueError("validation_window_size must be >= 1.")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")

    v_start = initial_train_size
    while True:
        v_end = v_start + validation_window_size - 1
        if v_end >= n:
            break
        train_end_pos = v_start - 1
        yield train_end_pos, v_start, v_end
        v_start += step_size


def _iter_walk_forward_folds(
    n: int,
    *,
    initial_train_size: int,
    validation_window_size: int,
    step_size: int,
) -> Iterator[tuple[int, int, int, int]]:
    """
    Yield ``(train_start, train_end_pos, v_start, v_end)`` for rolling folds.

    The training window length is fixed at ``initial_train_size`` observations
    (except when ``train_start`` would be negative — skipped).
    """
    if initial_train_size < 1:
        raise ValueError("initial_train_size must be >= 1.")
    if validation_window_size < 1:
        raise ValueError("validation_window_size must be >= 1.")
    if step_size < 1:
        raise ValueError("step_size must be >= 1.")

    v_start = initial_train_size
    while True:
        v_end = v_start + validation_window_size - 1
        if v_end >= n:
            break
        train_start = v_start - initial_train_size
        train_end_pos = v_start - 1
        if train_start < 0:
            v_start += step_size
            continue
        yield train_start, train_end_pos, v_start, v_end
        v_start += step_size


def _run_fold(
    model_factory: ModelFactory,
    series: pd.Series,
    *,
    fold_number: int,
    slice_start: int,
    v_start: int,
    v_end: int,
    train_start: int,
    train_end_pos: int,
    split: MetricSplit,
) -> WalkForwardResult:
    """
    Fit on ``series.iloc[slice_start : v_end + 1]`` with tail validation.

    The last ``(v_end - v_start + 1)`` rows of the slice are the validation
    set passed to ``fit(..., val_size=...)``.
    """
    val_size = v_end - v_start + 1
    fold_series = series.iloc[slice_start : v_end + 1]
    if len(fold_series) <= val_size:
        raise ValueError(
            f"Fold {fold_number}: slice length {len(fold_series)} "
            f"is not greater than val_size {val_size}."
        )

    try:
        model = model_factory()
    except Exception as exc:
        raise RuntimeError(
            f"Fold {fold_number}: model factory failed."
        ) from exc

    if model is None:
        raise TypeError(f"Fold {fold_number}: model factory returned None.")

    try:
        model.fit(fold_series, val_size=val_size)
        y_true_raw = model._val_y_true
        y_pred_raw = model._val_y_pred
        model_name = model.name
    except AttributeError as exc:
        raise TypeError(
            f"Fold {fold_number}: model must expose name, _val_y_true, "
            "and _val_y_pred after fit()."
        ) from exc

    y_true = np.asarray(y_true_raw, dtype=float).ravel()
    y_pred = np.asarray(y_pred_raw, dtype=float).ravel()
    if len(y_true) != val_size or len(y_pred) != val_size:
        raise RuntimeError(
            f"Fold {fold_number}: model {model_name!r} returned "
            f"{len(y_true)} validation labels, expected {val_size}."
        )
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise RuntimeError(
            f"Fold {fold_number}: model returned non-finite validation values."
        )

    metrics = compute_metrics(
        y_true,
        y_pred,
        model_name=model_name,
        split=split,
        extra={"fold_number": fold_number},
    )

    return WalkForwardResult(
        fold_number=fold_number,
        train_start=_index_label(series, train_start),
        train_end=_index_label(series, train_end_pos),
        validation_start=_index_label(series, v_start),
        validation_end=_index_label(series, v_end),
        y_true=y_true,
        y_pred=y_pred,
        metrics=metrics,
        timestamps=_validation_timestamps(series, v_start, v_end),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def expanding_window_validation(
    model_factory: ModelFactory,
    series: pd.Series | np.ndarray | list,
    *,
    initial_train_size: int,
    validation_window_size: int = 1,
    step_size: int = 1,
    split: MetricSplit | str = MetricSplit.VALIDATION,
) -> list[WalkForwardResult]:
    """
    Expanding-window validation: training always starts at the first observation.

    For each fold, the model is fit on ``series[0 : validation_end]`` with the
    final ``validation_window_size`` observations withheld as validation.  The
    validation window advances by ``step_size`` until it would extend past the
    end of the series.

    Parameters
    ----------
    model_factory : callable
        Zero-argument factory returning a fresh :class:`BaseForecaster`.
    series : array-like
        Chronological target series (typically Close prices).
    initial_train_size : int
        Number of observations in the initial training segment before the
        first validation window begins.
    validation_window_size : int
        Length of each validation window (default 1).
    step_size : int
        Advance of the validation window between folds (default 1).
    split : MetricSplit or str
        Label stored on each fold's :class:`EvaluationResult`.

    Returns
    -------
    list[WalkForwardResult]
        One entry per fold, in chronological order.

    Raises
    ------
    ValueError
        If parameters are invalid or no folds can be constructed.
    """
    series = _as_series(series)
    n = len(series)

    folds = list(
        _iter_expanding_folds(
            n,
            initial_train_size=initial_train_size,
            validation_window_size=validation_window_size,
            step_size=step_size,
        )
    )
    if not folds:
        raise ValueError(
            "No expanding-window folds fit in the series. "
            "Reduce initial_train_size or validation_window_size."
        )

    results: list[WalkForwardResult] = []
    for fold_number, (train_end_pos, v_start, v_end) in enumerate(folds, start=1):
        results.append(
            _run_fold(
                model_factory,
                series,
                fold_number=fold_number,
                slice_start=0,
                v_start=v_start,
                v_end=v_end,
                train_start=0,
                train_end_pos=train_end_pos,
                split=split,
            )
        )
    return results


def walk_forward_validation(
    model_factory: ModelFactory,
    series: pd.Series | np.ndarray | list,
    *,
    initial_train_size: int,
    validation_window_size: int = 1,
    step_size: int = 1,
    split: MetricSplit | str = MetricSplit.VALIDATION,
) -> list[WalkForwardResult]:
    """
    Rolling walk-forward validation with a fixed-length training window.

    Each fold trains on exactly ``initial_train_size`` consecutive observations
    immediately preceding the validation window, then validates on the next
    ``validation_window_size`` observations.  The window advances by
    ``step_size``.

    Parameters
    ----------
    model_factory : callable
        Zero-argument factory returning a fresh :class:`BaseForecaster`.
    series : array-like
        Chronological target series.
    initial_train_size : int
        Fixed training-window length (number of observations).
    validation_window_size : int
        Validation window length (default 1).
    step_size : int
        Step between successive validation windows (default 1).
    split : MetricSplit or str
        Label stored on each fold's :class:`EvaluationResult`.

    Returns
    -------
    list[WalkForwardResult]
        One entry per fold, in chronological order.

    Raises
    ------
    ValueError
        If parameters are invalid or no folds can be constructed.
    """
    series = _as_series(series)
    n = len(series)

    folds = list(
        _iter_walk_forward_folds(
            n,
            initial_train_size=initial_train_size,
            validation_window_size=validation_window_size,
            step_size=step_size,
        )
    )
    if not folds:
        raise ValueError(
            "No walk-forward folds fit in the series. "
            "Adjust initial_train_size, validation_window_size, or step_size."
        )

    results: list[WalkForwardResult] = []
    for fold_number, (train_start, train_end_pos, v_start, v_end) in enumerate(
        folds, start=1
    ):
        results.append(
            _run_fold(
                model_factory,
                series,
                fold_number=fold_number,
                slice_start=train_start,
                v_start=v_start,
                v_end=v_end,
                train_start=train_start,
                train_end_pos=train_end_pos,
                split=split,
            )
        )
    return results
