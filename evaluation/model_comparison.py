"""
evaluation/model_comparison.py
------------------------------
Multi-model evaluation and comparison using the project's ``BaseForecaster``
implementations.

Validation metrics are used for model comparison.  Optional test-set evaluation
is reported separately and is never used to rank or select models.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import numpy as np
import pandas as pd

from evaluation.metrics import (
    EvaluationResult,
    MetricSplit,
    aggregate_fold_results,
    compute_metrics,
)
from evaluation.time_series_validation import (
    WalkForwardResult,
    expanding_window_validation,
    walk_forward_validation,
)

from models.base_model import BaseForecaster


def _registry():
    """Lazy import so ``import evaluation`` does not load optional ML backends."""
    from models.model_registry import REGISTRY, available_models, get_model
    return REGISTRY, available_models, get_model


class EvaluationType(str, Enum):
    """Kind of evaluation performed."""
    VALIDATION = "validation"
    TEST = "test"
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"


@dataclass
class ModelEvaluationResult:
    """
    Structured outcome for one model on one evaluation protocol.

    Scalar metrics (``MAE`` … ``R2``) refer to the combined validation or test
    window unless ``number_of_folds`` > 1, in which case they are the mean of
    fold-level metrics from :func:`aggregate_fold_results`.
    """
    model_name: str
    evaluation_type: EvaluationType
    MAE: float
    MSE: float
    RMSE: float
    MAPE: float
    R2: float
    y_true: np.ndarray
    y_pred: np.ndarray
    timestamps: np.ndarray | None = None
    fold_results: list[WalkForwardResult] = field(default_factory=list)
    number_of_folds: int = 1
    training_time: float | None = None
    prediction_time: float | None = None
    status: str = "OK"
    error_message: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def split(self) -> MetricSplit:
        """Metric split label aligned with ``evaluation_type``."""
        if self.evaluation_type == EvaluationType.TEST:
            return MetricSplit.TEST
        return MetricSplit.VALIDATION

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "evaluation_type": self.evaluation_type.value,
            "MAE": self.MAE,
            "MSE": self.MSE,
            "RMSE": self.RMSE,
            "MAPE": self.MAPE,
            "R2": self.R2,
            "number_of_folds": self.number_of_folds,
            "training_time": self.training_time,
            "prediction_time": self.prediction_time,
            "status": self.status,
            "error_message": self.error_message,
        }


def _factory_for_name(name: str):
    """Return a zero-arg factory that builds a registry model by display name."""
    def _factory() -> BaseForecaster:
        _, _, get_model = _registry()
        return get_model(name)
    return _factory


def _resolve_model_names(
    model_names: list[str] | None,
    *,
    skip_unavailable: bool,
) -> list[str]:
    REGISTRY, available_models, _ = _registry()

    if model_names is None:
        names = available_models()
    else:
        names = list(model_names)

    resolved: list[str] = []
    for name in names:
        if name not in REGISTRY:
            if skip_unavailable:
                continue
            raise KeyError(f"Unknown model: {name!r}.")
        if REGISTRY[name] is None:
            if skip_unavailable:
                continue
            raise RuntimeError(
                f"Model {name!r} is not available — missing dependency."
            )
        resolved.append(name)
    return resolved


def _single_split_validation(
    model: BaseForecaster,
    series: pd.Series,
    val_size: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit with chronological tail validation; return val arrays and train time."""
    t0 = time.perf_counter()
    model.fit(series, val_size=val_size)
    train_time = time.perf_counter() - t0
    y_true = np.asarray(model._val_y_true, dtype=float)
    y_pred = np.asarray(model._val_y_pred, dtype=float)
    return y_true, y_pred, train_time


def _test_evaluation(
    model: BaseForecaster,
    series: pd.Series,
    val_size: int,
    test_size: int,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """
    Fit on ``series[:-test_size]``, then multi-step ``predict(test_size)``.

    Returns test arrays, training time, and prediction time.
    """
    if test_size < 1:
        raise ValueError("test_size must be >= 1 for test evaluation.")
    if len(series) <= test_size + val_size:
        raise ValueError(
            "Series too short for test evaluation with the given "
            f"val_size={val_size} and test_size={test_size}."
        )

    train_val = series.iloc[:-test_size]
    test_series = series.iloc[-test_size:]

    t0 = time.perf_counter()
    model.fit(train_val, val_size=val_size)
    train_time = time.perf_counter() - t0

    t1 = time.perf_counter()
    preds = model.predict(steps=test_size)
    pred_time = time.perf_counter() - t1

    y_true = test_series.to_numpy(dtype=float)
    y_pred = np.asarray(preds, dtype=float)
    if len(y_pred) != test_size:
        y_pred = y_pred[:test_size]
    return y_true, y_pred, train_time, pred_time


def _metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    model_name: str,
    split: MetricSplit,
) -> EvaluationResult:
    return compute_metrics(
        y_true, y_pred, model_name=model_name, split=split,
    )


def _failed_result(
    model_name: str,
    evaluation_type: EvaluationType,
    exc: Exception,
) -> ModelEvaluationResult:
    return ModelEvaluationResult(
        model_name=model_name,
        evaluation_type=evaluation_type,
        MAE=float("nan"),
        MSE=float("nan"),
        RMSE=float("nan"),
        MAPE=float("nan"),
        R2=float("nan"),
        y_true=np.array([], dtype=float),
        y_pred=np.array([], dtype=float),
        status="Error",
        error_message=str(exc),
    )


def evaluate_model(
    model_name: str,
    series: pd.Series | np.ndarray | list,
    *,
    val_size: int = 30,
    test_size: int = 0,
    include_test: bool = False,
    walk_forward: bool = False,
    expanding_window: bool = False,
    initial_train_size: int | None = None,
    validation_window_size: int = 1,
    step_size: int = 1,
) -> list[ModelEvaluationResult]:
    """
    Evaluate one registry model on a chronological series.

    Parameters
    ----------
    model_name : str
        Display name registered in ``models.model_registry.REGISTRY``.
    series : array-like
        Chronological Close-price series.
    val_size : int
        Tail validation size for single-split evaluation (default 30).
    test_size : int
        Optional held-out test observations at the series end.
    include_test : bool
        If True and ``test_size > 0``, append a separate TEST result.
        Test metrics are never mixed into validation ranking.
    walk_forward : bool
        Use rolling walk-forward validation instead of a single tail split.
    expanding_window : bool
        Use expanding-window validation (mutually exclusive with walk_forward).
    initial_train_size : int | None
        Required when ``walk_forward`` or ``expanding_window`` is True.
        For single-split mode, defaults to ``len(series) - val_size - test_size``
        when omitted (must remain >= 1).
    validation_window_size, step_size : int
        Fold construction parameters (multi-fold modes).

    Returns
    -------
    list[ModelEvaluationResult]
        One validation result, plus an optional test result.
    """
    if walk_forward and expanding_window:
        raise ValueError("Specify at most one of walk_forward or expanding_window.")

    if isinstance(series, pd.Series):
        s = series.copy()
    else:
        s = pd.Series(np.asarray(series, dtype=float).ravel())

    if s.isna().any():
        raise ValueError("Series contains NaN values.")
    if len(s) < val_size + 1:
        raise ValueError(
            f"Series length {len(s)} must exceed val_size={val_size}."
        )

    results: list[ModelEvaluationResult] = []

    try:
        factory = _factory_for_name(model_name)

        if walk_forward or expanding_window:
            if initial_train_size is None:
                raise ValueError(
                    "initial_train_size is required for multi-fold validation."
                )
            t0 = time.perf_counter()
            if walk_forward:
                folds = walk_forward_validation(
                    factory,
                    s,
                    initial_train_size=initial_train_size,
                    validation_window_size=validation_window_size,
                    step_size=step_size,
                )
                eval_type = EvaluationType.WALK_FORWARD
            else:
                folds = expanding_window_validation(
                    factory,
                    s,
                    initial_train_size=initial_train_size,
                    validation_window_size=validation_window_size,
                    step_size=step_size,
                )
                eval_type = EvaluationType.EXPANDING_WINDOW
            train_time = time.perf_counter() - t0

            y_true = np.concatenate([f.y_true for f in folds])
            y_pred = np.concatenate([f.y_pred for f in folds])
            ts = np.concatenate(
                [f.timestamps for f in folds if f.timestamps is not None]
            ) if folds else None

            fold_metrics = [f.metrics for f in folds]
            agg = aggregate_fold_results(fold_metrics)

            results.append(
                ModelEvaluationResult(
                    model_name=model_name,
                    evaluation_type=eval_type,
                    MAE=agg["MAE"],
                    MSE=agg["MSE"],
                    RMSE=agg["RMSE"],
                    MAPE=agg["MAPE"],
                    R2=agg["R2"],
                    y_true=y_true,
                    y_pred=y_pred,
                    timestamps=ts,
                    fold_results=folds,
                    number_of_folds=len(folds),
                    training_time=train_time,
                    status="OK",
                )
            )
        else:
            eval_series = s.iloc[:-test_size] if test_size > 0 else s
            if len(eval_series) <= val_size:
                raise ValueError(
                    "Not enough data for validation after reserving test_size."
                )

            model = factory()
            y_true, y_pred, train_time = _single_split_validation(
                model, eval_series, val_size,
            )
            ev = _metrics_from_arrays(
                y_true, y_pred,
                model_name=model_name,
                split=MetricSplit.VALIDATION,
            )
            ts = (
                np.asarray(eval_series.index[-val_size:])
                if hasattr(eval_series.index, "__getitem__")
                else None
            )
            results.append(
                ModelEvaluationResult(
                    model_name=model_name,
                    evaluation_type=EvaluationType.VALIDATION,
                    MAE=ev.MAE,
                    MSE=ev.MSE,
                    RMSE=ev.RMSE,
                    MAPE=ev.MAPE,
                    R2=ev.R2,
                    y_true=y_true,
                    y_pred=y_pred,
                    timestamps=ts,
                    number_of_folds=1,
                    training_time=train_time,
                    status="OK",
                )
            )

        if include_test and test_size > 0:
            model = factory()
            y_true, y_pred, train_time, pred_time = _test_evaluation(
                model, s, val_size, test_size,
            )
            ev = _metrics_from_arrays(
                y_true, y_pred,
                model_name=model_name,
                split=MetricSplit.TEST,
            )
            ts = np.asarray(s.index[-test_size:])
            results.append(
                ModelEvaluationResult(
                    model_name=model_name,
                    evaluation_type=EvaluationType.TEST,
                    MAE=ev.MAE,
                    MSE=ev.MSE,
                    RMSE=ev.RMSE,
                    MAPE=ev.MAPE,
                    R2=ev.R2,
                    y_true=y_true,
                    y_pred=y_pred,
                    timestamps=ts,
                    number_of_folds=1,
                    training_time=train_time,
                    prediction_time=pred_time,
                    status="OK",
                )
            )

    except Exception as exc:  # noqa: BLE001
        results.append(
            _failed_result(
                model_name,
                EvaluationType.VALIDATION,
                exc,
            )
        )

    return results


def compare_models(
    series: pd.Series | np.ndarray | list,
    *,
    model_names: list[str] | None = None,
    val_size: int = 30,
    test_size: int = 0,
    include_test: bool = False,
    skip_unavailable: bool = True,
    walk_forward: bool = False,
    expanding_window: bool = False,
    initial_train_size: int | None = None,
    validation_window_size: int = 1,
    step_size: int = 1,
) -> "ModelComparisonReport":
    """
    Evaluate multiple models and build a comparison report.

    Models are ranked by validation RMSE (ascending).  Test results, when
    requested, are stored separately and do not affect ranking.
    """
    names = _resolve_model_names(model_names, skip_unavailable=skip_unavailable)
    if not names:
        raise ValueError("No models available to compare.")

    validation_results: list[ModelEvaluationResult] = []
    test_results: list[ModelEvaluationResult] = []

    for name in names:
        evals = evaluate_model(
            name,
            series,
            val_size=val_size,
            test_size=test_size,
            include_test=include_test,
            walk_forward=walk_forward,
            expanding_window=expanding_window,
            initial_train_size=initial_train_size,
            validation_window_size=validation_window_size,
            step_size=step_size,
        )
        for res in evals:
            if res.evaluation_type == EvaluationType.TEST:
                test_results.append(res)
            else:
                validation_results.append(res)

    return ModelComparisonReport(
        validation_results=validation_results,
        test_results=test_results,
    )


class ModelComparisonReport:
    """
    Container for multi-model evaluation outcomes.

    Parameters
    ----------
    validation_results : list[ModelEvaluationResult]
        Results used for ranking and the main comparison table.
    test_results : list[ModelEvaluationResult]
        Optional held-out test evaluations (not used for ranking).
    """

    def __init__(
        self,
        validation_results: list[ModelEvaluationResult] | None = None,
        test_results: list[ModelEvaluationResult] | None = None,
        *,
        results: list[ModelEvaluationResult] | None = None,
    ) -> None:
        if results is not None:
            validation_results = [
                r for r in results
                if r.evaluation_type != EvaluationType.TEST
            ]
            test_results = [
                r for r in results
                if r.evaluation_type == EvaluationType.TEST
            ]
        self.validation_results = validation_results or []
        self.test_results = test_results or []
        self._by_name: dict[str, ModelEvaluationResult] = {
            r.model_name: r for r in self.validation_results
        }

    def get_model_results(self, model_name: str) -> ModelEvaluationResult | None:
        """Return the validation (or walk-forward) result for ``model_name``."""
        return self._by_name.get(model_name)

    def comparison_table(
        self,
        *,
        sort_by: str = "RMSE",
        ascending: bool = True,
    ) -> pd.DataFrame:
        """
        Build a comparison table from validation results only.

        Columns: Model, MAE, MSE, RMSE, MAPE, R2, n_samples, Status.
        """
        rows: list[dict[str, Any]] = []
        for res in self.validation_results:
            rows.append({
                "Model": res.model_name,
                "MAE": res.MAE,
                "MSE": res.MSE,
                "RMSE": res.RMSE,
                "MAPE": res.MAPE,
                "R2": res.R2,
                "n_samples": int(len(res.y_true)),
                "number_of_folds": res.number_of_folds,
                "Status": res.status,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        ok = df["Status"] == "OK"
        if ok.any() and sort_by in df.columns:
            df_ok = df[ok].sort_values(sort_by, ascending=ascending)
            df_bad = df[~ok]
            df = pd.concat([df_ok, df_bad], ignore_index=True)
        return df.reset_index(drop=True)

    def test_comparison_table(self) -> pd.DataFrame:
        """Comparison table for held-out test evaluations (if any)."""
        temp = ModelComparisonReport(validation_results=self.test_results)
        return temp.comparison_table()
