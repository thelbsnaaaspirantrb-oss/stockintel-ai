"""
tests/evaluation/test_time_series_validation.py
-------------------------------------------------
Unit tests for evaluation/time_series_validation.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.metrics import MetricSplit
from evaluation.time_series_validation import (
    WalkForwardResult,
    expanding_window_validation,
    walk_forward_validation,
)
from models.naive_model import NaiveModel


def _make_series(n: int = 120, seed: int = 0) -> pd.Series:
    rng = np.random.default_rng(seed)
    prices = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    index = pd.date_range("2020-01-01", periods=n, freq="B")
    return pd.Series(prices, index=index, name="Close")


class RecordingNaive(NaiveModel):
    """Naive model that records the max index seen during each fit."""

    name = "RecordingNaive"

    def __init__(self) -> None:
        super().__init__()
        self.last_fit_series: pd.Series | None = None

    def fit(self, series: pd.Series, val_size: int = 30) -> "RecordingNaive":
        self.last_fit_series = series.copy()
        return super().fit(series, val_size=val_size)


def _recording_factory() -> RecordingNaive:
    return RecordingNaive()


class TestExpandingWindowValidation:
    def test_returns_walk_forward_result_instances(self):
        series = _make_series(80)
        folds = expanding_window_validation(
            NaiveModel,
            series,
            initial_train_size=40,
            validation_window_size=5,
            step_size=5,
        )
        assert len(folds) >= 1
        assert all(isinstance(f, WalkForwardResult) for f in folds)

    def test_fold_boundaries_expanding(self):
        series = _make_series(50)
        initial_train = 20
        val_win = 3
        step = 3
        folds = expanding_window_validation(
            _recording_factory,
            series,
            initial_train_size=initial_train,
            validation_window_size=val_win,
            step_size=step,
        )
        for f in folds:
            assert f.train_start == series.index[0]
            assert f.validation_end <= series.index[-1]
            assert len(f.y_true) == val_win
            assert len(f.y_pred) == val_win
            assert f.metrics.n_samples == val_win

    def test_chronological_fold_order(self):
        series = _make_series(100)
        folds = expanding_window_validation(
            NaiveModel,
            series,
            initial_train_size=30,
            validation_window_size=4,
            step_size=4,
        )
        starts = [f.validation_start for f in folds]
        assert starts == sorted(starts)

    def test_no_future_data_in_fit_slice(self):
        series = _make_series(60)
        model_holder: list[RecordingNaive] = []

        def factory() -> RecordingNaive:
            m = RecordingNaive()
            model_holder.append(m)
            return m

        folds = expanding_window_validation(
            factory,
            series,
            initial_train_size=25,
            validation_window_size=2,
            step_size=2,
        )
        for f, m in zip(folds, model_holder):
            assert m.last_fit_series is not None
            # Fit slice ends at validation_end — no observations after
            assert m.last_fit_series.index[-1] == f.validation_end
            # Training portion within slice excludes validation tail
            assert m.last_fit_series.index[-(2)] <= f.train_end

    def test_metrics_integration(self):
        series = _make_series(70)
        folds = expanding_window_validation(
            NaiveModel,
            series,
            initial_train_size=30,
            validation_window_size=5,
            step_size=5,
        )
        for f in folds:
            assert f.metrics.model_name == "Baseline (Naive)"
            assert f.metrics.split == MetricSplit.VALIDATION
            assert f.metrics.RMSE >= 0

    def test_empty_folds_raises(self):
        series = _make_series(10)
        with pytest.raises(ValueError, match="No expanding-window folds"):
            expanding_window_validation(
                NaiveModel,
                series,
                initial_train_size=8,
                validation_window_size=5,
                step_size=1,
            )

    def test_invalid_parameters(self):
        series = _make_series(30)
        with pytest.raises(ValueError, match="initial_train_size"):
            expanding_window_validation(
                NaiveModel, series, initial_train_size=0,
            )


class TestWalkForwardValidation:
    def test_rolling_train_window_length(self):
        series = _make_series(100)
        initial_train = 25
        val_win = 4
        folds = walk_forward_validation(
            _recording_factory,
            series,
            initial_train_size=initial_train,
            validation_window_size=val_win,
            step_size=4,
        )
        for f in folds:
            # train_end - train_start + 1 == initial_train_size (by index position)
            t0 = series.index.get_loc(f.train_start)
            t1 = series.index.get_loc(f.train_end)
            assert t1 - t0 + 1 == initial_train

    def test_validation_window_size_respected(self):
        series = _make_series(90)
        val_win = 6
        folds = walk_forward_validation(
            NaiveModel,
            series,
            initial_train_size=30,
            validation_window_size=val_win,
            step_size=6,
        )
        for f in folds:
            assert len(f.y_true) == val_win
            assert len(f.y_pred) == val_win

    def test_step_size_advances_validation_start(self):
        series = _make_series(80)
        step = 5
        folds = walk_forward_validation(
            NaiveModel,
            series,
            initial_train_size=20,
            validation_window_size=2,
            step_size=step,
        )
        if len(folds) >= 2:
            p0 = series.index.get_loc(folds[0].validation_start)
            p1 = series.index.get_loc(folds[1].validation_start)
            assert p1 - p0 == step

    def test_y_true_y_pred_alignment(self):
        series = _make_series(75)
        folds = walk_forward_validation(
            NaiveModel,
            series,
            initial_train_size=25,
            validation_window_size=3,
            step_size=3,
        )
        for f in folds:
            assert np.allclose(f.y_true - f.y_pred, f.metrics.y_true - f.metrics.y_pred)

    def test_timestamps_preserved(self):
        series = _make_series(65)
        folds = expanding_window_validation(
            NaiveModel,
            series,
            initial_train_size=20,
            validation_window_size=3,
            step_size=3,
        )
        for f in folds:
            assert f.timestamps is not None
            assert len(f.timestamps) == len(f.y_true)

    def test_no_folds_raises(self):
        series = _make_series(15)
        with pytest.raises(ValueError, match="No walk-forward folds"):
            walk_forward_validation(
                NaiveModel,
                series,
                initial_train_size=12,
                validation_window_size=10,
                step_size=1,
            )
