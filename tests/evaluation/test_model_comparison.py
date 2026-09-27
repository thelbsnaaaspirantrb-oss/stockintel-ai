"""
tests/evaluation/test_model_comparison.py
-------------------------------------------
Unit tests for evaluation/model_comparison.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.model_comparison import (
    EvaluationType,
    ModelComparisonReport,
    compare_models,
    evaluate_model,
)
from tests.conftest import make_price_series


class TestEvaluateModel:
    def test_naive_validation_metrics(self):
        series = make_price_series(n=200, seed=1)
        results = evaluate_model("Baseline (Naive)", series, val_size=20)
        assert len(results) == 1
        res = results[0]
        assert res.status == "OK"
        assert res.evaluation_type == EvaluationType.VALIDATION
        assert res.number_of_folds == 1
        assert len(res.y_true) == 20
        assert len(res.y_pred) == 20
        assert res.RMSE >= 0
        assert res.training_time is not None

    def test_random_forest_validation(self):
        series = make_price_series(n=250, seed=2)
        results = evaluate_model("Random Forest", series, val_size=25)
        assert results[0].status == "OK"
        assert results[0].model_name == "Random Forest"

    def test_include_test_separate_result(self):
        series = make_price_series(n=220, seed=3)
        results = evaluate_model(
            "Baseline (Naive)",
            series,
            val_size=20,
            test_size=15,
            include_test=True,
        )
        assert len(results) == 2
        assert results[0].evaluation_type == EvaluationType.VALIDATION
        assert results[1].evaluation_type == EvaluationType.TEST
        assert len(results[1].y_true) == 15
        assert results[1].prediction_time is not None

    def test_walk_forward_mode(self):
        series = make_price_series(n=180, seed=4)
        results = evaluate_model(
            "Baseline (Naive)",
            series,
            walk_forward=True,
            initial_train_size=50,
            validation_window_size=5,
            step_size=5,
        )
        assert results[0].status == "OK"
        assert results[0].number_of_folds >= 1
        assert len(results[0].fold_results) == results[0].number_of_folds

    def test_unknown_model_error_result(self):
        series = make_price_series(n=100, seed=5)
        results = evaluate_model("NoSuchModel", series, val_size=10)
        assert results[0].status == "Error"
        assert np.isnan(results[0].RMSE)

    def test_empty_series_raises(self):
        with pytest.raises(ValueError):
            evaluate_model("Baseline (Naive)", [], val_size=5)

    def test_series_too_short_raises(self):
        series = make_price_series(n=10, seed=6)
        with pytest.raises(ValueError, match="must exceed val_size"):
            evaluate_model("Baseline (Naive)", series, val_size=20)

    def test_mutually_exclusive_fold_modes(self):
        series = make_price_series(n=100, seed=7)
        with pytest.raises(ValueError, match="walk_forward"):
            evaluate_model(
                "Baseline (Naive)",
                series,
                walk_forward=True,
                expanding_window=True,
                initial_train_size=30,
            )


class TestCompareModels:
    def test_comparison_table_columns(self):
        series = make_price_series(n=250, seed=8)
        report = compare_models(
            series,
            model_names=["Baseline (Naive)", "Random Forest"],
            val_size=20,
        )
        df = report.comparison_table()
        for col in ("Model", "MAE", "MSE", "RMSE", "MAPE", "R2", "Status"):
            assert col in df.columns
        assert len(df) == 2

    def test_sorted_by_rmse(self):
        series = make_price_series(n=250, seed=9)
        report = compare_models(
            series,
            model_names=["Baseline (Naive)", "Random Forest"],
            val_size=20,
        )
        df = report.comparison_table()
        ok = df[df["Status"] == "OK"]
        if len(ok) > 1:
            rmse = ok["RMSE"].to_numpy(dtype=float)
            assert list(rmse) == sorted(rmse)

    def test_skip_unavailable_models(self):
        series = make_price_series(n=200, seed=10)
        report = compare_models(
            series,
            model_names=["Baseline (Naive)", "UnknownModelXYZ"],
            val_size=15,
            skip_unavailable=True,
        )
        df = report.comparison_table()
        assert len(df) == 1
        assert df.iloc[0]["Model"] == "Baseline (Naive)"

    def test_no_models_raises(self):
        series = make_price_series(n=100, seed=11)
        with pytest.raises(ValueError, match="No models available"):
            compare_models(
                series,
                model_names=["UnknownModelXYZ"],
                skip_unavailable=True,
            )

    def test_get_model_results(self):
        series = make_price_series(n=200, seed=12)
        report = compare_models(
            series,
            model_names=["Baseline (Naive)"],
            val_size=15,
        )
        res = report.get_model_results("Baseline (Naive)")
        assert res is not None
        assert res.model_name == "Baseline (Naive)"

    def test_metrics_from_actual_predictions(self):
        series = make_price_series(n=200, seed=13)
        report = compare_models(
            series,
            model_names=["Baseline (Naive)"],
            val_size=20,
        )
        res = report.get_model_results("Baseline (Naive)")
        assert res is not None
        # Recompute MAE from stored arrays — must match reported MAE
        mae = float(np.mean(np.abs(res.y_true - res.y_pred)))
        assert res.MAE == pytest.approx(mae, rel=1e-9)

    def test_test_table_separate(self):
        series = make_price_series(n=230, seed=14)
        report = compare_models(
            series,
            model_names=["Baseline (Naive)"],
            val_size=20,
            test_size=10,
            include_test=True,
        )
        val_df = report.comparison_table()
        test_df = report.test_comparison_table()
        assert len(val_df) == 1
        assert len(test_df) == 1
        assert val_df.iloc[0]["Model"] == test_df.iloc[0]["Model"]


class TestModelComparisonReportLegacyCtor:
    def test_results_kwarg_splits_validation_and_test(self):
        series = make_price_series(n=200, seed=15)
        all_results = evaluate_model(
            "Baseline (Naive)",
            series,
            val_size=15,
            test_size=10,
            include_test=True,
        )
        report = ModelComparisonReport(results=all_results)
        assert len(report.validation_results) == 1
        assert len(report.test_results) == 1
