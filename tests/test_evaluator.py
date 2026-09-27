"""
tests/test_evaluator.py
-----------------------
Unit tests for models/evaluator.py.
"""

import math

import numpy as np
import pandas as pd
import pytest

from models.evaluator import (
    compute_metrics,
    evaluate_all,
    best_model_name,
    format_metrics,
)
from models.naive_model import NaiveModel
from models.random_forest_model import RandomForestModel
from tests.conftest import make_price_series


# ---------------------------------------------------------------------------
# compute_metrics
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_prediction_zero_error(self):
        y = np.array([100.0, 101.0, 102.0])
        m = compute_metrics(y, y)
        assert m["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert m["RMSE"] == pytest.approx(0.0, abs=1e-9)
        assert m["MAPE"] == pytest.approx(0.0, abs=1e-9)

    def test_r2_perfect_prediction(self):
        y = np.array([100.0, 101.0, 102.0])
        m = compute_metrics(y, y)
        assert m["R2"] == pytest.approx(1.0, abs=1e-9)

    def test_r2_mean_predictor(self):
        """Predicting the mean gives R2 = 0."""
        y    = np.array([100.0, 110.0, 120.0])
        mean = np.mean(y)
        m    = compute_metrics(y, np.full_like(y, mean))
        assert m["R2"] == pytest.approx(0.0, abs=1e-6)

    def test_mae_known_value(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([105.0, 95.0])
        m = compute_metrics(y_true, y_pred)
        assert m["MAE"] == pytest.approx(5.0)

    def test_rmse_known_value(self):
        y_true = np.array([100.0, 100.0])
        y_pred = np.array([103.0, 97.0])   # errors: 3, -3  → RMSE = 3
        m = compute_metrics(y_true, y_pred)
        assert m["RMSE"] == pytest.approx(3.0)

    def test_mape_known_value(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])  # errors: 10%, 10% → MAPE = 10%
        m = compute_metrics(y_true, y_pred)
        assert m["MAPE"] == pytest.approx(10.0, rel=1e-4)

    def test_n_test_correct(self):
        y = np.arange(10, dtype=float)
        m = compute_metrics(y, y)
        assert m["n_test"] == 10

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="length"):
            compute_metrics([1.0, 2.0], [1.0])

    def test_returns_dict(self):
        m = compute_metrics([100.0], [100.0])
        assert isinstance(m, dict)

    def test_list_input_accepted(self):
        m = compute_metrics([100.0, 101.0], [100.5, 101.5])
        assert m["MAE"] == pytest.approx(0.5)

    def test_rmse_ge_mae(self):
        rng = np.random.default_rng(0)
        y_t = rng.uniform(100, 200, 50)
        y_p = y_t + rng.normal(0, 5, 50)
        m = compute_metrics(y_t, y_p)
        assert m["RMSE"] >= m["MAE"] - 1e-9


# ---------------------------------------------------------------------------
# evaluate_all
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    @pytest.fixture(scope="class")
    def series(self):
        return make_price_series(n=200, seed=5)

    def test_returns_dataframe(self, series):
        models = [NaiveModel()]
        df = evaluate_all(models, series, val_size=20)
        assert hasattr(df, "columns")

    def test_columns_present(self, series):
        df = evaluate_all([NaiveModel()], series, val_size=20)
        for col in ("Model", "MAE", "RMSE", "MAPE", "R2", "n_test", "Status"):
            assert col in df.columns

    def test_rows_equal_number_of_models(self, series):
        models = [NaiveModel(), NaiveModel()]  # two instances
        df = evaluate_all(models, series, val_size=20)
        assert len(df) == 2

    def test_sorted_by_rmse(self, series):
        models = [
            NaiveModel(),
            RandomForestModel(n_estimators=5, random_state=0),
        ]
        df = evaluate_all(models, series, val_size=20)
        ok = df[df["Status"] == "OK"]
        if len(ok) > 1:
            rmse_vals = ok["RMSE"].to_numpy(dtype=float)
            assert list(rmse_vals) == sorted(rmse_vals)

    def test_failed_model_has_status_error(self, series):
        class BrokenModel(NaiveModel):
            name = "Broken"
            def fit(self, s, val_size=20):
                raise RuntimeError("intentional failure")
            def evaluate(self): ...
            def predict(self, steps): ...

        df = evaluate_all([BrokenModel()], series, val_size=20)
        assert "Error" in df.iloc[0]["Status"]

    def test_ok_status_on_success(self, series):
        df = evaluate_all([NaiveModel()], series, val_size=20)
        assert df.iloc[0]["Status"] == "OK"


# ---------------------------------------------------------------------------
# best_model_name
# ---------------------------------------------------------------------------

class TestBestModelName:
    def test_returns_lowest_rmse_name(self):
        """best_model_name picks the row with lowest RMSE among OK rows."""
        # Simulate a sorted DataFrame as evaluate_all produces
        import pandas as pd
        df = pd.DataFrame([
            {"Model": "B", "RMSE": 3.0, "Status": "OK"},
            {"Model": "A", "RMSE": 5.0, "Status": "OK"},
            {"Model": "C", "RMSE": 7.0, "Status": "OK"},
        ]).sort_values("RMSE").reset_index(drop=True)
        assert best_model_name(df) == "B"

    def test_ignores_error_rows(self):
        df = pd.DataFrame([
            {"Model": "Good", "RMSE": 5.0,  "Status": "OK"},
            {"Model": "Bad",  "RMSE": 1.0,  "Status": "Error: x"},
        ])
        assert best_model_name(df) == "Good"

    def test_empty_ok_rows_returns_none(self):
        df = pd.DataFrame([
            {"Model": "A", "RMSE": 1.0, "Status": "Error: x"},
        ])
        assert best_model_name(df) is None

    def test_single_model(self):
        df = pd.DataFrame([{"Model": "X", "RMSE": 2.5, "Status": "OK"}])
        assert best_model_name(df) == "X"


# ---------------------------------------------------------------------------
# format_metrics
# ---------------------------------------------------------------------------

class TestFormatMetrics:
    def test_returns_dict(self):
        raw = {"MAE": 1.5, "RMSE": 2.0, "MAPE": 3.0, "R2": 0.9, "n_test": 30}
        assert isinstance(format_metrics(raw), dict)

    def test_mape_has_percent(self):
        raw = {"MAE": 1.0, "RMSE": 1.5, "MAPE": 5.0, "R2": 0.8, "n_test": 20}
        assert "%" in format_metrics(raw)["MAPE"]

    def test_mae_has_currency_symbol(self):
        raw = {"MAE": 2.0, "RMSE": 3.0, "MAPE": 1.0, "R2": 0.7, "n_test": 10}
        assert "$" in format_metrics(raw, currency="$")["MAE"]

    def test_none_value_returns_na(self):
        raw = {"MAE": None, "RMSE": None, "MAPE": None, "R2": None, "n_test": 0}
        fm = format_metrics(raw)
        for key in ("MAE", "RMSE", "MAPE", "R2"):
            assert "N/A" in fm[key] or fm[key] in ("N/A", "$N/A")
