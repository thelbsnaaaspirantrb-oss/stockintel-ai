"""
tests/test_naive_model.py
-------------------------
Unit tests for models/naive_model.py (Naive Persistence Baseline).
"""

import numpy as np
import pytest

from tests.conftest import make_price_series
from models.naive_model import NaiveModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def series():
    return make_price_series(n=100)


@pytest.fixture
def fitted_model(series):
    m = NaiveModel()
    m.fit(series, val_size=20)
    return m


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestNaiveModelInterface:
    def test_name(self):
        assert NaiveModel.name == "Baseline (Naive)"

    def test_unfitted_flag(self):
        m = NaiveModel()
        assert m.is_fitted is False

    def test_fitted_flag_after_fit(self, fitted_model):
        assert fitted_model.is_fitted is True

    def test_predict_before_fit_raises(self):
        m = NaiveModel()
        with pytest.raises(RuntimeError, match="fit\\(\\)"):
            m.predict(5)

    def test_repr_contains_name(self):
        m = NaiveModel()
        assert "NaiveModel" in repr(m)


# ---------------------------------------------------------------------------
# fit() behaviour
# ---------------------------------------------------------------------------

class TestNaiveModelFit:
    def test_fit_returns_self(self, series):
        m = NaiveModel()
        result = m.fit(series, val_size=20)
        assert result is m

    def test_last_value_set(self, series, fitted_model):
        # last training value = series[-21] (before the 20-element val tail)
        expected = float(series.iloc[-21])
        assert fitted_model._last_value == pytest.approx(expected, rel=1e-6)

    def test_short_series_raises(self):
        s = make_price_series(n=10)
        m = NaiveModel()
        with pytest.raises(ValueError, match="exceed"):
            m.fit(s, val_size=10)

    def test_val_y_true_length(self, fitted_model):
        assert len(fitted_model._val_y_true) == 20

    def test_val_y_pred_length(self, fitted_model):
        assert len(fitted_model._val_y_pred) == 20

    def test_chaining(self, series):
        preds = NaiveModel().fit(series, val_size=10).predict(5)
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# predict() behaviour
# ---------------------------------------------------------------------------

class TestNaiveModelPredict:
    def test_predict_length(self, fitted_model):
        preds = fitted_model.predict(7)
        assert len(preds) == 7

    def test_predict_all_equal_last_value(self, fitted_model):
        preds = fitted_model.predict(10)
        expected = fitted_model._last_value
        assert all(p == pytest.approx(expected) for p in preds)

    def test_predict_single_step(self, fitted_model):
        preds = fitted_model.predict(1)
        assert len(preds) == 1
        assert preds[0] == pytest.approx(fitted_model._last_value)

    def test_predict_max_steps(self, fitted_model):
        preds = fitted_model.predict(30)
        assert len(preds) == 30

    def test_predict_returns_list_of_floats(self, fitted_model):
        preds = fitted_model.predict(5)
        assert isinstance(preds, list)
        assert all(isinstance(p, float) for p in preds)

    def test_predict_positive_prices(self, fitted_model):
        preds = fitted_model.predict(5)
        assert all(p > 0 for p in preds)


# ---------------------------------------------------------------------------
# evaluate() behaviour
# ---------------------------------------------------------------------------

class TestNaiveModelEvaluate:
    def test_evaluate_returns_required_keys(self, fitted_model):
        m = fitted_model.evaluate()
        for key in ("MAE", "RMSE", "MAPE", "R2", "n_test"):
            assert key in m, f"Missing key: {key}"

    def test_mae_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["MAE"] >= 0

    def test_rmse_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["RMSE"] >= 0

    def test_mape_non_negative(self, fitted_model):
        m = fitted_model.evaluate()
        assert m["MAPE"] >= 0

    def test_n_test_equals_val_size(self, fitted_model):
        assert fitted_model.evaluate()["n_test"] == 20

    def test_rmse_ge_mae(self, fitted_model):
        m = fitted_model.evaluate()
        assert m["RMSE"] >= m["MAE"] - 1e-9

    def test_evaluate_before_fit_raises(self):
        m = NaiveModel()
        with pytest.raises(RuntimeError):
            m.evaluate()

    def test_perfect_flat_series_has_zero_error(self):
        """On a constant price series, persistence has zero error."""
        import pandas as pd
        flat = pd.Series(
            [50.0] * 80,
            index=pd.date_range("2023-01-01", periods=80, freq="B"),
        )
        m = NaiveModel()
        m.fit(flat, val_size=20)
        metrics = m.evaluate()
        assert metrics["MAE"]  == pytest.approx(0.0, abs=1e-9)
        assert metrics["RMSE"] == pytest.approx(0.0, abs=1e-9)
