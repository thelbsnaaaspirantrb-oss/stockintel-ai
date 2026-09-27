"""
tests/test_random_forest_model.py
----------------------------------
Unit tests for models/random_forest_model.py.
"""

import numpy as np
import pytest

from tests.conftest import make_price_series
from models.random_forest_model import RandomForestModel


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series():
    return make_price_series(n=300, seed=10)


@pytest.fixture(scope="module")
def fitted_model(series):
    m = RandomForestModel(n_estimators=10, random_state=0)   # fast for tests
    m.fit(series, val_size=30)
    return m


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestRFInterface:
    def test_name(self):
        assert RandomForestModel.name == "Random Forest"

    def test_unfitted_flag(self):
        assert RandomForestModel().is_fitted is False

    def test_fitted_flag(self, fitted_model):
        assert fitted_model.is_fitted is True

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            RandomForestModel().predict(5)

    def test_fit_returns_self(self, series):
        m = RandomForestModel(n_estimators=5, random_state=1)
        assert m.fit(series, val_size=30) is m

    def test_chaining(self, series):
        preds = RandomForestModel(n_estimators=5).fit(series, val_size=20).predict(5)
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestRFfit:
    def test_short_series_raises(self):
        s = make_price_series(n=40)
        with pytest.raises(ValueError):
            RandomForestModel(n_estimators=5).fit(s, val_size=35)

    def test_val_arrays_populated(self, fitted_model):
        assert fitted_model._val_y_true is not None
        assert fitted_model._val_y_pred is not None
        assert len(fitted_model._val_y_true) == 30
        assert len(fitted_model._val_y_pred) == 30

    def test_train_series_stored(self, fitted_model, series):
        # Training series should be shorter than full series by val_size
        assert len(fitted_model._train_series) == len(series) - 30


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestRFPredict:
    def test_predict_length(self, fitted_model):
        assert len(fitted_model.predict(7)) == 7

    def test_predict_positive(self, fitted_model):
        preds = fitted_model.predict(5)
        assert all(p > 0 for p in preds), f"Non-positive predictions: {preds}"

    def test_predict_returns_floats(self, fitted_model):
        preds = fitted_model.predict(4)
        assert all(isinstance(p, float) for p in preds)

    def test_predict_single_step(self, fitted_model):
        assert len(fitted_model.predict(1)) == 1

    def test_predict_30_steps(self, fitted_model):
        assert len(fitted_model.predict(30)) == 30

    def test_predictions_within_plausible_range(self, fitted_model, series):
        """Predictions should be within 50% of the last known price."""
        last_price = float(series.iloc[-1])
        preds = fitted_model.predict(7)
        for p in preds:
            assert 0.5 * last_price <= p <= 2.0 * last_price, (
                f"Prediction {p} is far outside plausible range "
                f"for last price {last_price}"
            )


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestRFEvaluate:
    def test_keys_present(self, fitted_model):
        m = fitted_model.evaluate()
        for k in ("MAE", "RMSE", "MAPE", "R2", "n_test"):
            assert k in m

    def test_mae_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["MAE"] >= 0

    def test_rmse_ge_mae(self, fitted_model):
        m = fitted_model.evaluate()
        assert m["RMSE"] >= m["MAE"] - 1e-9

    def test_n_test_correct(self, fitted_model):
        assert fitted_model.evaluate()["n_test"] == 30

    def test_mape_between_0_and_200(self, fitted_model):
        """MAPE on a financial series should be in a sane range."""
        mape = fitted_model.evaluate()["MAPE"]
        assert 0 <= mape <= 200, f"Suspicious MAPE: {mape}"
