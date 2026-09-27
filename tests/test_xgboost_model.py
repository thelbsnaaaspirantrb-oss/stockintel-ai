"""
tests/test_xgboost_model.py
----------------------------
Unit tests for models/xgboost_model.py.
"""

import numpy as np
import pytest

from tests.conftest import make_price_series
from models.xgboost_model import XGBoostModel, _XGB_OK


# ---------------------------------------------------------------------------
# Skip entire module if xgboost is not installed
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not _XGB_OK, reason="xgboost not installed"
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series():
    return make_price_series(n=300, seed=20)


@pytest.fixture(scope="module")
def fitted_model(series):
    m = XGBoostModel(n_estimators=30, random_state=0)  # fast for tests
    m.fit(series, val_size=30)
    return m


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestXGBInterface:
    def test_name(self):
        assert XGBoostModel.name == "XGBoost"

    def test_unfitted_flag(self):
        assert XGBoostModel().is_fitted is False

    def test_fitted_flag(self, fitted_model):
        assert fitted_model.is_fitted is True

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            XGBoostModel().predict(5)

    def test_fit_returns_self(self, series):
        m = XGBoostModel(n_estimators=10, random_state=1)
        assert m.fit(series, val_size=30) is m

    def test_chaining(self, series):
        preds = XGBoostModel(n_estimators=10).fit(series, val_size=20).predict(5)
        assert len(preds) == 5


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestXGBFit:
    def test_short_series_raises(self):
        s = make_price_series(n=40)
        with pytest.raises(ValueError):
            XGBoostModel(n_estimators=5).fit(s, val_size=35)

    def test_val_arrays_populated(self, fitted_model):
        assert len(fitted_model._val_y_true) == 30
        assert len(fitted_model._val_y_pred) == 30

    def test_train_series_stored(self, fitted_model, series):
        assert len(fitted_model._train_series) == len(series) - 30

    def test_model_fitted(self, fitted_model):
        assert fitted_model._model is not None


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestXGBPredict:
    def test_predict_length(self, fitted_model):
        assert len(fitted_model.predict(7)) == 7

    def test_predict_positive(self, fitted_model):
        preds = fitted_model.predict(5)
        assert all(p > 0 for p in preds)

    def test_predict_returns_floats(self, fitted_model):
        preds = fitted_model.predict(4)
        assert all(isinstance(p, float) for p in preds)

    def test_predict_single_step(self, fitted_model):
        assert len(fitted_model.predict(1)) == 1

    def test_predict_30_steps(self, fitted_model):
        assert len(fitted_model.predict(30)) == 30

    def test_predictions_within_plausible_range(self, fitted_model, series):
        last_price = float(series.iloc[-1])
        preds = fitted_model.predict(7)
        for p in preds:
            assert 0.5 * last_price <= p <= 2.0 * last_price


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestXGBEvaluate:
    def test_keys_present(self, fitted_model):
        for k in ("MAE", "RMSE", "MAPE", "R2", "n_test"):
            assert k in fitted_model.evaluate()

    def test_mae_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["MAE"] >= 0

    def test_rmse_ge_mae(self, fitted_model):
        m = fitted_model.evaluate()
        assert m["RMSE"] >= m["MAE"] - 1e-9

    def test_n_test_correct(self, fitted_model):
        assert fitted_model.evaluate()["n_test"] == 30

    def test_mape_sane(self, fitted_model):
        mape = fitted_model.evaluate()["MAPE"]
        assert 0 <= mape <= 200
