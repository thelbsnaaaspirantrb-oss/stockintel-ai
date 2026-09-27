"""
tests/test_arima_model.py
-------------------------
Unit tests for models/arima_model.py (ARIMA).

Performance note
----------------
ARIMA validation uses rolling one-step-ahead refitting, which is O(val_size)
model fits.  All fixtures here use a short series (120 rows, val_size=15) and
a fixed order=(1,1,0) to keep the suite fast (< 10 s total).
The auto-order selection path is tested with a deliberately tiny grid.
"""

import numpy as np
import pytest

from tests.conftest import make_price_series
from models.arima_model import ARIMAModel, _best_arima_order


# ---------------------------------------------------------------------------
# Fixtures — deliberately short for speed
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series():
    """120 days — enough for ARIMA(1,1,0) + val_size=15."""
    return make_price_series(n=120, seed=7)


@pytest.fixture(scope="module")
def fitted_model(series):
    """Fixed order to avoid slow grid search in CI."""
    m = ARIMAModel(order=(1, 1, 0))
    m.fit(series, val_size=15)
    return m


# ---------------------------------------------------------------------------
# Interface compliance
# ---------------------------------------------------------------------------

class TestARIMAInterface:
    def test_name(self):
        assert ARIMAModel.name == "ARIMA"

    def test_unfitted_flag(self):
        assert ARIMAModel(order=(1, 1, 0)).is_fitted is False

    def test_fitted_flag(self, fitted_model):
        assert fitted_model.is_fitted is True

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            ARIMAModel(order=(1, 1, 0)).predict(5)

    def test_fit_returns_self(self, series):
        m = ARIMAModel(order=(1, 1, 0))
        assert m.fit(series, val_size=15) is m

    def test_order_property(self, fitted_model):
        assert fitted_model.order == (1, 1, 0)


# ---------------------------------------------------------------------------
# fit()
# ---------------------------------------------------------------------------

class TestARIMAFit:
    def test_short_series_raises(self):
        s = make_price_series(n=20)
        with pytest.raises(ValueError):
            ARIMAModel(order=(1, 1, 0)).fit(s, val_size=15)

    def test_val_arrays_length(self, fitted_model):
        assert len(fitted_model._val_y_true) == 15
        assert len(fitted_model._val_y_pred) == 15

    def test_auto_order_selection_tiny_grid(self, series):
        """Auto order with 1-step grids — fast, just checks valid tuple."""
        m = ARIMAModel()   # auto
        # Patch a tiny grid via _best_arima_order call
        from models.arima_model import _best_arima_order as _bao
        order = _bao(
            series.to_numpy(dtype=float),
            p_range=range(0, 2),
            d_range=range(1, 2),
            q_range=range(0, 2),
        )
        assert len(order) == 3
        p, d, q = order
        assert p >= 0 and d >= 1 and q >= 0


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestARIMAPredict:
    def test_predict_length(self, fitted_model):
        assert len(fitted_model.predict(7)) == 7

    def test_predict_positive(self, fitted_model):
        preds = fitted_model.predict(5)
        assert all(p > 0 for p in preds), f"Non-positive: {preds}"

    def test_predict_returns_floats(self, fitted_model):
        preds = fitted_model.predict(3)
        assert all(isinstance(p, float) for p in preds)

    def test_predict_single_step(self, fitted_model):
        assert len(fitted_model.predict(1)) == 1

    def test_predict_30_steps(self, fitted_model):
        assert len(fitted_model.predict(30)) == 30


# ---------------------------------------------------------------------------
# evaluate()
# ---------------------------------------------------------------------------

class TestARIMAEvaluate:
    def test_keys_present(self, fitted_model):
        m = fitted_model.evaluate()
        for k in ("MAE", "RMSE", "MAPE", "R2", "n_test"):
            assert k in m

    def test_mae_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["MAE"] >= 0

    def test_rmse_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["RMSE"] >= 0

    def test_n_test_correct(self, fitted_model):
        assert fitted_model.evaluate()["n_test"] == 15

    def test_rmse_ge_mae(self, fitted_model):
        m = fitted_model.evaluate()
        assert m["RMSE"] >= m["MAE"] - 1e-9


# ---------------------------------------------------------------------------
# _best_arima_order helper  (tiny grid — fast)
# ---------------------------------------------------------------------------

class TestBestARIMAOrder:
    def test_returns_tuple_length_3(self):
        s = make_price_series(n=80, seed=1).to_numpy()
        order = _best_arima_order(
            s,
            p_range=range(0, 2),
            d_range=range(1, 2),
            q_range=range(0, 2),
        )
        assert len(order) == 3

    def test_d_is_positive(self):
        s = make_price_series(n=80, seed=2).to_numpy()
        _, d, _ = _best_arima_order(
            s,
            p_range=range(0, 2),
            d_range=range(1, 2),
            q_range=range(0, 2),
        )
        assert d >= 1
