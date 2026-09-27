"""
tests/test_lstm_model.py
------------------------
Unit tests for models/lstm_model.py.

Because TensorFlow may not be installed in all environments, every test
class checks _TF_OK and skips gracefully when TF is absent.

When TF is present but lstm_model.h5 is missing, we verify that
FileNotFoundError is raised rather than crashing silently.
"""

import os
import pytest

from models.lstm_model import LSTMModel, _TF_OK
from tests.conftest import make_price_series

# ---------------------------------------------------------------------------
# Module-level skip when TF is unavailable
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.skipif(
    not _TF_OK, reason="tensorflow not installed"
)

_H5_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lstm_model.h5",
)
_H5_EXISTS = os.path.exists(_H5_PATH)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series():
    return make_price_series(n=500, seed=99)


@pytest.fixture(scope="module")
def fitted_model(series):
    if not _H5_EXISTS:
        pytest.skip("lstm_model.h5 not found — skipping LSTM fit tests")
    m = LSTMModel()
    m.fit(series, val_size=30)
    return m


# ---------------------------------------------------------------------------
# Interface compliance (importable regardless of h5 file)
# ---------------------------------------------------------------------------

class TestLSTMInterface:
    def test_name(self):
        assert LSTMModel.name == "LSTM"

    def test_unfitted_flag(self):
        assert LSTMModel().is_fitted is False

    def test_predict_before_fit_raises(self):
        with pytest.raises(RuntimeError):
            LSTMModel().predict(5)

    def test_missing_h5_raises_file_not_found(self, series):
        m = LSTMModel(model_path="/nonexistent/path/model.h5")
        with pytest.raises(FileNotFoundError):
            m.fit(series, val_size=30)

    def test_short_series_raises(self):
        if not _H5_EXISTS:
            pytest.skip("lstm_model.h5 not found")
        s = make_price_series(n=50)
        m = LSTMModel()
        with pytest.raises(ValueError, match="sequence_length"):
            m.fit(s, val_size=30)


# ---------------------------------------------------------------------------
# fit()  [requires h5]
# ---------------------------------------------------------------------------

class TestLSTMFit:
    def test_fitted_flag(self, fitted_model):
        assert fitted_model.is_fitted is True

    def test_fit_returns_self(self, series):
        if not _H5_EXISTS:
            pytest.skip("lstm_model.h5 not found")
        m = LSTMModel()
        assert m.fit(series, val_size=30) is m

    def test_val_arrays_length(self, fitted_model):
        assert len(fitted_model._val_y_true) == 30
        assert len(fitted_model._val_y_pred) == 30

    def test_last_sequence_shape(self, fitted_model):
        assert fitted_model._last_sequence is not None
        assert fitted_model._last_sequence.shape == (60, 1)


# ---------------------------------------------------------------------------
# predict()  [requires h5]
# ---------------------------------------------------------------------------

class TestLSTMPredict:
    def test_predict_length(self, fitted_model):
        assert len(fitted_model.predict(7)) == 7

    def test_predict_returns_floats(self, fitted_model):
        preds = fitted_model.predict(3)
        assert all(isinstance(p, float) for p in preds)

    def test_predict_positive(self, fitted_model):
        preds = fitted_model.predict(5)
        assert all(p > 0 for p in preds)

    def test_predict_30_steps(self, fitted_model):
        assert len(fitted_model.predict(30)) == 30


# ---------------------------------------------------------------------------
# evaluate()  [requires h5]
# ---------------------------------------------------------------------------

class TestLSTMEvaluate:
    def test_keys_present(self, fitted_model):
        for k in ("MAE", "RMSE", "MAPE", "R2", "n_test"):
            assert k in fitted_model.evaluate()

    def test_n_test(self, fitted_model):
        assert fitted_model.evaluate()["n_test"] == 30

    def test_mae_non_negative(self, fitted_model):
        assert fitted_model.evaluate()["MAE"] >= 0

    def test_rmse_ge_mae(self, fitted_model):
        m = fitted_model.evaluate()
        assert m["RMSE"] >= m["MAE"] - 1e-9
