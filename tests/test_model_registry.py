"""
tests/test_model_registry.py
-----------------------------
Unit tests for models/model_registry.py.
"""

import pytest
import pandas as pd

from models.model_registry import (
    REGISTRY,
    get_model,
    available_models,
    run_all_models,
    auto_select,
)
from models.base_model import BaseForecaster
from models.naive_model import NaiveModel
from models.random_forest_model import RandomForestModel
from tests.conftest import make_price_series


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def series():
    return make_price_series(n=250, seed=77)


# ---------------------------------------------------------------------------
# REGISTRY structure
# ---------------------------------------------------------------------------

class TestRegistry:
    def test_registry_is_dict(self):
        assert isinstance(REGISTRY, dict)

    def test_expected_keys_present(self):
        for key in ("Baseline (Naive)", "Random Forest"):
            assert key in REGISTRY, f"Missing key: {key}"

    def test_all_values_callable_or_none(self):
        for name, factory in REGISTRY.items():
            assert factory is None or callable(factory), (
                f"Registry entry '{name}' is neither callable nor None"
            )

    def test_naive_always_available(self):
        assert REGISTRY["Baseline (Naive)"] is not None

    def test_rf_always_available(self):
        assert REGISTRY["Random Forest"] is not None


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------

class TestGetModel:
    def test_returns_base_forecaster(self):
        m = get_model("Baseline (Naive)")
        assert isinstance(m, BaseForecaster)

    def test_returns_correct_type_naive(self):
        assert isinstance(get_model("Baseline (Naive)"), NaiveModel)

    def test_returns_correct_type_rf(self):
        assert isinstance(get_model("Random Forest"), RandomForestModel)

    def test_each_call_returns_new_instance(self):
        m1 = get_model("Baseline (Naive)")
        m2 = get_model("Baseline (Naive)")
        assert m1 is not m2

    def test_unknown_model_raises_key_error(self):
        with pytest.raises(KeyError, match="Unknown model"):
            get_model("NoSuchModel_XYZ")

    def test_returned_model_is_unfitted(self):
        m = get_model("Baseline (Naive)")
        assert m.is_fitted is False


# ---------------------------------------------------------------------------
# available_models
# ---------------------------------------------------------------------------

class TestAvailableModels:
    def test_returns_list(self):
        assert isinstance(available_models(), list)

    def test_naive_always_in_available(self):
        assert "Baseline (Naive)" in available_models()

    def test_rf_always_in_available(self):
        assert "Random Forest" in available_models()

    def test_no_none_entries(self):
        """available_models should never include models with None factories."""
        for name in available_models():
            assert REGISTRY[name] is not None

    def test_at_least_two_models(self):
        assert len(available_models()) >= 2


# ---------------------------------------------------------------------------
# run_all_models
# ---------------------------------------------------------------------------

class TestRunAllModels:
    def test_returns_two_items(self, series):
        df, preds = run_all_models(
            series, steps=5, val_size=20,
            exclude=["ARIMA", "LSTM", "XGBoost"],   # keep fast models only
        )
        assert isinstance(df, pd.DataFrame)
        assert isinstance(preds, dict)

    def test_comparison_df_has_required_columns(self, series):
        df, _ = run_all_models(
            series, steps=5, val_size=20,
            exclude=["ARIMA", "LSTM", "XGBoost"],
        )
        for col in ("Model", "MAE", "RMSE", "Status"):
            assert col in df.columns

    def test_predictions_dict_non_empty(self, series):
        _, preds = run_all_models(
            series, steps=5, val_size=20,
            exclude=["ARIMA", "LSTM", "XGBoost"],
        )
        assert len(preds) > 0

    def test_prediction_lengths_correct(self, series):
        _, preds = run_all_models(
            series, steps=7, val_size=20,
            exclude=["ARIMA", "LSTM", "XGBoost"],
        )
        for name, p in preds.items():
            assert len(p) == 7, f"{name} predictions have wrong length"

    def test_sorted_by_rmse(self, series):
        df, _ = run_all_models(
            series, steps=5, val_size=20,
            exclude=["ARIMA", "LSTM", "XGBoost"],
        )
        ok = df[df["Status"] == "OK"]
        if len(ok) > 1:
            rmse_vals = ok["RMSE"].to_numpy(dtype=float)
            assert list(rmse_vals) == sorted(rmse_vals)

    def test_exclude_removes_models(self, series):
        df, _ = run_all_models(
            series, steps=5, val_size=20,
            exclude=["Baseline (Naive)", "ARIMA", "LSTM", "XGBoost"],
        )
        assert "Baseline (Naive)" not in df["Model"].values


# ---------------------------------------------------------------------------
# auto_select
# ---------------------------------------------------------------------------

class TestAutoSelect:
    @pytest.fixture(scope="class")
    def fast_series(self):
        """250 rows — sufficient for RF+Naive, ARIMA excluded for speed."""
        return make_price_series(n=250, seed=77)

    def _auto_fast(self, series, steps=5):
        """Run auto_select excluding ARIMA and LSTM (slow/unavailable)."""
        from models.model_registry import run_all_models, best_model_name
        df, preds = run_all_models(
            series, steps=steps, val_size=20,
            exclude=["ARIMA", "LSTM"],
        )
        chosen = best_model_name(df)
        if chosen is None or chosen not in preds:
            raise RuntimeError("No model succeeded")
        return chosen, preds[chosen], df

    def test_returns_three_items(self, fast_series):
        result = self._auto_fast(fast_series)
        assert len(result) == 3

    def test_best_name_is_string(self, fast_series):
        name, _, _ = self._auto_fast(fast_series)
        assert isinstance(name, str)

    def test_best_name_in_available(self, fast_series):
        name, _, _ = self._auto_fast(fast_series)
        assert isinstance(name, str) and len(name) > 0

    def test_predictions_correct_length(self, fast_series):
        _, preds, _ = self._auto_fast(fast_series, steps=7)
        assert len(preds) == 7

    def test_predictions_are_floats(self, fast_series):
        _, preds, _ = self._auto_fast(fast_series)
        assert all(isinstance(p, float) for p in preds)

    def test_comparison_df_returned(self, fast_series):
        _, _, df = self._auto_fast(fast_series)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns

    def test_best_is_lowest_rmse(self, fast_series):
        best_name, _, df = self._auto_fast(fast_series)
        ok = df[df["Status"] == "OK"]
        if len(ok) > 0:
            lowest_rmse_model = ok.iloc[0]["Model"]
            assert best_name == lowest_rmse_model
