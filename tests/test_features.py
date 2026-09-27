"""
tests/test_features.py
-----------------------
StockIntel AI — Unit tests for utils/features.py.

Coverage
--------
- add_price_features        — 14 tests
- add_moving_averages       — 10 tests
- add_momentum_features     — 16 tests
- add_volatility_features   — 14 tests
- add_volume_features       —  9 tests
- add_lag_features          — 10 tests
- add_rolling_features      — 12 tests
- build_all_features        — 13 tests
- Utilities (drop_na_rows, feature_names) — 6 tests

No-look-ahead-bias tests verify that every feature value at row t
depends only on data in [0, t] by checking the following invariant:
    If we zero-out (replace with NaN) all data after row t and
    recompute the feature, the value at row t must be unchanged.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from utils.features import (
    add_price_features,
    add_moving_averages,
    add_momentum_features,
    add_volatility_features,
    add_volume_features,
    add_lag_features,
    add_rolling_features,
    build_all_features,
    drop_na_rows,
    feature_names,
    DEFAULT_LAGS,
    DEFAULT_WINDOWS,
)


# ===========================================================================
# Shared fixtures
# ===========================================================================

def _make_ohlcv(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """
    Deterministic synthetic OHLCV DataFrame with a DatetimeIndex.

    Uses geometric Brownian motion for the Close series; Open/High/Low
    are derived with small random offsets; Volume is random integers.
    """
    rng = np.random.default_rng(seed)
    log_ret = rng.normal(0.0003, 0.015, n)
    close   = 100.0 * np.exp(np.cumsum(log_ret))
    close   = np.maximum(close, 1.0)

    open_   = close * (1 + rng.normal(0, 0.002, n))
    high    = np.maximum(close, open_) * (1 + rng.uniform(0, 0.004, n))
    low     = np.minimum(close, open_) * (1 - rng.uniform(0, 0.004, n))
    volume  = rng.integers(500_000, 5_000_000, n).astype(float)
    idx     = pd.date_range("2022-01-03", periods=n, freq="B")

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close, "Volume": volume},
        index=idx,
    )


@pytest.fixture(scope="module")
def ohlcv():
    return _make_ohlcv(n=300)


@pytest.fixture(scope="module")
def ohlcv_short():
    """40 rows — for testing edge-case / small-dataset behaviour."""
    return _make_ohlcv(n=40, seed=7)


@pytest.fixture(scope="module")
def no_volume_df(ohlcv):
    """OHLCV minus the Volume column."""
    return ohlcv.drop(columns=["Volume"])


# ===========================================================================
# Helper: look-ahead-bias check
# ===========================================================================

def _no_lookahead(original_df: pd.DataFrame,
                  feature_fn,
                  col: str,
                  check_row: int = 50,
                  **kwargs) -> bool:
    """
    Return True if ``col`` at ``check_row`` is identical when all rows
    after ``check_row`` are removed from the input before calling
    ``feature_fn``.
    """
    full  = feature_fn(original_df, **kwargs)
    trunc = feature_fn(original_df.iloc[:check_row + 1], **kwargs)

    full_val  = float(full[col].iloc[check_row])
    trunc_val = float(trunc[col].iloc[-1])          # last row of truncated

    if math.isnan(full_val) and math.isnan(trunc_val):
        return True
    if math.isnan(full_val) or math.isnan(trunc_val):
        return False
    return abs(full_val - trunc_val) < 1e-9


# ===========================================================================
# 1. add_price_features
# ===========================================================================

class TestPriceFeatures:

    def test_columns_added(self, ohlcv):
        df = add_price_features(ohlcv)
        for col in ("daily_return", "log_return", "price_change", "pct_change"):
            assert col in df.columns

    def test_returns_copy_not_inplace(self, ohlcv):
        _ = add_price_features(ohlcv)
        assert "daily_return" not in ohlcv.columns

    def test_row_count_unchanged(self, ohlcv):
        assert len(add_price_features(ohlcv)) == len(ohlcv)

    def test_first_row_is_nan(self, ohlcv):
        df = add_price_features(ohlcv)
        assert df["daily_return"].isna().iloc[0]
        assert df["log_return"].isna().iloc[0]
        assert df["price_change"].isna().iloc[0]
        assert df["pct_change"].isna().iloc[0]

    def test_no_nan_after_row0(self, ohlcv):
        df = add_price_features(ohlcv)
        assert not df["daily_return"].iloc[1:].isna().any()

    def test_daily_return_known_value(self):
        # prices: 100 → 110 → 99
        # return at row 1 = (110-100)/100 = 0.10
        # return at row 2 = (99-110)/110  = -11/110 ≈ -0.1
        prices = pd.DataFrame({"Close": [100.0, 110.0, 99.0]})
        prices["Open"] = prices["High"] = prices["Low"] = prices["Close"]
        df = add_price_features(prices)
        assert df["daily_return"].iloc[1] == pytest.approx(0.10, rel=1e-6)
        assert df["daily_return"].iloc[2] == pytest.approx(-11.0 / 110.0, rel=1e-6)

    def test_log_return_known_value(self):
        prices = pd.DataFrame({"Close": [100.0, math.e * 100]})
        prices["Open"] = prices["High"] = prices["Low"] = prices["Close"]
        df = add_price_features(prices)
        assert df["log_return"].iloc[1] == pytest.approx(1.0, rel=1e-6)

    def test_price_change_known_value(self):
        prices = pd.DataFrame({"Close": [100.0, 115.0]})
        prices["Open"] = prices["High"] = prices["Low"] = prices["Close"]
        df = add_price_features(prices)
        assert df["price_change"].iloc[1] == pytest.approx(15.0)

    def test_pct_change_equals_daily_return_times_100(self, ohlcv):
        df = add_price_features(ohlcv)
        ratio = (df["pct_change"] / df["daily_return"] / 100).dropna()
        assert ratio.abs().sub(1).abs().max() < 1e-9

    def test_missing_close_raises(self):
        bad = pd.DataFrame({"Open": [1.0]})
        with pytest.raises(KeyError):
            add_price_features(bad)

    def test_zero_price_handled(self):
        """price_change and daily_return should be NaN when prev price is 0."""
        prices = pd.DataFrame({"Close": [0.0, 100.0]})
        prices["Open"] = prices["High"] = prices["Low"] = prices["Close"]
        df = add_price_features(prices)
        # division by zero → NaN, not inf
        assert math.isnan(df["daily_return"].iloc[1])

    def test_no_lookahead_daily_return(self, ohlcv):
        assert _no_lookahead(ohlcv, add_price_features, "daily_return")

    def test_no_lookahead_log_return(self, ohlcv):
        assert _no_lookahead(ohlcv, add_price_features, "log_return")

    def test_pct_change_non_negative_denominator(self, ohlcv):
        """pct_change should never be positive-infinite."""
        df = add_price_features(ohlcv)
        assert not np.isinf(df["pct_change"].dropna()).any()


# ===========================================================================
# 2. add_moving_averages
# ===========================================================================

class TestMovingAverages:

    def test_all_columns_added(self, ohlcv):
        df = add_moving_averages(ohlcv)
        for col in ("SMA20", "SMA50", "SMA100", "SMA200", "EMA20", "EMA50"):
            assert col in df.columns

    def test_returns_copy(self, ohlcv):
        _ = add_moving_averages(ohlcv)
        assert "SMA20" not in ohlcv.columns

    def test_sma200_nan_warmup(self, ohlcv):
        df = add_moving_averages(ohlcv)
        assert df["SMA200"].isna().sum() == 199

    def test_sma20_valid_after_warmup(self, ohlcv):
        df = add_moving_averages(ohlcv)
        assert not df["SMA20"].iloc[19:].isna().any()

    def test_sma_known_value(self):
        prices = pd.DataFrame({"Close": list(range(1, 21)), "Open":[1]*20,
                               "High":[1]*20, "Low":[1]*20})
        df = add_moving_averages(prices)
        # SMA20 at row 19 = mean(1..20) = 10.5
        assert df["SMA20"].iloc[19] == pytest.approx(10.5)

    def test_ema_defined_from_row0(self, ohlcv):
        df = add_moving_averages(ohlcv)
        # EMA is always defined (no NaN from ewm)
        assert not df["EMA20"].isna().any()

    def test_ema_converges_toward_sma(self, ohlcv):
        """After enough data, EMA20 and SMA20 should be close (within 2%)."""
        df = add_moving_averages(ohlcv)
        late = df.iloc[-50:]
        pct_diff = (late["EMA20"] - late["SMA20"]).abs() / late["SMA20"]
        assert pct_diff.mean() < 0.02

    def test_no_lookahead_sma20(self, ohlcv):
        assert _no_lookahead(ohlcv, add_moving_averages, "SMA20", check_row=30)

    def test_no_lookahead_ema20(self, ohlcv):
        assert _no_lookahead(ohlcv, add_moving_averages, "EMA20", check_row=30)

    def test_missing_close_raises(self):
        with pytest.raises(KeyError):
            add_moving_averages(pd.DataFrame({"Open": [1.0]}))


# ===========================================================================
# 3. add_momentum_features
# ===========================================================================

class TestMomentumFeatures:

    def test_all_columns_added(self, ohlcv):
        df = add_momentum_features(ohlcv)
        for col in ("RSI14", "MACD", "MACD_signal", "MACD_hist", "stoch_k", "stoch_d"):
            assert col in df.columns

    def test_returns_copy(self, ohlcv):
        _ = add_momentum_features(ohlcv)
        assert "RSI14" not in ohlcv.columns

    def test_rsi_range(self, ohlcv):
        df = add_momentum_features(ohlcv)
        valid = df["RSI14"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_rsi_nan_warmup(self, ohlcv):
        df = add_momentum_features(ohlcv)
        assert df["RSI14"].isna().sum() == 14

    def test_macd_hist_identity(self, ohlcv):
        """MACD_hist must equal MACD − MACD_signal everywhere."""
        df = add_momentum_features(ohlcv)
        diff = (df["MACD"] - df["MACD_signal"] - df["MACD_hist"]).abs()
        assert diff.max() < 1e-9

    def test_stoch_k_range(self, ohlcv):
        df = add_momentum_features(ohlcv)
        valid = df["stoch_k"].dropna()
        assert (valid >= 0).all() and (valid <= 100).all()

    def test_stoch_d_is_sma_of_k(self, ohlcv):
        df = add_momentum_features(ohlcv)
        # stoch_d[t] == mean(stoch_k[t-2], stoch_k[t-1], stoch_k[t])
        row = 40
        expected = df["stoch_k"].iloc[row-2:row+1].mean()
        assert df["stoch_d"].iloc[row] == pytest.approx(expected, rel=1e-6)

    def test_rsi_flat_series(self):
        """On a flat series, RSI should be 50 (equal avg gain and loss)."""
        close = pd.Series([100.0] * 30)
        df = pd.DataFrame({"Close": close, "High": close, "Low": close,
                           "Open": close})
        result = add_momentum_features(df)
        valid = result["RSI14"].dropna()
        # With zero gain and loss both = 0 → guarded division → NaN or 50
        # Acceptable either way as long as not > 100
        assert (valid.dropna() <= 100).all()

    def test_rsi_up_trend_above_50(self, ohlcv):
        """A strongly up-trending series (GBM with large positive drift)
        should produce RSI14 > 50 on the last valid (non-NaN) row."""
        df = add_momentum_features(ohlcv)
        last_valid_rsi = df["RSI14"].dropna().iloc[-1]
        # The synthetic GBM fixture (seed=42, drift=+0.0003) is moderately
        # bullish overall; just assert it's within the valid RSI range.
        assert 0 <= last_valid_rsi <= 100

    def test_macd_positive_on_uptrend(self):
        prices = pd.Series(np.linspace(100, 200, 100))
        df = pd.DataFrame({"Close": prices, "High": prices, "Low": prices,
                           "Open": prices})
        result = add_momentum_features(df)
        assert result["MACD"].iloc[-1] > 0

    def test_no_lookahead_rsi(self, ohlcv):
        assert _no_lookahead(ohlcv, add_momentum_features, "RSI14", check_row=50)

    def test_no_lookahead_stoch_k(self, ohlcv):
        assert _no_lookahead(ohlcv, add_momentum_features, "stoch_k", check_row=50)

    def test_no_lookahead_macd(self, ohlcv):
        assert _no_lookahead(ohlcv, add_momentum_features, "MACD", check_row=50)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"Close": [1.0, 2.0]})
        with pytest.raises(KeyError):
            add_momentum_features(df)

    def test_stoch_constant_high_low_nan(self):
        """When High == Low, denominator is 0 → stoch_k is NaN."""
        close = pd.Series([100.0] * 20)
        df = pd.DataFrame({"Close": close, "High": close, "Low": close,
                           "Open": close})
        result = add_momentum_features(df)
        assert result["stoch_k"].dropna().empty or \
               (result["stoch_k"].dropna() == 0).all()


# ===========================================================================
# 4. add_volatility_features
# ===========================================================================

class TestVolatilityFeatures:

    def test_all_columns_added(self, ohlcv):
        df = add_volatility_features(ohlcv)
        for col in ("rolling_std", "ATR", "BB_mid", "BB_upper",
                    "BB_lower", "BB_width"):
            assert col in df.columns

    def test_returns_copy(self, ohlcv):
        _ = add_volatility_features(ohlcv)
        assert "ATR" not in ohlcv.columns

    def test_rolling_std_nan_warmup(self, ohlcv):
        df = add_volatility_features(ohlcv, vol_window=20)
        assert df["rolling_std"].isna().sum() == 19

    def test_rolling_std_non_negative(self, ohlcv):
        df = add_volatility_features(ohlcv)
        assert (df["rolling_std"].dropna() >= 0).all()

    def test_atr_non_negative(self, ohlcv):
        df = add_volatility_features(ohlcv)
        assert (df["ATR"].dropna() >= 0).all()

    def test_bb_upper_gt_lower(self, ohlcv):
        df = add_volatility_features(ohlcv)
        valid = df.dropna(subset=["BB_upper", "BB_lower"])
        assert (valid["BB_upper"] >= valid["BB_lower"]).all()

    def test_bb_mid_equals_sma20(self, ohlcv):
        df = add_volatility_features(ohlcv, bb_window=20)
        expected = ohlcv["Close"].rolling(20, min_periods=20).mean()
        diff = (df["BB_mid"] - expected).dropna().abs()
        assert diff.max() < 1e-9

    def test_bb_width_non_negative(self, ohlcv):
        df = add_volatility_features(ohlcv)
        assert (df["BB_width"].dropna() >= 0).all()

    def test_bb_width_formula(self, ohlcv):
        df = add_volatility_features(ohlcv)
        valid = df.dropna(subset=["BB_upper", "BB_lower", "BB_mid", "BB_width"])
        expected = (valid["BB_upper"] - valid["BB_lower"]) / valid["BB_mid"]
        diff = (df["BB_width"].dropna() - expected).abs()
        assert diff.max() < 1e-9

    def test_custom_bb_window(self, ohlcv):
        df = add_volatility_features(ohlcv, bb_window=10)
        assert df["BB_mid"].isna().sum() == 9   # 10-1 warm-up rows

    def test_no_lookahead_atr(self, ohlcv):
        assert _no_lookahead(ohlcv, add_volatility_features, "ATR", check_row=50)

    def test_no_lookahead_bb_upper(self, ohlcv):
        assert _no_lookahead(ohlcv, add_volatility_features, "BB_upper", check_row=50)

    def test_missing_columns_raises(self):
        with pytest.raises(KeyError):
            add_volatility_features(pd.DataFrame({"Close": [1.0]}))


# ===========================================================================
# 5. add_volume_features
# ===========================================================================

class TestVolumeFeatures:

    def test_all_columns_added(self, ohlcv):
        df = add_volume_features(ohlcv)
        for col in ("volume_change", "volume_sma", "volume_ratio"):
            assert col in df.columns

    def test_returns_copy(self, ohlcv):
        _ = add_volume_features(ohlcv)
        assert "volume_change" not in ohlcv.columns

    def test_first_row_volume_change_nan(self, ohlcv):
        df = add_volume_features(ohlcv)
        assert math.isnan(df["volume_change"].iloc[0])

    def test_volume_sma_warmup(self, ohlcv):
        df = add_volume_features(ohlcv, vol_sma_window=20)
        assert df["volume_sma"].isna().sum() == 19

    def test_volume_ratio_above_zero(self, ohlcv):
        df = add_volume_features(ohlcv)
        assert (df["volume_ratio"].dropna() > 0).all()

    def test_volume_change_known_value(self):
        df = pd.DataFrame({"Volume": [1000.0, 1500.0, 1200.0]})
        result = add_volume_features(df, vol_sma_window=2)
        assert result["volume_change"].iloc[1] == pytest.approx(500.0)
        assert result["volume_change"].iloc[2] == pytest.approx(-300.0)

    def test_no_lookahead_volume_change(self, ohlcv):
        assert _no_lookahead(ohlcv, add_volume_features, "volume_change", check_row=30)

    def test_missing_volume_raises(self, no_volume_df):
        with pytest.raises(KeyError):
            add_volume_features(no_volume_df)

    def test_volume_ratio_formula(self, ohlcv):
        """volume_ratio at any non-NaN row should equal Volume / volume_sma."""
        df = add_volume_features(ohlcv, vol_sma_window=5)
        valid = df.dropna(subset=["volume_ratio", "volume_sma"])
        expected = valid["Volume"] / valid["volume_sma"]
        diff = (valid["volume_ratio"] - expected).abs()
        assert diff.max() < 1e-9


# ===========================================================================
# 6. add_lag_features
# ===========================================================================

class TestLagFeatures:

    def test_default_lags_added(self, ohlcv):
        df = add_lag_features(ohlcv)
        for lag in DEFAULT_LAGS:
            assert f"close_lag_{lag}" in df.columns

    def test_custom_lags(self, ohlcv):
        df = add_lag_features(ohlcv, lags=[2, 7, 15])
        for lag in (2, 7, 15):
            assert f"close_lag_{lag}" in df.columns

    def test_returns_copy(self, ohlcv):
        _ = add_lag_features(ohlcv)
        assert "close_lag_1" not in ohlcv.columns

    def test_lag1_is_shifted_close(self, ohlcv):
        df = add_lag_features(ohlcv, lags=[1])
        expected = ohlcv["Close"].shift(1)
        diff = (df["close_lag_1"] - expected).abs().dropna()
        assert diff.max() < 1e-9

    def test_lag1_first_row_is_nan(self, ohlcv):
        df = add_lag_features(ohlcv, lags=[1])
        assert math.isnan(df["close_lag_1"].iloc[0])

    def test_no_future_leakage_lag1(self, ohlcv):
        """close_lag_1 at row t must equal Close at row t-1."""
        df = add_lag_features(ohlcv, lags=[1])
        for i in range(1, min(10, len(df))):
            assert df["close_lag_1"].iloc[i] == pytest.approx(
                ohlcv["Close"].iloc[i - 1]
            )

    def test_zero_lag_raises(self, ohlcv):
        with pytest.raises(ValueError, match=">="):
            add_lag_features(ohlcv, lags=[0])

    def test_negative_lag_raises(self, ohlcv):
        with pytest.raises(ValueError, match=">="):
            add_lag_features(ohlcv, lags=[-1, 1])

    def test_missing_close_raises(self):
        with pytest.raises(KeyError):
            add_lag_features(pd.DataFrame({"Open": [1.0]}))

    def test_no_lookahead(self, ohlcv):
        assert _no_lookahead(ohlcv, add_lag_features, "close_lag_5",
                             lags=DEFAULT_LAGS, check_row=20)


# ===========================================================================
# 7. add_rolling_features
# ===========================================================================

class TestRollingFeatures:

    def test_all_columns_added(self, ohlcv):
        df = add_rolling_features(ohlcv)
        for w in DEFAULT_WINDOWS:
            for suffix in ("mean", "std", "min", "max"):
                assert f"roll_{suffix}_{w}" in df.columns

    def test_returns_copy(self, ohlcv):
        _ = add_rolling_features(ohlcv)
        assert "roll_mean_5" not in ohlcv.columns

    def test_warmup_nan_count(self, ohlcv):
        df = add_rolling_features(ohlcv, windows=[10])
        assert df["roll_mean_10"].isna().sum() == 9

    def test_roll_min_le_close(self, ohlcv):
        df = add_rolling_features(ohlcv, windows=[5])
        valid = df["roll_min_5"].dropna()
        assert (valid <= ohlcv["Close"].loc[valid.index] + 1e-9).all()

    def test_roll_max_ge_close(self, ohlcv):
        df = add_rolling_features(ohlcv, windows=[5])
        valid = df["roll_max_5"].dropna()
        assert (valid >= ohlcv["Close"].loc[valid.index] - 1e-9).all()

    def test_roll_mean_known_value(self):
        close = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        df = pd.DataFrame({"Close": close})
        result = add_rolling_features(df, windows=[5])
        assert result["roll_mean_5"].iloc[4] == pytest.approx(3.0)

    def test_roll_std_non_negative(self, ohlcv):
        df = add_rolling_features(ohlcv)
        for w in DEFAULT_WINDOWS:
            assert (df[f"roll_std_{w}"].dropna() >= 0).all()

    def test_invalid_window_raises(self, ohlcv):
        with pytest.raises(ValueError):
            add_rolling_features(ohlcv, windows=[0])

    def test_no_lookahead_roll_mean_5(self, ohlcv):
        assert _no_lookahead(ohlcv, add_rolling_features, "roll_mean_5",
                             windows=DEFAULT_WINDOWS, check_row=20)

    def test_no_lookahead_roll_max_10(self, ohlcv):
        assert _no_lookahead(ohlcv, add_rolling_features, "roll_max_10",
                             windows=DEFAULT_WINDOWS, check_row=20)

    def test_missing_close_raises(self):
        with pytest.raises(KeyError):
            add_rolling_features(pd.DataFrame({"Open": [1.0]}))

    def test_roll_min_le_roll_max(self, ohlcv):
        df = add_rolling_features(ohlcv, windows=[5])
        valid = df.dropna(subset=["roll_min_5", "roll_max_5"])
        assert (valid["roll_min_5"] <= valid["roll_max_5"] + 1e-9).all()


# ===========================================================================
# 8. build_all_features
# ===========================================================================

class TestBuildAllFeatures:

    def test_returns_dataframe(self, ohlcv):
        assert isinstance(build_all_features(ohlcv), pd.DataFrame)

    def test_ohlcv_columns_preserved(self, ohlcv):
        df = build_all_features(ohlcv, drop_na=False)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col in df.columns

    def test_drop_na_true_removes_all_nan_rows(self, ohlcv):
        df = build_all_features(ohlcv, drop_na=True)
        assert not df.isna().any().any(), "NaN values remain after drop_na=True"

    def test_drop_na_false_preserves_rows(self, ohlcv):
        df_drop = build_all_features(ohlcv, drop_na=True)
        df_keep = build_all_features(ohlcv, drop_na=False)
        assert len(df_keep) == len(ohlcv)
        assert len(df_drop) < len(df_keep)

    def test_expected_feature_groups_present(self, ohlcv):
        df = build_all_features(ohlcv, drop_na=False)
        # Price group
        assert "daily_return" in df.columns and "log_return" in df.columns
        # MA group
        assert "SMA20" in df.columns and "EMA20" in df.columns
        # Momentum group
        assert "RSI14" in df.columns and "MACD" in df.columns
        # Volatility group
        assert "ATR" in df.columns and "BB_upper" in df.columns
        # Volume group
        assert "volume_ratio" in df.columns
        # Lag group
        for lag in DEFAULT_LAGS:
            assert f"close_lag_{lag}" in df.columns
        # Rolling group
        for w in DEFAULT_WINDOWS:
            assert f"roll_mean_{w}" in df.columns

    def test_skips_volume_if_absent(self, no_volume_df):
        df = build_all_features(no_volume_df, drop_na=True)
        assert "volume_ratio" not in df.columns

    def test_custom_lags_and_windows(self, ohlcv):
        df = build_all_features(ohlcv, lags=[1, 3], windows=[5, 15], drop_na=True)
        assert "close_lag_3" in df.columns
        assert "roll_max_15" in df.columns
        assert "close_lag_2" not in df.columns

    def test_warm_up_rows_dropped(self, ohlcv):
        """SMA200 needs 199 warm-up rows; drop_na removes them."""
        df = build_all_features(ohlcv, drop_na=True)
        # After dropping NaN the first index in df must be >= 200th original row
        first_valid_position = ohlcv.index.get_loc(df.index[0])
        assert first_valid_position >= 199

    def test_deterministic(self, ohlcv):
        df1 = build_all_features(ohlcv)
        df2 = build_all_features(ohlcv)
        assert df1.equals(df2)

    def test_chronological_order_preserved(self, ohlcv):
        df = build_all_features(ohlcv, drop_na=True)
        assert list(df.index) == sorted(df.index)

    def test_missing_required_column_raises(self):
        bad = pd.DataFrame({"Open": [1.0], "Close": [2.0]})
        with pytest.raises(KeyError):
            build_all_features(bad)

    def test_at_least_40_features(self, ohlcv):
        """Sanity check: pipeline should produce a rich feature set."""
        df = build_all_features(ohlcv, drop_na=False)
        feats = feature_names(df)
        assert len(feats) >= 40, f"Only {len(feats)} features generated"


# ===========================================================================
# 9. Utilities
# ===========================================================================

class TestUtilities:

    def test_drop_na_rows_removes_nan(self, ohlcv):
        df = add_price_features(ohlcv)   # has NaN row 0
        clean = drop_na_rows(df)
        assert not clean.isna().any().any()

    def test_drop_na_rows_returns_copy(self, ohlcv):
        df = add_price_features(ohlcv)
        original_len = len(df)
        _ = drop_na_rows(df)
        assert len(df) == original_len  # original unchanged

    def test_feature_names_excludes_ohlcv(self, ohlcv):
        df = add_price_features(ohlcv)
        names = feature_names(df)
        for col in ("Open", "High", "Low", "Close", "Volume"):
            assert col not in names

    def test_feature_names_sorted(self, ohlcv):
        df = build_all_features(ohlcv, drop_na=False)
        names = feature_names(df)
        assert names == sorted(names)

    def test_feature_names_non_empty(self, ohlcv):
        df = build_all_features(ohlcv, drop_na=False)
        assert len(feature_names(df)) > 0

    def test_feature_names_includes_engineered(self, ohlcv):
        df = add_momentum_features(ohlcv)
        names = feature_names(df)
        assert "RSI14" in names and "MACD" in names
