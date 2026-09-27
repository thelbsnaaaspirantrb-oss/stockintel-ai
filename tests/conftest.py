"""
tests/conftest.py
-----------------
StockIntel AI — shared pytest fixtures.

Provides deterministic, synthetic OHLCV data so tests run offline,
quickly, and without network calls.  All series are chronological
business-day DatetimeIndex — matching the real yfinance output format.
"""

import numpy as np
import pandas as pd
import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_price_series(
    n: int = 300,
    start_price: float = 100.0,
    seed: int = 42,
    freq: str = "B",          # business-day frequency
    start: str = "2022-01-01",
) -> pd.Series:
    """
    Generate a synthetic Close-price series using geometric Brownian motion.

    Parameters
    ----------
    n           : Number of observations.
    start_price : Starting price.
    seed        : NumPy RNG seed for reproducibility.
    freq        : Pandas frequency string.
    start       : Start date string.

    Returns
    -------
    pd.Series with DatetimeIndex, name='Close', no NaN.
    """
    rng = np.random.default_rng(seed)
    # Daily log returns ~ N(0.0003, 0.015)  — realistic equity vol
    log_returns = rng.normal(loc=0.0003, scale=0.015, size=n)
    prices = start_price * np.exp(np.cumsum(log_returns))
    prices = np.maximum(prices, 1.0)   # guard against non-positive prices
    index = pd.date_range(start=start, periods=n, freq=freq)
    return pd.Series(prices, index=index, name="Close")


def _make_ohlcv(
    n: int = 300,
    seed: int = 42,
    start: str = "2022-01-01",
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame.

    Open/High/Low are derived from Close with small random offsets.
    Volume is random integers in [500k, 5M].
    """
    close = _make_price_series(n=n, seed=seed, start=start)
    rng   = np.random.default_rng(seed + 1)

    open_  = close * (1 + rng.normal(0, 0.003, n))
    high   = np.maximum(close, open_) * (1 + rng.uniform(0, 0.005, n))
    low    = np.minimum(close, open_) * (1 - rng.uniform(0, 0.005, n))
    volume = rng.integers(500_000, 5_000_000, n).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low,
         "Close": close.values, "Volume": volume},
        index=close.index,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def close_series_300() -> pd.Series:
    """300-day synthetic Close series — sufficient for all models."""
    return _make_price_series(n=300)


@pytest.fixture(scope="session")
def close_series_500() -> pd.Series:
    """500-day synthetic Close series — used for LSTM (needs 60+30+buffer)."""
    return _make_price_series(n=500)


@pytest.fixture(scope="session")
def close_series_short() -> pd.Series:
    """50-day synthetic Close series — used to test insufficient-data errors."""
    return _make_price_series(n=50)


@pytest.fixture(scope="session")
def ohlcv_300() -> pd.DataFrame:
    """300-day synthetic OHLCV DataFrame."""
    return _make_ohlcv(n=300)


@pytest.fixture(scope="session")
def ohlcv_500() -> pd.DataFrame:
    """500-day synthetic OHLCV DataFrame."""
    return _make_ohlcv(n=500)


# Expose the helper at module level so individual test files can also use it
make_price_series = _make_price_series
make_ohlcv        = _make_ohlcv
