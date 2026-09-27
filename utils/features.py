"""
utils/features.py
-----------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Reusable, look-ahead-safe feature engineering pipeline for OHLCV stock data.

Design principles
-----------------
1. **No look-ahead bias** — every feature at row t uses only information
   available at or before time t.  This is enforced by:
   - Using only ``shift(n)`` with n ≥ 1 for lag features.
   - Using only ``rolling(window=w)`` (closed on the right — default pandas
     behaviour) so window [t-w+1, t] is used, never [t+1, …].
   - Shifting the target forward (``shift(-1)``) only for supervised-
     learning label construction, never mixed into feature computation.

2. **NaN handling** — warm-up NaN rows are left intact by default so callers
   can decide whether to drop or forward-fill.  Every function documents
   how many leading NaN rows its output introduces.

3. **Deterministic** — no random state; outputs depend only on the input
   DataFrame and explicit parameters.

4. **Composable** — each group (price, moving-average, momentum, volatility,
   volume, lag, rolling) is a standalone function.  ``build_all_features``
   composes them into one wide DataFrame in a single call.

5. **Separation of concerns** — this module handles feature computation only.
   Train/val splitting, scaling, and target construction live in
   ``models/feature_engineering.py``.

Public API
----------
    # Individual feature groups
    add_price_features(df)
    add_moving_averages(df)
    add_momentum_features(df)
    add_volatility_features(df)
    add_volume_features(df)
    add_lag_features(df, lags)
    add_rolling_features(df, windows)

    # All-in-one pipeline
    build_all_features(df, *, drop_na, lags, windows) -> pd.DataFrame

    # Utility
    drop_na_rows(df)    -> pd.DataFrame
    feature_names(df)   -> list[str]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Module-level defaults
# ---------------------------------------------------------------------------

#: Default lag depths (trading days) for lag-feature generation.
DEFAULT_LAGS: list[int] = [1, 2, 3, 5, 10]

#: Default rolling windows (trading days) for rolling-statistic features.
DEFAULT_WINDOWS: list[int] = [5, 10, 20]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _require_columns(df: pd.DataFrame, *cols: str) -> None:
    """Raise KeyError if any of ``cols`` are missing from ``df``."""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"Required column(s) missing from DataFrame: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _safe_div(a: pd.Series, b: pd.Series) -> pd.Series:
    """Element-wise division; returns NaN where denominator is zero."""
    return a / b.replace(0, np.nan)


# ===========================================================================
# 1. PRICE FEATURES
# ===========================================================================

def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add price-derived return and change features.

    All features use only past or current prices (no look-ahead).

    Features added
    --------------
    daily_return
        Arithmetic return: (Close_t − Close_{t-1}) / Close_{t-1}.
        Alias for ``pct_change``.  NaN on row 0.

    log_return
        Continuous (log) return: ln(Close_t / Close_{t-1}).
        Preferred over arithmetic returns for statistical modelling
        because log returns are additive over time.  NaN on row 0.

    price_change
        Absolute price change: Close_t − Close_{t-1}.
        NaN on row 0.

    pct_change
        Percentage price change: (Close_t − Close_{t-1}) / Close_{t-1} × 100.
        Same as daily_return scaled to percentage.  NaN on row 0.

    Args:
        df : DataFrame with at least a ``Close`` column.

    Returns:
        Copy of ``df`` with four new columns appended.
        Rows with NaN in the source are propagated as NaN (not dropped).

    Raises:
        KeyError : If ``Close`` is not present in ``df``.

    Example
    -------
    >>> df = add_price_features(ohlcv_df)
    >>> df[["daily_return", "log_return", "price_change", "pct_change"]].tail()
    """
    _require_columns(df, "Close")
    df = df.copy()
    prev = df["Close"].shift(1)

    df["daily_return"] = _safe_div(df["Close"] - prev, prev)
    df["log_return"]   = np.log(_safe_div(df["Close"], prev))
    df["price_change"] = df["Close"] - prev
    df["pct_change"]   = _safe_div(df["Close"] - prev, prev) * 100.0

    return df


# ===========================================================================
# 2. MOVING AVERAGES
# ===========================================================================

def add_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add Simple and Exponential Moving Average columns.

    Features added
    --------------
    SMA20, SMA50, SMA100, SMA200
        Simple Moving Averages over 20, 50, 100, and 200 periods.
        SMA_n is NaN for the first n-1 rows (warm-up period).

    EMA20, EMA50
        Exponential Moving Averages using pandas ewm (span-based,
        adjust=False so each new value is:
        EMA_t = α × Close_t + (1−α) × EMA_{t-1},  α = 2/(span+1)).
        EMA is defined from row 0 but converges after ~3 × span rows.

    No look-ahead: ``rolling()`` uses the default closed='right' which
    includes rows [t-window+1, t]; ``ewm()`` processes rows in order.

    Args:
        df : DataFrame with at least a ``Close`` column.

    Returns:
        Copy of ``df`` with six new columns appended.

    Raises:
        KeyError : If ``Close`` is not present in ``df``.
    """
    _require_columns(df, "Close")
    df = df.copy()
    close = df["Close"]

    # Simple Moving Averages
    for period in (20, 50, 100, 200):
        df[f"SMA{period}"] = close.rolling(window=period, min_periods=period).mean()

    # Exponential Moving Averages
    for span in (20, 50):
        df[f"EMA{span}"] = close.ewm(span=span, adjust=False).mean()

    return df


# ===========================================================================
# 3. MOMENTUM FEATURES
# ===========================================================================

def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add RSI, MACD family, and Stochastic Oscillator features.

    Features added
    --------------
    RSI14
        Relative Strength Index (14-period).
        Formula: RSI = 100 − 100/(1 + RS),  RS = avg_gain / avg_loss.
        Uses simple rolling mean (Wilder approximation).
        NaN for first 14 rows.

    MACD
        MACD line = EMA(12) − EMA(26).  Defined from row 0 but
        meaningful only after ~26 rows of convergence.

    MACD_signal
        Signal line = EMA(MACD, 9).

    MACD_hist
        Histogram = MACD − MACD_signal.
        Positive histogram → upward momentum.

    stoch_k
        Stochastic %K: (Close − LowestLow_14) / (HighestHigh_14 − LowestLow_14) × 100.
        NaN for first 14 rows.

    stoch_d
        Stochastic %D: SMA(stoch_k, 3) — slow signal line.
        NaN for first 14+3-1 = 16 rows.

    Args:
        df : DataFrame with ``Close``, ``High``, and ``Low`` columns.

    Returns:
        Copy of ``df`` with six new columns appended.

    Raises:
        KeyError : If any of ``Close``, ``High``, ``Low`` are missing.
    """
    _require_columns(df, "Close", "High", "Low")
    df = df.copy()

    # ── RSI (14-period) ──────────────────────────────────────────────────────
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = (-delta).clip(lower=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = _safe_div(avg_gain, avg_loss)
    df["RSI14"] = 100.0 - (100.0 / (1.0 + rs))

    # ── MACD family ──────────────────────────────────────────────────────────
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"]        = ema12 - ema26
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_hist"]   = df["MACD"] - df["MACD_signal"]

    # ── Stochastic Oscillator (%K / %D) ─────────────────────────────────────
    low14  = df["Low"].rolling(window=14, min_periods=14).min()
    high14 = df["High"].rolling(window=14, min_periods=14).max()
    denom  = (high14 - low14).replace(0, np.nan)
    df["stoch_k"] = ((df["Close"] - low14) / denom) * 100.0
    df["stoch_d"] = df["stoch_k"].rolling(window=3, min_periods=3).mean()

    return df


# ===========================================================================
# 4. VOLATILITY FEATURES
# ===========================================================================

def add_volatility_features(
    df: pd.DataFrame,
    bb_window: int = 20,
    bb_std:    float = 2.0,
    atr_period: int = 14,
    vol_window: int = 20,
) -> pd.DataFrame:
    """
    Add rolling volatility, ATR, and Bollinger Band features.

    Features added
    --------------
    rolling_std
        Rolling standard deviation of Close prices over ``vol_window`` days.
        Measures raw price dispersion (not annualised).
        NaN for first ``vol_window``-1 rows.

    ATR
        Average True Range (``atr_period``-period, EMA-smoothed).
        True Range = max(High−Low, |High−PrevClose|, |Low−PrevClose|).
        NaN on row 0 due to PrevClose shift.

    BB_upper, BB_lower, BB_mid
        Bollinger Bands (``bb_window``-period SMA ± ``bb_std`` × rolling σ).
        BB_mid  = SMA(Close, bb_window).
        BB_upper = BB_mid + bb_std × rolling_std(bb_window).
        BB_lower = BB_mid − bb_std × rolling_std(bb_window).
        NaN for first ``bb_window``-1 rows.

    BB_width
        Bandwidth normalised by mid band: (BB_upper − BB_lower) / BB_mid.
        Captures volatility expansion / contraction.

    Args:
        df         : DataFrame with ``Open``, ``High``, ``Low``, ``Close``.
        bb_window  : Bollinger Band lookback (default 20).
        bb_std     : Number of standard deviations (default 2.0).
        atr_period : ATR smoothing period (default 14).
        vol_window : Rolling std window (default 20).

    Returns:
        Copy of ``df`` with seven new columns appended.

    Raises:
        KeyError : If ``High``, ``Low``, or ``Close`` are missing.
    """
    _require_columns(df, "High", "Low", "Close")
    df = df.copy()

    # ── Rolling standard deviation ───────────────────────────────────────────
    df["rolling_std"] = df["Close"].rolling(
        window=vol_window, min_periods=vol_window
    ).std()

    # ── ATR ──────────────────────────────────────────────────────────────────
    prev_close = df["Close"].shift(1)
    hl  = df["High"] - df["Low"]
    hpc = (df["High"] - prev_close).abs()
    lpc = (df["Low"]  - prev_close).abs()
    true_range = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df["ATR"] = true_range.ewm(span=atr_period, adjust=False).mean()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_sma = df["Close"].rolling(window=bb_window, min_periods=bb_window).mean()
    bb_s   = df["Close"].rolling(window=bb_window, min_periods=bb_window).std()
    df["BB_mid"]   = bb_sma
    df["BB_upper"] = bb_sma + bb_std * bb_s
    df["BB_lower"] = bb_sma - bb_std * bb_s
    df["BB_width"] = _safe_div(df["BB_upper"] - df["BB_lower"], df["BB_mid"])

    return df


# ===========================================================================
# 5. VOLUME FEATURES
# ===========================================================================

def add_volume_features(
    df: pd.DataFrame,
    vol_sma_window: int = 20,
) -> pd.DataFrame:
    """
    Add volume-derived momentum and normalisation features.

    Features added
    --------------
    volume_change
        Absolute change in volume vs previous period:
        Volume_t − Volume_{t-1}.  NaN on row 0.

    volume_sma
        Simple Moving Average of volume over ``vol_sma_window`` periods.
        Smoothed baseline for comparing current volume.
        NaN for first ``vol_sma_window``-1 rows.

    volume_ratio
        Current volume relative to its SMA: Volume_t / volume_sma_t.
        Values > 1 indicate above-average volume (potential breakout signal).
        NaN wherever volume_sma is NaN or zero.

    Args:
        df            : DataFrame with a ``Volume`` column.
        vol_sma_window : Rolling window for volume SMA (default 20).

    Returns:
        Copy of ``df`` with three new columns appended.

    Raises:
        KeyError : If ``Volume`` is not present in ``df``.
    """
    _require_columns(df, "Volume")
    df = df.copy()

    df["volume_change"] = df["Volume"].diff()
    df["volume_sma"]    = df["Volume"].rolling(
        window=vol_sma_window, min_periods=vol_sma_window
    ).mean()
    df["volume_ratio"]  = _safe_div(df["Volume"], df["volume_sma"])

    return df


# ===========================================================================
# 6. LAG FEATURES
# ===========================================================================

def add_lag_features(
    df: pd.DataFrame,
    lags: list[int] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """
    Add lagged Close-price features.

    Each lag column ``close_lag_N`` holds the Close price from N trading
    days ago: close_lag_N_t = Close_{t-N}.  Because these use ``shift(N)``
    with N ≥ 1, they strictly use only past prices — no look-ahead.

    Default lags: 1, 2, 3, 5, 10.

    Warm-up NaN rows introduced:
        The first ``max(lags)`` rows will have at least one NaN lag column.

    Args:
        df   : DataFrame with at least a ``Close`` column.
        lags : List of positive integers specifying lag depths.
               All values must be ≥ 1 (enforced).

    Returns:
        Copy of ``df`` with ``len(lags)`` new columns appended.

    Raises:
        KeyError  : If ``Close`` is not present in ``df``.
        ValueError: If any lag value is < 1.
    """
    _require_columns(df, "Close")
    invalid = [l for l in lags if l < 1]
    if invalid:
        raise ValueError(
            f"All lag values must be >= 1. Got: {invalid}"
        )
    df = df.copy()
    for lag in lags:
        df[f"close_lag_{lag}"] = df["Close"].shift(lag)
    return df


# ===========================================================================
# 7. ROLLING FEATURES
# ===========================================================================

def add_rolling_features(
    df: pd.DataFrame,
    windows: list[int] = DEFAULT_WINDOWS,
) -> pd.DataFrame:
    """
    Add rolling mean, std, min, and max of Close prices.

    For each window W in ``windows``, four columns are added:

    roll_mean_W
        Rolling arithmetic mean of Close over W days.

    roll_std_W
        Rolling standard deviation of Close over W days.
        Measures local price dispersion.

    roll_min_W
        Rolling minimum Close over W days.
        Useful as a dynamic support level proxy.

    roll_max_W
        Rolling maximum Close over W days.
        Useful as a dynamic resistance level proxy.

    All rolling calculations use ``min_periods=W``, so the first W-1 rows
    of each column are NaN (strict warm-up — no partial-window leakage).

    No look-ahead: window is [t-W+1, t], never including t+1 or later.

    Args:
        df      : DataFrame with at least a ``Close`` column.
        windows : List of positive integer window sizes.

    Returns:
        Copy of ``df`` with ``4 × len(windows)`` new columns appended.

    Raises:
        KeyError  : If ``Close`` is not present in ``df``.
        ValueError: If any window value is < 1.
    """
    _require_columns(df, "Close")
    invalid = [w for w in windows if w < 1]
    if invalid:
        raise ValueError(f"All window values must be >= 1. Got: {invalid}")

    df = df.copy()
    close = df["Close"]

    for w in windows:
        r = close.rolling(window=w, min_periods=w)
        df[f"roll_mean_{w}"] = r.mean()
        df[f"roll_std_{w}"]  = r.std()
        df[f"roll_min_{w}"]  = r.min()
        df[f"roll_max_{w}"]  = r.max()

    return df


# ===========================================================================
# All-in-one pipeline
# ===========================================================================

def build_all_features(
    df: pd.DataFrame,
    *,
    lags:           list[int]  = DEFAULT_LAGS,
    windows:        list[int]  = DEFAULT_WINDOWS,
    bb_window:      int        = 20,
    bb_std:         float      = 2.0,
    atr_period:     int        = 14,
    vol_window:     int        = 20,
    vol_sma_window: int        = 20,
    drop_na:        bool       = True,
) -> pd.DataFrame:
    """
    Run the full feature-engineering pipeline in a single call.

    Applies, in order:
        1. ``add_price_features``
        2. ``add_moving_averages``
        3. ``add_momentum_features``
        4. ``add_volatility_features``
        5. ``add_volume_features``      (skipped if ``Volume`` absent)
        6. ``add_lag_features``
        7. ``add_rolling_features``

    Args:
        df             : Raw OHLCV DataFrame.  Must contain ``Open``,
                         ``High``, ``Low``, ``Close``; ``Volume`` optional.
        lags           : Lag depths for lag features (default [1,2,3,5,10]).
        windows        : Window sizes for rolling features (default [5,10,20]).
        bb_window      : Bollinger Band window (default 20).
        bb_std         : Bollinger Band std multiplier (default 2.0).
        atr_period     : ATR smoothing period (default 14).
        vol_window     : Rolling std window in volatility features (default 20).
        vol_sma_window : Volume SMA window (default 20).
        drop_na        : If True (default), drop all rows that contain any NaN
                         after feature construction.  Set to False to keep the
                         raw NaN rows for inspection.

    Returns:
        Wide DataFrame with all feature columns.  Original OHLCV columns
        are preserved.  Column order: OHLCV → price → MA → momentum →
        volatility → volume → lags → rolling.

    Raises:
        KeyError : If any required OHLCV column is missing.

    Notes
    -----
    **No look-ahead bias**: every feature at row t is computed solely from
    data in the closed interval [0, t].  The dominant warm-up cost is
    SMA200 (199 leading NaN rows), which ``drop_na=True`` removes.

    Example
    -------
    >>> import yfinance as yf
    >>> raw = yf.download("AAPL", period="2y", auto_adjust=True)
    >>> features = build_all_features(raw)
    >>> features.shape
    (N, 47+)   # exact count depends on lags and windows chosen
    """
    _require_columns(df, "Open", "High", "Low", "Close")

    out = df.copy()
    out = add_price_features(out)
    out = add_moving_averages(out)
    out = add_momentum_features(out)
    out = add_volatility_features(
        out,
        bb_window=bb_window,
        bb_std=bb_std,
        atr_period=atr_period,
        vol_window=vol_window,
    )
    if "Volume" in out.columns:
        out = add_volume_features(out, vol_sma_window=vol_sma_window)
    out = add_lag_features(out, lags=lags)
    out = add_rolling_features(out, windows=windows)

    if drop_na:
        out = out.dropna()

    return out


# ===========================================================================
# Utility
# ===========================================================================

def drop_na_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop all rows that contain at least one NaN value.

    Convenience wrapper around ``pd.DataFrame.dropna``.

    Args:
        df : Any DataFrame.

    Returns:
        Copy with NaN rows removed.
    """
    return df.dropna()


def feature_names(df: pd.DataFrame) -> list[str]:
    """
    Return a sorted list of all feature column names in ``df``.

    Excludes the standard OHLCV source columns (Open, High, Low, Close,
    Volume) so only engineered features are returned.

    Args:
        df : DataFrame produced by any feature function.

    Returns:
        Alphabetically sorted list of engineered feature column names.
    """
    ohlcv = {"Open", "High", "Low", "Close", "Volume"}
    return sorted(c for c in df.columns if c not in ohlcv)
