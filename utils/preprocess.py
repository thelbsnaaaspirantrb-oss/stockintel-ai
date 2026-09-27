"""
utils/preprocess.py
-------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Purpose
-------
Cleans raw OHLCV stock data, computes technical indicators, and reshapes
the data into LSTM-ready sequences. Called by app.py, training scripts,
and the test suite.

Public API
----------
    add_technical_indicators(df)             -> pd.DataFrame
    preprocess_for_lstm(df, sequence_length) -> (X, y, scaler)
    compute_macd(df)                         -> pd.DataFrame  (adds MACD cols)
    compute_stochastic(df)                   -> pd.DataFrame  (adds Stoch cols)
    compute_atr(df)                          -> pd.DataFrame  (adds ATR col)
    compute_volatility(df, window)           -> pd.DataFrame  (adds Volatility col)

Original project this module is derived from:
    ModelHub by BadakalaYashwanth
    https://github.com/BadakalaYashwanth/ModelHub
    Licensed under the MIT License — see LICENSE for details.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Core indicator suite (used by LSTM pipeline)
# ---------------------------------------------------------------------------

def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a raw OHLCV DataFrame with core technical indicators.

    Indicators added
    ----------------
    RSI        — 14-period Relative Strength Index.
    SMA20      — 20-period Simple Moving Average.
    SMA50      — 50-period Simple Moving Average.
    SMA200     — 200-period Simple Moving Average.
    EMA20      — 20-period Exponential Moving Average.
    BB_upper   — Upper Bollinger Band (SMA20 + 2σ, 20-period).
    BB_lower   — Lower Bollinger Band (SMA20 − 2σ, 20-period).
    BB_mid     — Middle band (= SMA20).

    NaN rows produced during the warm-up period are dropped before returning.

    Args:
        df (pd.DataFrame): Raw OHLCV DataFrame with at least a 'Close' column.

    Returns:
        pd.DataFrame: Copy with indicator columns appended, NaN rows dropped.

    Raises:
        KeyError: If 'Close' is not present in ``df``.
    """
    df = df.copy()

    # ── RSI (14-period) ──────────────────────────────────────────────────────
    delta    = df["Close"].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14, min_periods=14).mean()
    avg_loss = loss.rolling(window=14, min_periods=14).mean()
    rs       = avg_gain / (avg_loss + 1e-10)
    df["RSI"] = 100 - (100 / (1 + rs))

    # ── Simple Moving Averages ───────────────────────────────────────────────
    df["SMA20"]  = df["Close"].rolling(window=20).mean()
    df["SMA50"]  = df["Close"].rolling(window=50).mean()
    df["SMA200"] = df["Close"].rolling(window=200).mean()

    # ── Exponential Moving Average ───────────────────────────────────────────
    df["EMA20"] = df["Close"].ewm(span=20, adjust=False).mean()

    # ── Bollinger Bands (20-period ±2σ) ─────────────────────────────────────
    rolling_std  = df["Close"].rolling(window=20).std()
    df["BB_mid"]   = df["SMA20"]
    df["BB_upper"] = df["SMA20"] + (2 * rolling_std)
    df["BB_lower"] = df["SMA20"] - (2 * rolling_std)

    # Drop warm-up NaN rows
    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Extended indicators (used by dashboard analytics sections)
# ---------------------------------------------------------------------------

def compute_macd(
    df: pd.DataFrame,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD, Signal line, and Histogram columns.

    MACD    = EMA(fast) − EMA(slow)
    Signal  = EMA(MACD, signal)
    Hist    = MACD − Signal

    Args:
        df     : DataFrame with a 'Close' column.
        fast   : Fast EMA period (default 12).
        slow   : Slow EMA period (default 26).
        signal : Signal EMA period (default 9).

    Returns:
        pd.DataFrame: Copy with 'MACD', 'MACD_Signal', 'MACD_Hist' added.
    """
    df = df.copy()
    ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
    df["MACD"]        = ema_fast - ema_slow
    df["MACD_Signal"] = df["MACD"].ewm(span=signal, adjust=False).mean()
    df["MACD_Hist"]   = df["MACD"] - df["MACD_Signal"]
    return df


def compute_stochastic(
    df: pd.DataFrame,
    k_period: int = 14,
    d_period: int = 3,
) -> pd.DataFrame:
    """
    Compute Stochastic Oscillator %K and %D.

    %K = (Close − Lowest Low) / (Highest High − Lowest Low) × 100
    %D = SMA(%K, d_period)

    Args:
        df       : DataFrame with 'High', 'Low', 'Close' columns.
        k_period : Lookback window for %K (default 14).
        d_period : Smoothing for %D (default 3).

    Returns:
        pd.DataFrame: Copy with 'Stoch_K' and 'Stoch_D' added.
    """
    df = df.copy()
    low_min  = df["Low"].rolling(window=k_period).min()
    high_max = df["High"].rolling(window=k_period).max()
    denom    = (high_max - low_min).replace(0, np.nan)
    df["Stoch_K"] = ((df["Close"] - low_min) / denom) * 100
    df["Stoch_D"] = df["Stoch_K"].rolling(window=d_period).mean()
    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR) — a measure of market volatility.

    True Range = max(High−Low, |High−Prev Close|, |Low−Prev Close|)
    ATR        = EMA(True Range, period)

    Args:
        df     : DataFrame with 'High', 'Low', 'Close' columns.
        period : ATR smoothing period (default 14).

    Returns:
        pd.DataFrame: Copy with 'ATR' column added.
    """
    df = df.copy()
    prev_close = df["Close"].shift(1)
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - prev_close).abs(),
        (df["Low"]  - prev_close).abs(),
    ], axis=1).max(axis=1)
    df["ATR"] = tr.ewm(span=period, adjust=False).mean()
    return df


def compute_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Compute rolling annualised volatility from log returns.

    Volatility = std(log returns, window) × √252 × 100  [expressed as %]

    Args:
        df     : DataFrame with a 'Close' column.
        window : Rolling window in trading days (default 20).

    Returns:
        pd.DataFrame: Copy with 'Volatility' column added (annualised %).
    """
    df = df.copy()
    log_ret = np.log(df["Close"] / df["Close"].shift(1))
    df["Volatility"] = log_ret.rolling(window=window).std() * np.sqrt(252) * 100
    return df


# ---------------------------------------------------------------------------
# LSTM sequence builder (unchanged from original pipeline)
# ---------------------------------------------------------------------------

def preprocess_for_lstm(
    df: pd.DataFrame,
    sequence_length: int = 60,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler]:
    """
    Scale 'Close' prices and build sliding-window sequences for LSTM input.

    Args:
        df              : Processed DataFrame with a 'Close' column.
        sequence_length : Input window length (must match training; default 60).

    Returns:
        tuple:
            X (np.ndarray)        : Shape (n_samples, sequence_length, 1).
            y (np.ndarray)        : Shape (n_samples,) — scaled targets.
            scaler (MinMaxScaler) : Fitted scaler for inverse-transforming preds.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(df[["Close"]])

    X: list[np.ndarray] = []
    y: list[float] = []

    for i in range(sequence_length, len(scaled)):
        X.append(scaled[i - sequence_length : i, 0])
        y.append(scaled[i, 0])

    X_arr = np.array(X).reshape(-1, sequence_length, 1)
    y_arr = np.array(y)
    return X_arr, y_arr, scaler
