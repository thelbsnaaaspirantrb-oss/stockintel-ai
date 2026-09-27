"""
models/feature_engineering.py
------------------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Tabular feature engineering for tree-based and regression models
(Random Forest, XGBoost).

Design
------
- Input  : pd.Series of Close prices, chronological.
- Output : (X, y) numpy arrays suitable for sklearn / xgboost.
- Features are built from lag values, rolling statistics, and calendar
  features — no future data leaks into any row.
- The split between train and validation is always chronological (tail slice).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: Default lag depths used as input features.
DEFAULT_LAGS: list[int] = [1, 2, 3, 5, 10, 20]

#: Rolling window sizes for mean and std features.
DEFAULT_WINDOWS: list[int] = [5, 10, 20]


# ---------------------------------------------------------------------------
# Feature builder
# ---------------------------------------------------------------------------

def build_features(
    series: pd.Series,
    lags: list[int] = DEFAULT_LAGS,
    windows: list[int] = DEFAULT_WINDOWS,
    add_calendar: bool = True,
) -> pd.DataFrame:
    """
    Construct a tabular feature matrix from a Close-price series.

    Features created
    ----------------
    lag_N         : Close price N days ago  (for N in ``lags``).
    roll_mean_W   : Rolling mean over W days (for W in ``windows``).
    roll_std_W    : Rolling std  over W days (for W in ``windows``).
    log_return_1  : log(Close_t / Close_{t-1})  — 1-day log return.
    log_return_5  : log(Close_t / Close_{t-5})  — 5-day log return.
    pct_change_1  : Percentage change vs previous day.
    day_of_week   : 0–4 integer (Monday–Friday)   [if add_calendar].
    month         : 1–12 integer                   [if add_calendar].
    quarter       : 1–4 integer                    [if add_calendar].

    The target column is named ``target`` and equals Close_{t+1}
    (next-day close), shifted backwards by one row.  Rows with NaN
    (warm-up period) are dropped.

    Args:
        series       : pd.Series with DatetimeIndex, name='Close'.
        lags         : List of lag depths to include.
        windows      : List of rolling window sizes.
        add_calendar : Whether to include calendar features.

    Returns:
        pd.DataFrame with feature columns + 'target' column,
        NaN rows removed, chronological order preserved.
    """
    df = pd.DataFrame({"Close": series.values}, index=series.index)

    # Lag features
    for lag in lags:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Rolling statistics
    for w in windows:
        df[f"roll_mean_{w}"] = df["Close"].rolling(window=w).mean()
        df[f"roll_std_{w}"]  = df["Close"].rolling(window=w).std()

    # Return features
    df["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_5"] = np.log(df["Close"] / df["Close"].shift(5))
    df["pct_change_1"] = df["Close"].pct_change(1)

    # Calendar features
    if add_calendar and hasattr(df.index, "dayofweek"):
        df["day_of_week"] = df.index.dayofweek
        df["month"]       = df.index.month
        df["quarter"]     = df.index.quarter

    # Target: next-day close
    df["target"] = df["Close"].shift(-1)

    # Drop the raw Close from features (it would be a data leak for t+1)
    df = df.drop(columns=["Close"])

    # Drop NaN rows (warm-up + last row where target is NaN)
    df = df.dropna()

    return df


def get_X_y(feature_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Split a feature DataFrame (from :func:`build_features`) into X and y.

    Args:
        feature_df : DataFrame with a 'target' column.

    Returns:
        (X, y) where X has shape (n, n_features) and y has shape (n,).
    """
    y = feature_df["target"].to_numpy(dtype=float)
    X = feature_df.drop(columns=["target"]).to_numpy(dtype=float)
    return X, y


def chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    val_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split arrays into train and validation sets chronologically.

    The last ``val_size`` rows become the validation set.  No shuffling.

    Args:
        X        : Feature matrix, shape (n, n_features).
        y        : Target array,   shape (n,).
        val_size : Number of tail rows to use as validation.

    Returns:
        (X_train, X_val, y_train, y_val)

    Raises:
        ValueError : If val_size >= len(X).
    """
    if val_size >= len(X):
        raise ValueError(
            f"val_size={val_size} must be smaller than dataset length={len(X)}."
        )
    X_train, X_val = X[:-val_size], X[-val_size:]
    y_train, y_val = y[:-val_size], y[-val_size:]
    return X_train, X_val, y_train, y_val
