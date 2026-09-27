"""
utils/data_fetcher.py
---------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Purpose
-------
Fetches historical OHLCV stock data and company metadata from Yahoo Finance
for Indian (NSE) and US markets.

On cloud platforms (Streamlit Cloud, Render, Railway) Yahoo Finance can
rate-limit plain requests. This module works around that by:
    1. Attaching a browser-like User-Agent via a custom requests.Session
    2. Using Ticker.history() which is more robust than yf.download()
    3. Retrying up to 3 times with exponential back-off
    4. Falling back to yf.download() if Ticker.history() fails

Public API
----------
    fetch_indian_stock(ticker, period) -> pd.DataFrame | None
    fetch_us_stock(ticker, period)     -> pd.DataFrame | None
    fetch_ticker_info(ticker)          -> dict

Original project this module is derived from:
    ModelHub by BadakalaYashwanth
    https://github.com/BadakalaYashwanth/ModelHub
    Licensed under the MIT License — see LICENSE for details.
"""

import random
import time

import pandas as pd
import requests
import yfinance as yf


# ---------------------------------------------------------------------------
# Browser-like session — avoids Yahoo Finance rate-limiting
# ---------------------------------------------------------------------------

def _make_session() -> requests.Session:
    """
    Return a requests.Session configured to mimic a real browser.

    Returns:
        requests.Session: Session with browser-like headers.
    """
    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
            "Accept": (
                "text/html,application/xhtml+xml,application/xml;"
                "q=0.9,image/webp,*/*;q=0.8"
            ),
            "Accept-Language": "en-US,en;q=0.9",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Cache-Control": "max-age=0",
        }
    )
    return session


# ---------------------------------------------------------------------------
# Column normalisation
# ---------------------------------------------------------------------------

def _clean_dataframe(data: pd.DataFrame) -> pd.DataFrame | None:
    """
    Normalise a raw yfinance DataFrame into a consistent OHLCV structure.

    Args:
        data (pd.DataFrame): Raw DataFrame returned by yfinance.

    Returns:
        pd.DataFrame | None: Cleaned OHLCV DataFrame, or None on failure.
    """
    if data is None or data.empty:
        return None

    # Flatten multi-level columns: ('Close', 'AAPL') → 'Close'
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Remove duplicate column names after flattening
    data = data.loc[:, ~data.columns.duplicated()]

    # 'Adj Close' → 'Close' fallback
    if "Adj Close" in data.columns and "Close" not in data.columns:
        data = data.rename(columns={"Adj Close": "Close"})

    if "Close" not in data.columns:
        return None

    # Keep only standard OHLCV columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in data.columns]
    data = data[keep].copy()
    data.dropna(subset=["Close"], inplace=True)

    return data if not data.empty else None


# ---------------------------------------------------------------------------
# Core download with retry + fallback
# ---------------------------------------------------------------------------

def _fetch(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Download OHLCV data with retry loop and secondary fallback strategy.

    Args:
        ticker (str): Yahoo Finance ticker symbol.
        period (str): yfinance period string — '1y', '2y', '5y', etc.

    Returns:
        pd.DataFrame | None: Cleaned OHLCV DataFrame or None on failure.
    """
    session = _make_session()

    for attempt in range(1, 4):
        try:
            tk = yf.Ticker(ticker, session=session)
            data = tk.history(period=period, auto_adjust=True)
            cleaned = _clean_dataframe(data)
            if cleaned is not None:
                return cleaned
        except Exception as exc:
            print(
                f"[StockIntel AI | data_fetcher] "
                f"Ticker.history attempt {attempt} failed for {ticker}: {exc}"
            )

        try:
            data = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=True,
                threads=False,
                session=session,
            )
            cleaned = _clean_dataframe(data)
            if cleaned is not None:
                return cleaned
        except Exception as exc:
            print(
                f"[StockIntel AI | data_fetcher] "
                f"yf.download attempt {attempt} failed for {ticker}: {exc}"
            )

        wait = (2 ** attempt) + random.uniform(0, 1)
        print(
            f"[StockIntel AI | data_fetcher] "
            f"Waiting {wait:.1f}s before retry {attempt + 1} for {ticker}…"
        )
        time.sleep(wait)

    return None


# ---------------------------------------------------------------------------
# Public API — OHLCV fetchers
# ---------------------------------------------------------------------------

def fetch_indian_stock(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for an Indian (NSE) stock.

    Args:
        ticker (str): NSE ticker with .NS suffix (e.g. 'RELIANCE.NS').
        period (str): yfinance period string. Defaults to '2y'.

    Returns:
        pd.DataFrame | None: OHLCV DataFrame indexed by Date, or None.
    """
    return _fetch(ticker, period)


def fetch_us_stock(ticker: str, period: str = "2y") -> pd.DataFrame | None:
    """
    Fetch historical OHLCV data for a US-listed stock.

    Args:
        ticker (str): US ticker symbol (e.g. 'AAPL').
        period (str): yfinance period string. Defaults to '2y'.

    Returns:
        pd.DataFrame | None: OHLCV DataFrame indexed by Date, or None.
    """
    return _fetch(ticker, period)


# ---------------------------------------------------------------------------
# Public API — Company metadata
# ---------------------------------------------------------------------------

def fetch_ticker_info(ticker: str) -> dict:
    """
    Fetch company metadata and key statistics for the Stock Summary Card.

    Attempts to retrieve data from ``yf.Ticker.info``. Returns a dict with
    safe fallbacks (None or 'N/A') for every key so callers never crash on
    missing fields — Yahoo Finance does not always populate all fields.

    Keys returned
    -------------
    company_name    : str  — Long company name (e.g. 'Apple Inc.')
    sector          : str  — Business sector
    industry        : str  — Industry classification
    currency        : str  — Trading currency code (e.g. 'USD', 'INR')
    exchange        : str  — Exchange name
    market_cap      : float | None  — Market capitalisation in native currency
    pe_ratio        : float | None  — Trailing P/E ratio
    eps             : float | None  — Trailing twelve-month EPS
    dividend_yield  : float | None  — Annual dividend yield (0–1 float)
    week_52_high    : float | None  — 52-week high price
    week_52_low     : float | None  — 52-week low price
    avg_volume      : int | None    — 10-day average volume
    beta            : float | None  — Beta coefficient

    Args:
        ticker (str): Yahoo Finance ticker symbol.

    Returns:
        dict: Metadata dict with safe fallbacks. Never raises.
    """
    defaults: dict = {
        "company_name":   ticker,
        "sector":         "N/A",
        "industry":       "N/A",
        "currency":       "N/A",
        "exchange":       "N/A",
        "market_cap":     None,
        "pe_ratio":       None,
        "eps":            None,
        "dividend_yield": None,
        "week_52_high":   None,
        "week_52_low":    None,
        "avg_volume":     None,
        "beta":           None,
    }

    try:
        info = yf.Ticker(ticker).info

        def _get(key, fallback=None):
            v = info.get(key, fallback)
            return v if v not in (None, "", 0) or fallback is None else fallback

        defaults.update({
            "company_name":   _get("longName") or _get("shortName") or ticker,
            "sector":         _get("sector",         "N/A"),
            "industry":       _get("industry",       "N/A"),
            "currency":       _get("currency",       "N/A"),
            "exchange":       _get("exchange",       "N/A"),
            "market_cap":     info.get("marketCap"),
            "pe_ratio":       info.get("trailingPE"),
            "eps":            info.get("trailingEps"),
            "dividend_yield": info.get("dividendYield"),
            "week_52_high":   info.get("fiftyTwoWeekHigh"),
            "week_52_low":    info.get("fiftyTwoWeekLow"),
            "avg_volume":     info.get("averageVolume10days") or info.get("averageVolume"),
            "beta":           info.get("beta"),
        })
    except Exception as exc:
        print(f"[StockIntel AI | data_fetcher] fetch_ticker_info failed for {ticker}: {exc}")

    return defaults
