"""
config.py
---------
Centralized configuration for StockIntel AI.

All user-facing identity, metadata, and application-wide constants live here.
Import from this module instead of hardcoding values throughout the application.

Usage:
    from config import APP_NAME, APP_TAGLINE, AUTHOR_NAME, DISCLAIMER
"""

# ---------------------------------------------------------------------------
# Application Identity
# ---------------------------------------------------------------------------
APP_NAME: str = "StockIntel AI"
APP_TAGLINE: str = "Intelligent Stock Analytics & Forecasting Platform"
APP_VERSION: str = "1.1.0"
APP_ICON: str = "📈"

# ---------------------------------------------------------------------------
# Author & Project Links
# ---------------------------------------------------------------------------
AUTHOR_NAME: str = "Rohan Bondre"
AUTHOR_ROLE: str = "Computer Engineering Graduate"
AUTHOR_INTERESTS: str = "Software Development · Backend Development · AI/ML"
GITHUB_URL: str = "https://github.com/RohanBondre/StockIntel-AI"
LINKEDIN_URL: str = ""   # add your LinkedIn URL here

# ---------------------------------------------------------------------------
# Original Project Attribution (MIT License requirement)
# ---------------------------------------------------------------------------
ORIGINAL_PROJECT_NAME: str = "ModelHub"
ORIGINAL_AUTHOR: str = "BadakalaYashwanth"
ORIGINAL_REPO_URL: str = "https://github.com/BadakalaYashwanth/ModelHub"
ORIGINAL_LICENSE: str = "MIT"

# ---------------------------------------------------------------------------
# Data Source
# ---------------------------------------------------------------------------
DATA_SOURCE: str = "Yahoo Finance (via yfinance)"
DEFAULT_FETCH_PERIOD: str = "2y"
SEQUENCE_LENGTH: int = 60
MAX_FORECAST_DAYS: int = 30
DEFAULT_FORECAST_DAYS: int = 7

# Historical period choices shown in the sidebar selector
HISTORICAL_PERIODS: dict[str, str] = {
    "1 Month":  "1mo",
    "3 Months": "3mo",
    "6 Months": "6mo",
    "1 Year":   "1y",
    "2 Years":  "2y",
    "5 Years":  "5y",
}
DEFAULT_HIST_PERIOD: str = "2 Years"

# ---------------------------------------------------------------------------
# ML Models available in the selector
# ---------------------------------------------------------------------------
# "Auto" runs every available model and selects the lowest-RMSE result.
# The order here controls the sidebar dropdown order.
MODEL_OPTIONS: list[str] = [
    "Auto",
    "LSTM",
    "Random Forest",
    "XGBoost",
    "ARIMA",
    "Baseline (Naive)",
]
DEFAULT_MODEL: str = "Auto"

# Maps selector label → model_registry key  (label == registry key here,
# but keeping explicit mapping avoids fragile string equality elsewhere)
MODEL_DISPLAY_TO_KEY: dict[str, str] = {
    "Auto":             "Auto",
    "LSTM":             "LSTM",
    "Random Forest":    "Random Forest",
    "XGBoost":          "XGBoost",
    "ARIMA":            "ARIMA",
    "Baseline (Naive)": "Baseline (Naive)",
}

# Validation split size (trading days) used for all model evaluations
MODEL_VAL_SIZE: int = 30

# ---------------------------------------------------------------------------
# Markets
# ---------------------------------------------------------------------------
SUPPORTED_MARKETS: list[str] = ["Indian", "US"]
CURRENCY_SYMBOL: dict[str, str] = {
    "Indian": "₹",
    "US": "$",
}

# ---------------------------------------------------------------------------
# UI / Chart
# ---------------------------------------------------------------------------
CHART_TEMPLATE: str = "plotly_dark"

# Individual chart heights (px)
CHART_HEIGHT_PRICE: int = 520       # main price / candlestick chart
CHART_HEIGHT_FORECAST: int = 420    # forecast continuation chart
CHART_HEIGHT_RSI: int = 220         # RSI sub-chart
CHART_HEIGHT_MACD: int = 240        # MACD sub-chart
CHART_HEIGHT_INDICATORS: int = 300  # SMA/EMA/BB overview chart
CHART_HEIGHT_VOL: int = 200         # volatility / volume chart

# Kept for backward compatibility
CHART_HEIGHT: int = CHART_HEIGHT_PRICE
HISTORICAL_DISPLAY_DAYS: int = 120  # candlestick lookback shown on price chart

# Brand colours (also defined in .streamlit/config.toml)
COLOR_PRIMARY: str = "#00C4FF"      # electric blue accent
COLOR_UP: str = "#26a641"           # bullish green
COLOR_DOWN: str = "#f85149"         # bearish red
COLOR_NEUTRAL: str = "#8B949E"      # muted text / neutral
COLOR_SURFACE: str = "#161B22"      # card background
COLOR_BORDER: str = "#21262D"       # card border
COLOR_WARNING: str = "#E3B341"      # amber — disclaimer / warning

# Indicator line colours
COLOR_SMA20: str = "#E3B341"
COLOR_SMA50: str = "#FF7B72"
COLOR_SMA200: str = "#79C0FF"
COLOR_EMA20: str = "#D2A8FF"
COLOR_BB: str = "rgba(100,180,255,0.35)"
COLOR_RSI: str = "#A371F7"
COLOR_MACD: str = "#00C4FF"
COLOR_SIGNAL: str = "#FF7B72"
COLOR_HIST_POS: str = "#26a641"
COLOR_HIST_NEG: str = "#f85149"
COLOR_VOLUME: str = "rgba(100,180,255,0.25)"

# ---------------------------------------------------------------------------
# Legal Disclaimer
# ---------------------------------------------------------------------------
DISCLAIMER: str = (
    "⚠️ **Disclaimer:** StockIntel AI is provided for **informational and "
    "educational purposes only**. All forecasts are generated by experimental "
    "machine learning models and do **not** constitute financial advice, "
    "investment recommendations, or guarantees of future performance. "
    "Past performance is not indicative of future results. Always consult a "
    "qualified financial advisor before making any investment decisions."
)
