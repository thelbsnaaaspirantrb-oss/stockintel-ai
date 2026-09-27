# Changelog

All notable changes to **StockIntel AI** are documented here.

This project follows [Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/en/1.0.0/) format.

---

## [Unreleased]

Planned for upcoming phases:
- Multi-model comparison (LSTM vs GRU vs Linear Baseline)
- Expanded technical indicators: MACD, Stochastic %K/%D, OBV, ATR
- Prediction history persistence via SQLite (`utils/history_db.py`)
- News sentiment overlay (`utils/sentiment.py`)
- Monte Carlo Dropout for forecast confidence intervals
- Proper `pytest` suite with fixtures and mocking
- Docker / Streamlit Cloud deployment automation

---

## [1.0.0] — 2026-09-22

### 🎉 Fork Baseline — StockIntel AI

This release marks the initial transformation of the upstream
[ModelHub](https://github.com/BadakalaYashwanth/ModelHub) fork into
**StockIntel AI**, an independently branded and substantially enhanced
stock analytics platform.

### Added
- `config.py` — centralised application identity, UI constants, and ML
  hyperparameters (`APP_NAME`, `APP_TAGLINE`, `AUTHOR_NAME`, `APP_VERSION`,
  `GITHUB_URL`, `DATA_SOURCE`, `DISCLAIMER`, `SEQUENCE_LENGTH`, etc.)
- `.streamlit/config.toml` — custom dark theme (electric blue `#00C4FF`
  accent on `#0D1117` background)
- Professional sidebar: StockIntel AI logo block, tagline, author info,
  GitHub link, and data timestamp
- In-app **About** tab with tech stack, model architecture, indicator
  reference table, developer profile, and original-project attribution
- Chart overlay toggles for **SMA 20/50**, **EMA 20**, and
  **Bollinger Bands** (previously computed but never rendered)
- RSI overbought (70) / oversold (30) reference lines on the chart
- Candlestick `increasing_line_color` / `decreasing_line_color` set to
  standard green/red (`#26a641` / `#f85149`)
- `CHANGELOG.md` (this file)

### Changed
- `app.py` — full rewrite:
  - All hardcoded strings replaced with `config.py` imports
  - `page_title` updated to `"StockIntel AI | Intelligent Stock Analytics"`
  - `page_icon` set to `📈`
  - `build_forecast_chart()` now receives all parameters explicitly
    (fixes closure bug where `days_to_predict` was captured from global scope)
  - Stock lists imported from `stock_lists.py` (eliminates duplication)
  - Two-tab layout: **Forecast** and **About**
  - `run_forecast()` extracted as a clean, documented function
  - Legal disclaimer rendered from `config.DISCLAIMER`
- `models.py` — model named `"stockintel_lstm"`, docstrings updated
- `utils/data_fetcher.py` — log prefix updated to `[StockIntel AI | data_fetcher]`,
  docstrings and module header updated
- `utils/preprocess.py` — full docstring rewrite with inline algorithm notes
- `utils/__init__.py` — updated package description
- `stock_lists.py` — converted to typed `list[str]`, section comments added,
  module header updated; this is now the **single source of truth** for tickers
- `README.md` — complete rewrite: StockIntel AI branding, updated project
  structure, deployment instructions, and full attribution/acknowledgment section

### Removed
- `app1.py` — deleted; was a broken earlier draft with an unclosed parenthesis
  syntax error. All useful functionality is present in `app.py`.

### Fixed
- `app.py` closure bug: `days_to_predict` is now passed explicitly to
  `build_forecast_chart()` rather than captured from the outer scope
- Chart RSI y2-axis conditional layout applied only when RSI is toggled on
  (previously an empty dict was always merged into layout, causing a Plotly
  warning)

---

## [0.1.0] — Fork Origin

Upstream source: **ModelHub** by BadakalaYashwanth  
Repository: https://github.com/BadakalaYashwanth/ModelHub  
License: MIT

> This entry records the state of the codebase at the point of forking.
> No code changes were made at this version — it is purely a reference marker.

### Upstream features at fork point
- Streamlit app (`app.py`) with LSTM forecast for Indian and US stocks
- 2-layer LSTM architecture in `models.py`
- yfinance data fetching with browser-UA spoofing and exponential retry
  (`utils/data_fetcher.py`)
- Technical indicators: RSI, SMA20/50, EMA20, Bollinger Bands
  (`utils/preprocess.py`)
- 200+ ticker symbols (`stock_lists.py`)
- Pre-trained model weights (`lstm_model.h5`)
- Script-style smoke tests (`test_data.py`, `test_models.py`, `test_pipeline.py`)
