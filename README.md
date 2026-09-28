# 📈 StockIntel AI — Intelligent Stock Analytics & Forecasting Platform

> **StockIntel AI** is an open-source stock analytics and forecasting platform
> powered by a 2-layer LSTM neural network. It supports both **Indian (NSE)**
> and **US** equity markets, covers 200+ ticker symbols, and delivers
> interactive candlestick charts with full technical-indicator overlays and
> multi-day price forecasts.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red?logo=streamlit)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.20.0-orange?logo=tensorflow)
![License](https://img.shields.io/badge/License-MIT-green)
![Version](https://img.shields.io/badge/Version-1.0.0-informational)

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🌏 **Dual Market Support** | Indian (NSE — Nifty 50 + Nifty Next 50) and US (S&P 500 + Nasdaq top 50) |
| 🤖 **LSTM Forecasting** | Pre-trained 2-layer LSTM forecasts up to **30 days** ahead |
| 📊 **Interactive Charts** | Plotly dark-theme candlestick with zoom, pan, and unified hover |
| 📉 **Technical Indicators** | RSI, SMA 20/50, EMA 20, Bollinger Bands — all individually toggleable |
| 🔄 **Live Data** | Pulls up to 2 years of OHLCV history via `yfinance` with retry logic |
| 🗂️ **200+ Symbols** | Nifty 100 Indian stocks + top US equities in one dropdown |
| ⚙️ **Centralised Config** | All identity, UI, and ML constants live in `config.py` |
| 🎨 **Custom Dark Theme** | Electric-blue accent on deep-grey background via `.streamlit/config.toml` |
| ℹ️ **About Tab** | In-app credits, methodology, disclaimer, and original-project attribution |

---

## 🖼️ Screenshots

> *(Add screenshots here after first run)*

| Forecast Dashboard | Indicator Overlays |
|--------------------|--------------------|
| *(coming soon)*    | *(coming soon)*    |

---

## 🏗️ Project Structure

```
StockIntel-AI/
│
├── app.py                   # 🚀 Main Streamlit application (entry point)
├── config.py                # ⚙️  Centralised identity, UI, and ML constants
├── models.py                # 🧠 LSTM model architecture (build_lstm_model)
├── stock_lists.py           # 📋 Single source of truth for all ticker symbols
├── lstm_model.h5            # 💾 Pre-trained LSTM weights
├── requirements.txt         # 📦 Python dependencies
│
├── utils/
│   ├── __init__.py          # Package description
│   ├── data_fetcher.py      # 📡 yfinance wrappers with retry & rate-limit mitigation
│   └── preprocess.py        # 🔧 Technical indicators + LSTM sequence builder
│
├── test_data.py             # 🧪 Data-fetch smoke test (script)
├── test_models.py           # 🧪 Model-load smoke test (script)
├── test_pipeline.py         # 🧪 Full train-predict pipeline / model retraining script
│
├── .streamlit/
│   └── config.toml          # 🎨 Custom dark theme & server settings
│
├── CHANGELOG.md             # 📝 Version history
└── README.md
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **UI** | [Streamlit](https://streamlit.io/) |
| **Charts** | [Plotly](https://plotly.com/python/) |
| **ML Model** | TensorFlow / Keras — 2-layer LSTM |
| **Data** | [yfinance](https://github.com/ranaroussi/yfinance) (Yahoo Finance) |
| **Feature Engineering** | pandas · NumPy · scikit-learn |
| **Testing** | pytest |

---

## 🚀 Quick Start

### 1 — Clone the repository
```bash
git clone https://github.com/RohanBondre/StockIntel-AI.git
cd StockIntel-AI
```

### 2 — Create & activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3 — Install dependencies
```bash
pip install -r requirements.txt
```

### 4 — Run the app
```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🕹️ How to Use

1. **Select Market** — choose *Indian* or *US* from the sidebar.
2. **Pick a Stock** — select any ticker from the dropdown (200+ symbols).
3. **Set Forecast Horizon** — drag the slider (1–30 days).
4. **Toggle Overlays** — enable RSI, SMA, EMA, or Bollinger Bands.
5. **Click ⚡ Run Forecast** — the app:
   - Fetches live OHLCV data from Yahoo Finance
   - Computes technical indicators
   - Runs the LSTM for N-day forecasts
   - Renders a candlestick chart with colour-coded prediction segments  
     (🟢 green = up, 🔴 red = down)
   - Shows a Day-by-Day Forecast Summary panel on the right
6. **Show Raw OHLCV Data** — checkbox to view the underlying dataframe.
7. **About Tab** — methodology, credits, and legal disclaimer.

---

## 🧠 Model Architecture

```
Input  (60 trading days)
    →  LSTM(50, return_sequences=True)
    →  Dropout(0.2)
    →  LSTM(50, return_sequences=False)
    →  Dropout(0.2)
    →  Dense(25)
    →  Dense(1)   ← Predicted next-day close price (scaled 0–1)
```

- **Sequence length**: 60 trading days
- **Loss function**: Mean Squared Error
- **Optimiser**: Adam
- **Retraining**: `test_pipeline.py` — fetches 5 years of data, trains, saves `lstm_model.h5`

---

## 🔧 Technical Indicators

| Indicator | Period | Description |
|-----------|--------|-------------|
| RSI | 14 | Relative Strength Index — momentum oscillator |
| SMA20 / SMA50 | 20 / 50 | Simple Moving Averages — trend direction |
| EMA20 | 20 | Exponential Moving Average — faster trend signal |
| BB_upper / BB_lower | 20 ± 2σ | Bollinger Bands — volatility envelope |

All indicators are computed in `utils/preprocess.py` and are individually
toggled on/off from the sidebar.

---

## 🧪 Running Tests

```bash
# Smoke tests (script-style, print-based)
python test_data.py        # Verify data fetching returns correct columns/shape
python test_models.py      # Verify model builder loads without error
python test_pipeline.py    # Full retrain + save + predict (overwrites lstm_model.h5)
```

> **Note:** Proper `pytest` test suites with assertions and mocking are planned
> for Phase 7 of the roadmap.

---

## ☁️ Deployment

### Streamlit Community Cloud (recommended — free)

1. Push your fork to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and sign in with GitHub.
3. Click **New app** → select this repo → set **Main file path** to `app.py`.
4. Click **Deploy** — live in ~2 minutes.

### Docker

```bash
# Build
docker build -t stockintel-ai .

# Run
docker run -p 8501:8501 stockintel-ai
```

*(Dockerfile coming in Phase 8)*

---

## ⚠️ Disclaimer

StockIntel AI is for **informational and educational purposes only**.
Predictions are generated by machine learning models and do **not** constitute
financial advice. Always consult a qualified financial advisor before making
investment decisions. Past performance does not guarantee future results.

---

## 🤝 Contributing

Contributions are welcome!

1. Fork the repo.
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m 'Add your feature'`
4. Push: `git push origin feature/your-feature`
5. Open a Pull Request.

Please open an issue first for significant changes so we can discuss the approach.

---

## 📄 License

This project is licensed under the **MIT License** — see [`LICENSE`](LICENSE) for details.

---

## 🙏 Original Project / Acknowledgment

**StockIntel AI** is a substantially enhanced fork of
**[ModelHub](https://github.com/BadakalaYashwanth/ModelHub)**
by **[BadakalaYashwanth](https://github.com/BadakalaYashwanth)**.

The original ModelHub project provided the foundational data pipeline
(yfinance fetching with rate-limit mitigation), the LSTM preprocessing
pipeline, and the core Streamlit application structure that StockIntel AI
builds upon.

> Original repository: https://github.com/BadakalaYashwanth/ModelHub  
> Original license: MIT License — copyright remains with BadakalaYashwanth.  
> The MIT License text and original copyright notice are preserved in the
> [`LICENSE`](LICENSE) file in this repository, as required by the license terms.

We are grateful to BadakalaYashwanth for open-sourcing ModelHub and making
this work possible.

---

> Built with ❤️ by [Rohan Bondre](https://github.com/RohanBondre)  
> Computer Engineering Graduate · Software Development · Backend Development · AI/ML
