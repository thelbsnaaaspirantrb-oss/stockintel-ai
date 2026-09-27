"""
app.py
------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Professional financial analytics dashboard featuring:
  • Live stock summary with company metadata
  • Interactive candlestick price chart with indicator overlays
  • LSTM-powered multi-day price forecast
  • Technical analysis panels (RSI, MACD, Stochastic, Bollinger Bands,
    ATR, Volatility, Volume)
  • Model performance metrics (MAE, RMSE, MAPE, R²)
  • Expandable historical OHLCV dataframe
  • One-click CSV / report downloads
  • Legal disclaimer

Original project this fork is based on:
    ModelHub by BadakalaYashwanth
    https://github.com/BadakalaYashwanth/ModelHub
    Licensed under the MIT License — see LICENSE for details.
"""

# ---------------------------------------------------------------------------
# Standard library
# ---------------------------------------------------------------------------
import io
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Third-party
# ---------------------------------------------------------------------------
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
try:
    from tensorflow.keras.models import load_model as _keras_load
    _TF_AVAILABLE = True
except ImportError:
    _TF_AVAILABLE = False
    _keras_load = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------
from config import (
    APP_ICON, APP_NAME, APP_TAGLINE, APP_VERSION,
    AUTHOR_INTERESTS, AUTHOR_NAME, AUTHOR_ROLE,
    CHART_TEMPLATE,
    CHART_HEIGHT_PRICE, CHART_HEIGHT_FORECAST,
    CHART_HEIGHT_RSI, CHART_HEIGHT_MACD,
    CHART_HEIGHT_INDICATORS, CHART_HEIGHT_VOL,
    COLOR_BB, COLOR_BORDER, COLOR_DOWN, COLOR_EMA20,
    COLOR_HIST_NEG, COLOR_HIST_POS,
    COLOR_MACD, COLOR_NEUTRAL, COLOR_PRIMARY,
    COLOR_RSI, COLOR_SIGNAL, COLOR_SMA20,
    COLOR_SMA200, COLOR_SMA50, COLOR_SURFACE,
    COLOR_UP, COLOR_VOLUME, COLOR_WARNING,
    CURRENCY_SYMBOL, DATA_SOURCE,
    DEFAULT_FORECAST_DAYS, DEFAULT_HIST_PERIOD, DEFAULT_MODEL,
    DISCLAIMER, GITHUB_URL,
    HISTORICAL_DISPLAY_DAYS, HISTORICAL_PERIODS,
    MAX_FORECAST_DAYS, MODEL_OPTIONS, MODEL_DISPLAY_TO_KEY, MODEL_VAL_SIZE,
    ORIGINAL_AUTHOR, ORIGINAL_PROJECT_NAME, ORIGINAL_REPO_URL,
    SEQUENCE_LENGTH,
)
from stock_lists import indian_stocks, us_stocks
from utils.data_fetcher import fetch_indian_stock, fetch_us_stock, fetch_ticker_info
from utils.preprocess import (
    add_technical_indicators,
    compute_macd,
    compute_stochastic,
    compute_atr,
    compute_volatility,
    preprocess_for_lstm,
)
from models.model_registry import (
    get_model,
    available_models,
    run_all_models,
    auto_select,
)
from evaluation.model_comparison import (
    ModelComparisonReport,
    ModelEvaluationResult,
    compare_models,
)

# ===========================================================================
# Page configuration  (must be the very first Streamlit call)
# ===========================================================================
st.set_page_config(
    page_title=f"{APP_NAME} | Intelligent Stock Analytics",
    page_icon=APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": GITHUB_URL,
        "Report a bug": f"{GITHUB_URL}/issues",
        "About": (
            f"**{APP_NAME}** v{APP_VERSION}\n\n"
            f"{APP_TAGLINE}\n\n"
            f"Built by {AUTHOR_NAME}."
        ),
    },
)

# ===========================================================================
# CSS — polished dark financial theme
# ===========================================================================
st.markdown(
    f"""
    <style>
    /* ── Global ─────────────────────────────────────────────────────────── */
    html, body, [data-testid="stAppViewContainer"] {{
        background-color: #0D1117;
        color: #E6EDF3;
    }}
    [data-testid="stSidebar"] {{
        background-color: #0D1117;
        border-right: 1px solid {COLOR_BORDER};
    }}

    /* ── Hero header ─────────────────────────────────────────────────────── */
    .si-hero {{
        background: linear-gradient(135deg, #0D1117 0%, #161B22 50%, #0D1117 100%);
        border: 1px solid {COLOR_BORDER};
        border-radius: 12px;
        padding: 1.6rem 2rem 1.4rem 2rem;
        margin-bottom: 1.2rem;
        display: flex;
        align-items: center;
        gap: 1.2rem;
    }}
    .si-hero-icon {{ font-size: 2.8rem; line-height: 1; }}
    .si-hero-title {{
        font-size: 2rem;
        font-weight: 800;
        color: {COLOR_PRIMARY};
        letter-spacing: -0.5px;
        margin: 0;
        line-height: 1.1;
    }}
    .si-hero-tagline {{
        font-size: 0.88rem;
        color: {COLOR_NEUTRAL};
        margin: 0.2rem 0 0 0;
    }}
    .si-hero-badges {{
        margin-left: auto;
        text-align: right;
        font-size: 0.72rem;
        color: {COLOR_NEUTRAL};
        line-height: 1.8;
    }}

    /* ── Summary card row ────────────────────────────────────────────────── */
    .si-card {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 0.9rem 1.1rem;
        height: 100%;
    }}
    .si-card-label {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: {COLOR_NEUTRAL};
        margin-bottom: 0.25rem;
    }}
    .si-card-value {{
        font-size: 1.35rem;
        font-weight: 700;
        color: #E6EDF3;
        margin: 0;
    }}
    .si-card-sub {{
        font-size: 0.75rem;
        color: {COLOR_NEUTRAL};
        margin-top: 0.15rem;
    }}
    .si-up   {{ color: {COLOR_UP};   font-weight: 600; }}
    .si-down {{ color: {COLOR_DOWN}; font-weight: 600; }}
    .si-neutral {{ color: {COLOR_NEUTRAL}; }}

    /* ── Section headers ─────────────────────────────────────────────────── */
    .si-section {{
        font-size: 1.05rem;
        font-weight: 700;
        color: #E6EDF3;
        border-left: 3px solid {COLOR_PRIMARY};
        padding-left: 0.7rem;
        margin: 1.6rem 0 0.7rem 0;
    }}

    /* ── Metric cards (st.metric override) ──────────────────────────────── */
    [data-testid="metric-container"] {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 8px;
        padding: 0.7rem 0.9rem;
    }}

    /* ── Performance metric row ──────────────────────────────────────────── */
    .perf-card {{
        background: {COLOR_SURFACE};
        border: 1px solid {COLOR_BORDER};
        border-radius: 10px;
        padding: 1rem 1.2rem;
        text-align: center;
    }}
    .perf-label {{
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: {COLOR_NEUTRAL};
    }}
    .perf-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {COLOR_PRIMARY};
        margin: 0.2rem 0 0 0;
    }}
    .perf-desc {{
        font-size: 0.7rem;
        color: {COLOR_NEUTRAL};
        margin-top: 0.1rem;
    }}

    /* ── Forecast summary table ──────────────────────────────────────────── */
    .fc-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.35rem 0;
        border-bottom: 1px solid {COLOR_BORDER};
        font-size: 0.85rem;
    }}
    .fc-row:last-child {{ border-bottom: none; }}
    .fc-label {{ color: {COLOR_NEUTRAL}; }}
    .fc-val   {{ font-weight: 600; color: #E6EDF3; }}

    /* ── Sidebar labels ──────────────────────────────────────────────────── */
    .sb-section {{
        font-size: 0.68rem;
        text-transform: uppercase;
        letter-spacing: 0.09em;
        color: {COLOR_NEUTRAL};
        margin: 1rem 0 0.25rem 0;
    }}

    /* ── Disclaimer ──────────────────────────────────────────────────────── */
    .si-disclaimer {{
        background: #1C2128;
        border-left: 3px solid {COLOR_WARNING};
        border-radius: 0 8px 8px 0;
        padding: 0.8rem 1rem;
        font-size: 0.8rem;
        color: {COLOR_NEUTRAL};
        margin-top: 1.4rem;
    }}

    /* ── Divider override ────────────────────────────────────────────────── */
    hr {{ border-color: {COLOR_BORDER} !important; }}

    /* ── Tab strip ───────────────────────────────────────────────────────── */
    [data-testid="stTabs"] [role="tablist"] {{
        border-bottom: 1px solid {COLOR_BORDER};
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# ── SIDEBAR ─────────────────────────────────────────────────────────────────
# ===========================================================================
with st.sidebar:
    # Brand block
    st.markdown(
        f"""
        <div style="text-align:center;padding:0.5rem 0 1rem 0;">
          <div style="font-size:2.4rem;">{APP_ICON}</div>
          <div style="font-size:1.2rem;font-weight:800;
                      color:{COLOR_PRIMARY};letter-spacing:-0.3px;">
            {APP_NAME}
          </div>
          <div style="font-size:0.68rem;color:{COLOR_NEUTRAL};
                      margin-top:0.15rem;line-height:1.5;">
            {APP_TAGLINE}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Market ───────────────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">🌏 Market</p>', unsafe_allow_html=True)
    market = st.radio(
        "Market", options=["Indian", "US"],
        horizontal=True, label_visibility="collapsed",
    )

    # ── Stock search / select ────────────────────────────────────────────────
    st.markdown('<p class="sb-section">🔍 Stock</p>', unsafe_allow_html=True)
    stock_list = indian_stocks if market == "Indian" else us_stocks
    ticker = st.selectbox(
        "Select Stock", options=stock_list, label_visibility="collapsed",
    )

    # ── Historical period ────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">📅 Historical Period</p>', unsafe_allow_html=True)
    hist_period_label = st.selectbox(
        "Period", options=list(HISTORICAL_PERIODS.keys()),
        index=list(HISTORICAL_PERIODS.keys()).index(DEFAULT_HIST_PERIOD),
        label_visibility="collapsed",
    )
    hist_period = HISTORICAL_PERIODS[hist_period_label]

    # ── Forecast horizon ─────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">🔮 Forecast Horizon</p>', unsafe_allow_html=True)
    days_to_predict = st.slider(
        "Days to Forecast",
        min_value=1, max_value=MAX_FORECAST_DAYS,
        value=DEFAULT_FORECAST_DAYS,
    )

    # ── Model selector ───────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">🧠 Model</p>', unsafe_allow_html=True)
    _avail = available_models()
    _avail_display = ["Auto"] + [m for m in MODEL_OPTIONS if m != "Auto" and
                                  (m in _avail or m == "LSTM")]
    selected_model = st.selectbox(
        "Model", options=_avail_display,
        index=0,
        label_visibility="collapsed",
        help=(
            "**Auto** — evaluates all available models and picks the best "
            "by validation RMSE.\n\n"
            "Individual models are trained fresh on your selected period "
            "with a chronological train/validation split."
        ),
    )

    # ── Overlay toggles ──────────────────────────────────────────────────────
    st.markdown('<p class="sb-section">📊 Chart Overlays</p>', unsafe_allow_html=True)
    show_sma20  = st.checkbox("SMA 20",           value=True)
    show_sma50  = st.checkbox("SMA 50",           value=True)
    show_sma200 = st.checkbox("SMA 200",          value=False)
    show_ema20  = st.checkbox("EMA 20",           value=False)
    show_bb     = st.checkbox("Bollinger Bands",  value=False)
    show_volume = st.checkbox("Volume bars",      value=True)

    st.divider()

    # ── Run analysis button ──────────────────────────────────────────────────
    run_clicked = st.button(
        "⚡ Run Analysis", use_container_width=True, type="primary",
    )

    # ── Dedicated model evaluation controls ─────────────────────────────────
    st.markdown(
        '<p class="sb-section">🧪 Model Evaluation</p>',
        unsafe_allow_html=True,
    )
    evaluation_method = st.selectbox(
        "Validation method",
        options=["Walk-forward", "Expanding window"],
        key="evaluation_method",
    )
    evaluation_models = st.multiselect(
        "Models to evaluate",
        options=[m for m in MODEL_OPTIONS if m != "Auto" and m in _avail],
        default=[m for m in MODEL_OPTIONS if m != "Auto" and m in _avail],
        key="evaluation_models",
    )
    evaluation_metric = st.selectbox(
        "Ranking metric",
        options=["RMSE", "MAE", "MSE", "MAPE", "R2"],
        key="evaluation_metric",
    )
    evaluation_initial_train = st.number_input(
        "Initial training observations",
        min_value=1,
        value=120,
        step=10,
        key="evaluation_initial_train",
    )
    evaluation_window = st.number_input(
        "Validation window",
        min_value=1,
        value=5,
        step=1,
        key="evaluation_window",
    )
    evaluation_step = st.number_input(
        "Validation step",
        min_value=1,
        value=5,
        step=1,
        key="evaluation_step",
    )
    evaluation_test = st.checkbox(
        "Evaluate held-out test performance",
        value=True,
        key="evaluation_test",
    )
    run_evaluation = st.button(
        "🧪 Run Model Evaluation",
        use_container_width=True,
        key="run_model_evaluation",
    )

    # ── Footer ───────────────────────────────────────────────────────────────
    st.divider()
    st.markdown(
        f"""
        <div style="font-size:0.68rem;color:{COLOR_NEUTRAL};line-height:1.8;">
          <b style="color:#E6EDF3;">{APP_NAME}</b> v{APP_VERSION}<br>
          By <b style="color:#E6EDF3;">{AUTHOR_NAME}</b> · {AUTHOR_ROLE}<br>
          <a href="{GITHUB_URL}" target="_blank"
             style="color:{COLOR_PRIMARY};text-decoration:none;">
            🔗 GitHub
          </a><br>
          <span>Data: {DATA_SOURCE}</span><br>
          <span>Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# ── CACHED HELPERS ────────────────────────────────────────────────────────────
# ===========================================================================

@st.cache_resource
def _cached_keras_model():
    """Load and cache the raw Keras model for the LSTM wrapper."""
    if not _TF_AVAILABLE or _keras_load is None:
        return None
    import os
    path = os.path.join(os.path.dirname(__file__), "lstm_model.h5")
    return _keras_load(path) if os.path.exists(path) else None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(ticker: str, period: str) -> pd.DataFrame | None:
    """Cached OHLCV fetch — TTL 1 hour."""
    # Route by suffix rather than market string so the function signature
    # stays stable as a cache key regardless of market label changes.
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return fetch_indian_stock(ticker, period)
    return fetch_us_stock(ticker, period)


@st.cache_data(ttl=3600, show_spinner=False)
def cached_info(ticker: str) -> dict:
    """Cached company metadata — TTL 1 hour."""
    return fetch_ticker_info(ticker)


# ===========================================================================
# ── CHART BUILDERS ────────────────────────────────────────────────────────────
# ===========================================================================

def _build_price_chart(
    df: pd.DataFrame,
    ticker: str,
    currency: str,
    show_sma20: bool,
    show_sma50: bool,
    show_sma200: bool,
    show_ema20: bool,
    show_bb: bool,
    show_volume: bool,
    display_days: int = HISTORICAL_DISPLAY_DAYS,
) -> go.Figure:
    """
    Interactive candlestick price chart with optional indicator overlays
    and a volume sub-plot.
    """
    sl = df.iloc[-display_days:]

    if show_volume:
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            row_heights=[0.78, 0.22],
            vertical_spacing=0.02,
        )
        price_row, vol_row = 1, 2
    else:
        fig = go.Figure()
        price_row = None   # signal to use plain fig.add_trace

    def _add(trace, row=None):
        if price_row is not None:
            fig.add_trace(trace, row=row or price_row, col=1)
        else:
            fig.add_trace(trace)

    # Bollinger Bands (drawn first so candlestick sits on top)
    if show_bb and "BB_upper" in sl.columns and "BB_lower" in sl.columns:
        _add(go.Scatter(
            x=sl.index, y=sl["BB_upper"], name="BB Upper",
            line=dict(color=COLOR_BB, width=1), showlegend=True,
        ))
        _add(go.Scatter(
            x=sl.index, y=sl["BB_lower"], name="BB Lower",
            line=dict(color=COLOR_BB, width=1),
            fill="tonexty", fillcolor="rgba(100,180,255,0.06)",
            showlegend=True,
        ))

    # Candlestick
    _add(go.Candlestick(
        x=sl.index,
        open=sl["Open"], high=sl["High"],
        low=sl["Low"],   close=sl["Close"],
        name="Price",
        increasing_line_color=COLOR_UP,
        decreasing_line_color=COLOR_DOWN,
        increasing_fillcolor=COLOR_UP,
        decreasing_fillcolor=COLOR_DOWN,
    ))

    # Moving averages
    if show_sma20 and "SMA20" in sl.columns:
        _add(go.Scatter(x=sl.index, y=sl["SMA20"], name="SMA 20",
                        line=dict(color=COLOR_SMA20, width=1.4)))
    if show_sma50 and "SMA50" in sl.columns:
        _add(go.Scatter(x=sl.index, y=sl["SMA50"], name="SMA 50",
                        line=dict(color=COLOR_SMA50, width=1.4)))
    if show_sma200 and "SMA200" in sl.columns:
        _add(go.Scatter(x=sl.index, y=sl["SMA200"], name="SMA 200",
                        line=dict(color=COLOR_SMA200, width=1.4)))
    if show_ema20 and "EMA20" in sl.columns:
        _add(go.Scatter(x=sl.index, y=sl["EMA20"], name="EMA 20",
                        line=dict(color=COLOR_EMA20, width=1.4, dash="dot")))

    # Volume bars
    if show_volume and "Volume" in sl.columns and price_row is not None:
        colors = [
            COLOR_UP if c >= o else COLOR_DOWN
            for c, o in zip(sl["Close"], sl["Open"])
        ]
        fig.add_trace(go.Bar(
            x=sl.index, y=sl["Volume"],
            name="Volume", marker_color=colors,
            opacity=0.6, showlegend=False,
        ), row=vol_row, col=1)

    # Layout
    common_layout = dict(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT_PRICE,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=11),
        ),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    if price_row is not None:
        fig.update_layout(
            **common_layout,
            yaxis=dict(title=f"Price ({currency})", side="right"),
            yaxis2=dict(title="Volume", side="left", showgrid=False),
            xaxis2=dict(showgrid=False),
        )
    else:
        fig.update_layout(
            **common_layout,
            yaxis_title=f"Price ({currency})",
        )
    return fig


def _build_forecast_chart(
    df: pd.DataFrame,
    predictions: list[float],
    ticker: str,
    currency: str,
    days_to_predict: int,
    hist_lookback: int = 60,
) -> go.Figure:
    """
    Forecast chart: historical closing prices (solid) + forecast line (dashed),
    with a shaded region marking the prediction window.
    """
    hist_sl = df["Close"].iloc[-hist_lookback:]
    last_date = df.index[-1]
    pred_dates = pd.bdate_range(start=last_date, periods=days_to_predict + 1)[1:]

    fig = go.Figure()

    # Historical line
    fig.add_trace(go.Scatter(
        x=hist_sl.index, y=hist_sl.values,
        name="Historical Close",
        line=dict(color=COLOR_PRIMARY, width=2),
        mode="lines",
    ))

    # Bridge point (connects history to forecast)
    bridge_x = [hist_sl.index[-1]] + list(pred_dates)
    bridge_y = [float(hist_sl.iloc[-1])] + [float(p) for p in predictions]

    # Shaded forecast region
    fig.add_vrect(
        x0=str(last_date.date()), x1=str(pred_dates[-1].date()),
        fillcolor="rgba(0,196,255,0.04)",
        line_width=0,
        annotation_text="Forecast Window",
        annotation_position="top left",
        annotation_font=dict(color=COLOR_NEUTRAL, size=11),
    )

    # Forecast line (colour-coded per segment)
    for i in range(1, len(bridge_y)):
        seg_color = COLOR_UP if bridge_y[i] >= bridge_y[i - 1] else COLOR_DOWN
        fig.add_trace(go.Scatter(
            x=[bridge_x[i - 1], bridge_x[i]],
            y=[bridge_y[i - 1], bridge_y[i]],
            mode="lines+markers",
            line=dict(color=seg_color, width=2.5, dash="dash"),
            marker=dict(size=6, color=seg_color),
            showlegend=(i == 1),
            name="Forecast",
        ))

    # Vertical separator line
    fig.add_vline(
        x=last_date, line_dash="dot",
        line_color=COLOR_NEUTRAL, line_width=1,
        annotation_text="Today",
        annotation_font=dict(color=COLOR_NEUTRAL, size=10),
    )

    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT_FORECAST,
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        yaxis_title=f"Price ({currency})",
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01,
            xanchor="right", x=1, font=dict(size=11),
        ),
    )
    return fig


def _build_rsi_chart(df: pd.DataFrame) -> go.Figure:
    """RSI chart with OB/OS bands."""
    sl = df.iloc[-HISTORICAL_DISPLAY_DAYS:]
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(248,81,73,0.08)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(38,166,65,0.08)",  line_width=0)
    fig.add_hline(y=70, line_dash="dot", line_color=COLOR_DOWN,    line_width=1,
                  annotation_text="Overbought (70)",
                  annotation_position="bottom right",
                  annotation_font=dict(color=COLOR_DOWN, size=10))
    fig.add_hline(y=30, line_dash="dot", line_color=COLOR_UP,      line_width=1,
                  annotation_text="Oversold (30)",
                  annotation_position="top right",
                  annotation_font=dict(color=COLOR_UP, size=10))
    fig.add_trace(go.Scatter(
        x=sl.index, y=sl["RSI"],
        name="RSI (14)",
        line=dict(color=COLOR_RSI, width=1.8),
        fill="tozeroy", fillcolor="rgba(163,113,247,0.07)",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_RSI,
        yaxis=dict(title="RSI", range=[0, 100]),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def _build_macd_chart(df: pd.DataFrame) -> go.Figure:
    """MACD line, signal line, and histogram."""
    d = compute_macd(df).iloc[-HISTORICAL_DISPLAY_DAYS:]
    hist_colors = [
        COLOR_HIST_POS if v >= 0 else COLOR_HIST_NEG
        for v in d["MACD_Hist"]
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=d.index, y=d["MACD_Hist"],
        name="Histogram", marker_color=hist_colors, opacity=0.7,
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["MACD"],
        name="MACD", line=dict(color=COLOR_MACD, width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["MACD_Signal"],
        name="Signal", line=dict(color=COLOR_SIGNAL, width=1.4, dash="dot"),
    ))
    fig.add_hline(y=0, line_color=COLOR_NEUTRAL, line_width=0.8, line_dash="solid")
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_MACD,
        yaxis_title="MACD",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def _build_ma_chart(df: pd.DataFrame, currency: str) -> go.Figure:
    """Overlay of Close, SMA20, SMA50, SMA200, EMA20."""
    sl = df.iloc[-HISTORICAL_DISPLAY_DAYS:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sl.index, y=sl["Close"], name="Close",
        line=dict(color=COLOR_PRIMARY, width=1.6),
    ))
    for col, color, label in [
        ("SMA20",  COLOR_SMA20,  "SMA 20"),
        ("SMA50",  COLOR_SMA50,  "SMA 50"),
        ("SMA200", COLOR_SMA200, "SMA 200"),
        ("EMA20",  COLOR_EMA20,  "EMA 20"),
    ]:
        if col in sl.columns:
            fig.add_trace(go.Scatter(
                x=sl.index, y=sl[col], name=label,
                line=dict(color=color, width=1.4,
                          dash="dot" if col == "EMA20" else "solid"),
            ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_INDICATORS,
        yaxis_title=f"Price ({currency})",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def _build_bb_chart(df: pd.DataFrame, currency: str) -> go.Figure:
    """Bollinger Bands with close price and bandwidth indicator."""
    sl = df.iloc[-HISTORICAL_DISPLAY_DAYS:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=sl.index, y=sl["BB_upper"], name="Upper Band",
        line=dict(color="rgba(100,180,255,0.5)", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=sl.index, y=sl["BB_lower"], name="Lower Band",
        line=dict(color="rgba(100,180,255,0.5)", width=1),
        fill="tonexty", fillcolor="rgba(100,180,255,0.06)",
    ))
    fig.add_trace(go.Scatter(
        x=sl.index, y=sl["BB_mid"], name="Middle (SMA20)",
        line=dict(color=COLOR_SMA20, width=1.2, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=sl.index, y=sl["Close"], name="Close",
        line=dict(color=COLOR_PRIMARY, width=1.8),
    ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_INDICATORS,
        yaxis_title=f"Price ({currency})",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def _build_stoch_chart(df: pd.DataFrame) -> go.Figure:
    """Stochastic Oscillator %K / %D."""
    d = compute_stochastic(df).iloc[-HISTORICAL_DISPLAY_DAYS:]
    fig = go.Figure()
    fig.add_hrect(y0=80, y1=100, fillcolor="rgba(248,81,73,0.07)", line_width=0)
    fig.add_hrect(y0=0,  y1=20,  fillcolor="rgba(38,166,65,0.07)",  line_width=0)
    fig.add_hline(y=80, line_dash="dot", line_color=COLOR_DOWN, line_width=0.8)
    fig.add_hline(y=20, line_dash="dot", line_color=COLOR_UP,   line_width=0.8)
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Stoch_K"], name="%K",
        line=dict(color=COLOR_PRIMARY, width=1.6),
    ))
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Stoch_D"], name="%D",
        line=dict(color=COLOR_SIGNAL, width=1.4, dash="dot"),
    ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_RSI,
        yaxis=dict(title="Stochastic", range=[0, 100]),
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(orientation="h", yanchor="bottom", y=1.01,
                    xanchor="right", x=1, font=dict(size=11)),
    )
    return fig


def _build_volatility_chart(df: pd.DataFrame) -> go.Figure:
    """Rolling 20-day annualised volatility."""
    d = compute_volatility(df).iloc[-HISTORICAL_DISPLAY_DAYS:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["Volatility"],
        name="Volatility (20d Ann.)",
        line=dict(color=COLOR_WARNING, width=1.8),
        fill="tozeroy", fillcolor="rgba(227,179,65,0.08)",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_VOL,
        yaxis_title="Annualised Volatility (%)",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def _build_atr_chart(df: pd.DataFrame, currency: str) -> go.Figure:
    """Average True Range (14-period)."""
    d = compute_atr(df).iloc[-HISTORICAL_DISPLAY_DAYS:]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=d.index, y=d["ATR"],
        name="ATR (14)",
        line=dict(color=COLOR_MACD, width=1.8),
        fill="tozeroy", fillcolor="rgba(0,196,255,0.07)",
    ))
    fig.update_layout(
        template=CHART_TEMPLATE, height=CHART_HEIGHT_VOL,
        yaxis_title=f"ATR ({currency})",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def _evaluation_results_frame(
    results: list[ModelEvaluationResult],
    *,
    include_status: bool = False,
) -> pd.DataFrame:
    """Build a display table from completed evaluation results only."""
    rows = []
    for result in results:
        row = {
            "Model": result.model_name,
            "MAE": result.MAE,
            "MSE": result.MSE,
            "RMSE": result.RMSE,
            "MAPE": result.MAPE,
            "R²": result.R2,
            "Samples": len(result.y_true),
            "Folds": result.number_of_folds,
        }
        if include_status:
            row["Status"] = result.status
        rows.append(row)
    return pd.DataFrame(rows)


def _evaluation_timestamps(result: ModelEvaluationResult) -> np.ndarray:
    """Return result timestamps or a stable integer axis when unavailable."""
    if result.timestamps is not None and len(result.timestamps) == len(result.y_true):
        return np.asarray(result.timestamps)
    return np.arange(len(result.y_true))


def _build_actual_predicted_chart(result: ModelEvaluationResult) -> go.Figure:
    """Build an actual-versus-predicted Plotly chart for one model."""
    x = _evaluation_timestamps(result)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=result.y_true, name="Actual",
        mode="lines+markers", line=dict(color=COLOR_PRIMARY, width=2),
    ))
    fig.add_trace(go.Scatter(
        x=x, y=result.y_pred, name="Predicted",
        mode="lines+markers", line=dict(color=COLOR_UP, width=2, dash="dash"),
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT_FORECAST,
        hovermode="x unified",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _build_residual_chart(result: ModelEvaluationResult) -> go.Figure:
    """Build a residuals-over-time chart for one model."""
    x = _evaluation_timestamps(result)
    residuals = result.y_true - result.y_pred
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=residuals, name="Residual",
        mode="lines+markers", line=dict(color=COLOR_WARNING, width=1.8),
    ))
    fig.add_hline(y=0, line_color=COLOR_NEUTRAL, line_dash="dot")
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT_FORECAST,
        hovermode="x unified",
        yaxis_title="Actual − Predicted",
        xaxis_rangeslider_visible=False,
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _build_error_distribution_chart(result: ModelEvaluationResult) -> go.Figure:
    """Build a histogram of prediction residuals for one model."""
    residuals = result.y_true - result.y_pred
    fig = go.Figure(go.Histogram(
        x=residuals,
        name="Residuals",
        marker_color=COLOR_PRIMARY,
        opacity=0.8,
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT_FORECAST,
        xaxis_title="Actual − Predicted",
        yaxis_title="Count",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _build_model_comparison_chart(
    table: pd.DataFrame,
    metric: str,
) -> go.Figure:
    """Build a bar chart for a selected evaluation metric."""
    fig = go.Figure(go.Bar(
        x=table["Model"],
        y=table[metric],
        marker_color=COLOR_PRIMARY,
        name=metric,
    ))
    fig.update_layout(
        template=CHART_TEMPLATE,
        height=CHART_HEIGHT_FORECAST,
        yaxis_title=metric,
        xaxis_title="Model",
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def _build_fold_performance_table(report: ModelComparisonReport) -> pd.DataFrame:
    """Flatten completed fold metrics for validation diagnostics."""
    rows = []
    for result in report.validation_results:
        for fold in result.fold_results:
            rows.append({
                "Model": result.model_name,
                "Fold": fold.fold_number,
                "Validation start": fold.validation_start,
                "Validation end": fold.validation_end,
                "MAE": fold.metrics.MAE,
                "MSE": fold.metrics.MSE,
                "RMSE": fold.metrics.RMSE,
                "MAPE": fold.metrics.MAPE,
                "R²": fold.metrics.R2,
            })
    return pd.DataFrame(rows)


def _render_model_evaluation(report: ModelComparisonReport, metric: str) -> None:
    """Render validation/test reports and diagnostics from a completed report."""
    validation = report.comparison_table(sort_by=metric, ascending=(metric != "R2"))
    validation_ok = validation[validation["Status"] == "OK"].copy() \
        if not validation.empty else validation
    test_results = [item for item in report.test_results if item.status == "OK"]
    test = _evaluation_results_frame(test_results)

    st.markdown(
        '<p class="si-section">🧪 Model Evaluation Results</p>',
        unsafe_allow_html=True,
    )

    st.markdown("### Validation Performance")
    if validation_ok.empty:
        st.warning("No models completed validation successfully.")
        failed = (
            validation[validation["Status"] != "OK"]
            if not validation.empty else validation
        )
        if not failed.empty:
            st.dataframe(
                failed[["Model", "Status"]],
                use_container_width=True,
                hide_index=True,
            )
        st.markdown("### Test Performance")
        if test.empty:
            st.info("No held-out test evaluation completed successfully.")
        else:
            st.dataframe(
                test[["Model", "MAE", "MSE", "RMSE", "MAPE", "R²"]],
                use_container_width=True,
                hide_index=True,
            )
        return

    validation_display = validation_ok.rename(columns={"R2": "R²"})
    st.dataframe(
        validation_display[["Model", "MAE", "MSE", "RMSE", "MAPE", "R²"]],
        use_container_width=True,
        hide_index=True,
    )

    successful = [
        result for result in report.validation_results if result.status == "OK"
    ]
    if successful:
        selected = st.selectbox(
            "Diagnostic model",
            options=[result.model_name for result in successful],
            key="evaluation_diagnostic_model",
        )
        result = next(item for item in successful if item.model_name == selected)

        chart_col1, chart_col2 = st.columns(2, gap="medium")
        with chart_col1:
            st.plotly_chart(
                _build_actual_predicted_chart(result),
                use_container_width=True,
            )
        with chart_col2:
            st.plotly_chart(
                _build_residual_chart(result),
                use_container_width=True,
            )

        chart_col3, chart_col4 = st.columns(2, gap="medium")
        with chart_col3:
            st.plotly_chart(
                _build_error_distribution_chart(result),
                use_container_width=True,
            )
        with chart_col4:
            st.plotly_chart(
                _build_model_comparison_chart(validation_ok, metric),
                use_container_width=True,
            )

        fold_table = _build_fold_performance_table(report)
        if not fold_table.empty:
            st.markdown("### Validation Fold Performance")
            st.dataframe(fold_table, use_container_width=True, hide_index=True)

        timing = pd.DataFrame([
            {
                "Evaluation": "Validation",
                "Model": item.model_name,
                "Training time (s)": item.training_time,
                "Prediction time (s)": item.prediction_time,
                "Status": item.status,
            }
            for item in report.validation_results
        ] + [
            {
                "Evaluation": "Test",
                "Model": item.model_name,
                "Training time (s)": item.training_time,
                "Prediction time (s)": item.prediction_time,
                "Status": item.status,
            }
            for item in report.test_results
        ])
        st.markdown("### Training and Prediction Time")
        st.dataframe(timing, use_container_width=True, hide_index=True)

    st.markdown("### Test Performance")
    if test.empty:
        if report.test_results:
            st.warning("Held-out test evaluation ran, but no model completed successfully.")
            failed_test = _evaluation_results_frame(
                [item for item in report.test_results if item.status != "OK"],
                include_status=True,
            )
            if not failed_test.empty:
                st.dataframe(
                    failed_test[["Model", "Status"]],
                    use_container_width=True,
                    hide_index=True,
                )
        else:
            st.info("No held-out test evaluation was executed.")
    else:
        st.dataframe(
            test[["Model", "MAE", "MSE", "RMSE", "MAPE", "R²"]],
            use_container_width=True,
            hide_index=True,
        )


def _evaluation_signature(
    ticker: str,
    period: str,
    horizon: int,
    method: str,
    models: list[str],
    metric: str,
    initial_train_size: int,
    validation_window_size: int,
    step_size: int,
    include_test: bool,
) -> tuple:
    """Identify the controls used to produce the stored evaluation report."""
    return (
        ticker, period, horizon, method, tuple(models), metric,
        initial_train_size, validation_window_size, step_size, include_test,
    )



# ===========================================================================
# ── METRIC HELPERS ────────────────────────────────────────────────────────────
# ===========================================================================

def _compute_model_metrics(
    df: pd.DataFrame, scaler, X: np.ndarray, y: np.ndarray
) -> dict:
    """
    Compute in-sample model performance metrics on the available sequences.

    Returns MAE, RMSE, MAPE, R² computed against the actual scaled-then-
    inverted close prices using the last 20% of available sequences as a
    pseudo hold-out set.
    """
    keras_model = _cached_keras_model()
    if keras_model is None:
        return {"MAE": None, "RMSE": None, "MAPE": None, "R2": None, "n_test": 0}
    # Use last 20 % of sequences as pseudo test set (max 100 samples)
    n_test = min(max(int(len(X) * 0.20), 5), 100)
    X_test = X[-n_test:]
    y_true_scaled = y[-n_test:].reshape(-1, 1)

    y_pred_scaled = keras_model.predict(X_test, verbose=0)

    y_true = scaler.inverse_transform(y_true_scaled).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled).flatten()

    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))

    nonzero = y_true != 0
    mape = float(np.mean(np.abs((y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero])) * 100)

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape, "R2": r2,
            "n_test": n_test, "y_true": y_true, "y_pred": y_pred}


def _fmt_currency(value: float | None, currency: str, precision: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{currency}{value:,.{precision}f}"


def _fmt_large(value: float | None) -> str:
    """Format large numbers with B / M suffixes."""
    if value is None:
        return "N/A"
    if value >= 1e12:
        return f"{value / 1e12:.2f} T"
    if value >= 1e9:
        return f"{value / 1e9:.2f} B"
    if value >= 1e6:
        return f"{value / 1e6:.2f} M"
    return f"{value:,.0f}"


def _fmt_pct(value: float | None, scale: float = 1.0) -> str:
    if value is None:
        return "N/A"
    return f"{value * scale:.2f}%"


def _delta_class(delta: float) -> str:
    if delta > 0:
        return "si-up"
    if delta < 0:
        return "si-down"
    return "si-neutral"


def _signal_badge(rsi: float | None) -> str:
    if rsi is None:
        return ""
    if rsi >= 70:
        return "🔴 **Overbought** (RSI ≥ 70)"
    if rsi <= 30:
        return "🟢 **Oversold** (RSI ≤ 30)"
    return "🟡 **Neutral**"


# ===========================================================================
# ── DOWNLOAD HELPERS ──────────────────────────────────────────────────────────
# ===========================================================================

def _make_hist_csv(df: pd.DataFrame) -> bytes:
    out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    out.index.name = "Date"
    return out.to_csv().encode("utf-8")


def _make_forecast_csv(
    ticker: str, current_price: float, predictions: list[float],
    last_date, currency: str,
) -> bytes:
    pred_dates = pd.bdate_range(start=last_date, periods=len(predictions) + 1)[1:]
    rows = []
    ref = current_price
    for i, (date, price) in enumerate(zip(pred_dates, predictions), start=1):
        delta = price - ref
        delta_pct = (delta / ref * 100) if ref != 0 else 0.0
        rows.append({
            "Day": i,
            "Date": date.date(),
            "Forecast_Price": round(price, 4),
            "Change": round(delta, 4),
            "Change_Pct": round(delta_pct, 4),
            "Currency": currency,
        })
        ref = price
    return pd.DataFrame(rows).to_csv(index=False).encode("utf-8")


def _make_report_text(
    ticker: str, info: dict, current_price: float, predictions: list[float],
    metrics: dict | None, currency: str, hist_period: str,
    selected_model: str, days_to_predict: int,
) -> bytes:
    lines = [
        "=" * 60,
        f"  {APP_NAME} — Analytics Report",
        f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 60,
        "",
        "STOCK SUMMARY",
        f"  Ticker         : {ticker}",
        f"  Company        : {info.get('company_name', 'N/A')}",
        f"  Sector         : {info.get('sector', 'N/A')}",
        f"  Industry       : {info.get('industry', 'N/A')}",
        f"  Exchange       : {info.get('exchange', 'N/A')}",
        f"  Market Cap     : {_fmt_large(info.get('market_cap'))}",
        f"  52-Week High   : {_fmt_currency(info.get('week_52_high'), currency)}",
        f"  52-Week Low    : {_fmt_currency(info.get('week_52_low'), currency)}",
        f"  Beta           : {info.get('beta', 'N/A')}",
        f"  P/E Ratio      : {info.get('pe_ratio', 'N/A')}",
        "",
        "FORECAST",
        f"  Model          : {selected_model}",
        f"  Horizon        : {days_to_predict} day(s)",
        f"  Current Price  : {_fmt_currency(current_price, currency)}",
    ]
    if predictions:
        final = predictions[-1]
        total_chg = final - current_price
        total_pct = (total_chg / current_price * 100) if current_price != 0 else 0
        lines += [
            f"  Forecast (D{days_to_predict}) : {_fmt_currency(final, currency)}",
            f"  Expected Move  : {total_chg:+.2f} ({total_pct:+.2f}%)",
        ]
    lines += [""]
    if metrics:
        lines += [
            "MODEL PERFORMANCE (pseudo hold-out)",
            f"  MAE            : {metrics['MAE']:.4f}",
            f"  RMSE           : {metrics['RMSE']:.4f}",
            f"  MAPE           : {metrics['MAPE']:.2f}%",
            f"  R²             : {metrics['R2']:.4f}",
            f"  Test samples   : {metrics['n_test']}",
            "",
        ]
    lines += [
        "DISCLAIMER",
        DISCLAIMER.replace("**", "").replace("⚠️ ", ""),
        "",
        f"Data source: {DATA_SOURCE}",
        f"Original project: {ORIGINAL_PROJECT_NAME} by {ORIGINAL_AUTHOR}",
        f"  {ORIGINAL_REPO_URL}",
    ]
    return "\n".join(lines).encode("utf-8")


# ===========================================================================
# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────
# ===========================================================================

def run_analysis(
    ticker: str,
    market: str,
    period: str,
    days_to_predict: int,
    selected_model: str,
):
    """
    Full analysis pipeline: fetch → enrich → model registry → forecast.

    Uses the new modular model registry.  The selected_model name is mapped
    to the registry key and dispatched accordingly.  When "Auto" is selected,
    all available models are evaluated and the best-RMSE model is used.

    Returns
    -------
    raw_df          : pd.DataFrame — cleaned OHLCV (for historical table)
    df              : pd.DataFrame — OHLCV + all technical indicators
    predictions     : list[float]  — forecast prices (empty list on failure)
    model_used      : str          — display name of the model actually used
    comparison_df   : pd.DataFrame | None — model comparison table (Auto only)
    all_predictions : dict[str, list[float]] — predictions from every model
    X, y            : np.ndarray   — LSTM sequences (may be None for non-LSTM)
    scaler          : MinMaxScaler | None
    """
    with st.spinner(f"Fetching {period} of data for **{ticker}**…"):
        raw_df = cached_fetch(ticker, period)

    if raw_df is None or raw_df.empty:
        st.error(
            f"❌ Could not retrieve data for **{ticker}**.\n\n"
            "**Possible causes:**\n"
            "- Yahoo Finance rate-limiting — try again in ~30 s\n"
            "- Ticker may be delisted or incorrect"
        )
        return None, None, [], selected_model, None, {}, None, None

    df = add_technical_indicators(raw_df.copy())
    close_series = df["Close"].rename("Close")

    # LSTM sequences — always built when TF is available (for metrics panel)
    X, y_seq, scaler = None, None, None
    if _TF_AVAILABLE:
        X, y_seq, scaler = preprocess_for_lstm(df, sequence_length=SEQUENCE_LENGTH)
        if len(X) == 0:
            X, y_seq, scaler = None, None, None

    predictions:    list[float]        = []
    comparison_df:  pd.DataFrame | None = None
    all_predictions: dict[str, list[float]] = {}
    model_used = selected_model

    # ── Auto: evaluate all, pick best ───────────────────────────────────────
    if selected_model == "Auto":
        with st.spinner("Running all models — evaluating and selecting the best…"):
            try:
                best_name, predictions, comparison_df = auto_select(
                    close_series, steps=days_to_predict, val_size=MODEL_VAL_SIZE
                )
                model_used = best_name
                # Collect predictions from the comparison run for the overlay
                _, all_predictions = run_all_models(
                    close_series, steps=days_to_predict, val_size=MODEL_VAL_SIZE
                )
                all_predictions[best_name] = predictions
            except Exception as exc:
                st.error(f"❌ Auto model selection failed: {exc}")

    # ── Specific model ───────────────────────────────────────────────────────
    else:
        registry_key = MODEL_DISPLAY_TO_KEY.get(selected_model, selected_model)
        with st.spinner(f"Running {selected_model}…"):
            try:
                model_obj = get_model(registry_key)
                model_obj.fit(close_series, val_size=MODEL_VAL_SIZE)
                predictions = model_obj.predict(steps=days_to_predict)
                all_predictions[selected_model] = predictions
            except ImportError as exc:
                st.error(
                    f"❌ **{selected_model}** is not available: {exc}\n\n"
                    "Try selecting a different model."
                )
            except Exception as exc:
                st.error(f"❌ {selected_model} failed: {exc}")

    return raw_df, df, predictions, model_used, comparison_df, all_predictions, X, y_seq, scaler


# ===========================================================================
# ── HERO HEADER ─────────────────────────────────────────────────────────────
# ===========================================================================
st.markdown(
    f"""
    <div class="si-hero">
      <div class="si-hero-icon">{APP_ICON}</div>
      <div>
        <p class="si-hero-title">{APP_NAME}</p>
        <p class="si-hero-tagline">{APP_TAGLINE}</p>
      </div>
      <div class="si-hero-badges">
        v{APP_VERSION} &nbsp;|&nbsp; {DATA_SOURCE}<br>
        Built by <strong style="color:#E6EDF3;">{AUTHOR_NAME}</strong>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ===========================================================================
# ── DEDICATED MODEL EVALUATION ──────────────────────────────────────────────
# ===========================================================================
current_evaluation_signature = _evaluation_signature(
    ticker=ticker,
    period=hist_period,
    horizon=days_to_predict,
    method=evaluation_method,
    models=evaluation_models,
    metric=evaluation_metric,
    initial_train_size=int(evaluation_initial_train),
    validation_window_size=int(evaluation_window),
    step_size=int(evaluation_step),
    include_test=evaluation_test,
)

if run_evaluation:
    st.session_state.pop("model_evaluation_report", None)
    st.session_state.pop("model_evaluation_signature", None)
    st.session_state.pop("model_evaluation_dataset", None)

    if not evaluation_models:
        st.error("Select at least one available model to evaluate.")
    else:
        with st.spinner(f"Fetching {hist_period} of data for model evaluation…"):
            evaluation_raw = cached_fetch(ticker, hist_period)

        if evaluation_raw is None or evaluation_raw.empty:
            st.error(f"Could not retrieve historical data for {ticker}.")
        elif "Close" not in evaluation_raw.columns:
            st.error("Fetched data does not contain a Close price column.")
        else:
            evaluation_series = (
                evaluation_raw["Close"]
                .dropna()
                .sort_index()
                .astype(float)
                .rename("Close")
            )
            if not evaluation_series.index.is_unique:
                st.error(
                    "Historical data contains duplicate timestamps; "
                    "evaluation was not run."
                )
            elif len(evaluation_series) < 2:
                st.error("Not enough valid historical observations to evaluate.")
            else:
                is_walk_forward = evaluation_method == "Walk-forward"
                with st.spinner(
                    "Running selected models on chronological validation folds…"
                ):
                    evaluation_report = compare_models(
                        evaluation_series,
                        model_names=evaluation_models,
                        val_size=int(evaluation_window),
                        test_size=days_to_predict if evaluation_test else 0,
                        include_test=bool(evaluation_test),
                        skip_unavailable=False,
                        walk_forward=is_walk_forward,
                        expanding_window=not is_walk_forward,
                        initial_train_size=int(evaluation_initial_train),
                        validation_window_size=int(evaluation_window),
                        step_size=int(evaluation_step),
                    )
                st.session_state["model_evaluation_report"] = evaluation_report
                st.session_state["model_evaluation_signature"] = (
                    current_evaluation_signature
                )
                st.session_state["model_evaluation_dataset"] = {
                    "ticker": ticker,
                    "period": hist_period,
                    "observations": len(evaluation_series),
                    "start": str(evaluation_series.index[0]),
                    "end": str(evaluation_series.index[-1]),
                    "validation_method": evaluation_method,
                    "initial_train_size": int(evaluation_initial_train),
                    "validation_window_size": int(evaluation_window),
                    "step_size": int(evaluation_step),
                    "test_horizon": days_to_predict if evaluation_test else 0,
                }

stored_evaluation_report = st.session_state.get("model_evaluation_report")
stored_evaluation_signature = st.session_state.get("model_evaluation_signature")
evaluation_is_current = (
    isinstance(stored_evaluation_report, ModelComparisonReport)
    and stored_evaluation_signature == current_evaluation_signature
)

if evaluation_is_current:
    st.markdown(
        '<p class="si-section">🧪 Model Comparison and Evaluation</p>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Results for {ticker} · {hist_period} · {evaluation_method} validation · "
        f"{days_to_predict}-day forecast horizon"
    )
    dataset = st.session_state["model_evaluation_dataset"]
    st.markdown("### Dataset Information")
    st.dataframe(
        pd.DataFrame([dataset]),
        use_container_width=True,
        hide_index=True,
    )
    _render_model_evaluation(stored_evaluation_report, evaluation_metric)

# ===========================================================================
# ── LANDING STATE (before first run) ─────────────────────────────────────────
# ===========================================================================
if not run_clicked:
    if not evaluation_is_current:
        st.markdown(
            f"""
            <div style="text-align:center;padding:3rem 1rem 2rem 1rem;
                        color:{COLOR_NEUTRAL};">
                <div style="font-size:3rem;margin-bottom:0.5rem;">📊</div>
                <div style="font-size:1.1rem;font-weight:600;
                            color:#E6EDF3;margin-bottom:0.4rem;">
                    Select a stock and click <span style="color:{COLOR_PRIMARY};">
                    ⚡ Run Analysis</span> or run a dedicated model evaluation.
                </div>
                <div style="font-size:0.88rem;">
                    Configure ticker, historical period, forecast horizon, and
                    evaluation settings in the sidebar.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        c1, c2, c3, c4 = st.columns(4, gap="small")
        for col, icon, title, desc in [
            (c1, "📡", "Live Data",
             f"Up to 5 years of OHLCV from {DATA_SOURCE}"),
            (c2, "📊", "Technical Analysis",
             "RSI, MACD, Stochastic, Bollinger, ATR, Volatility"),
            (c3, "🧠", "Multi-Model Forecast",
             f"LSTM · Random Forest · XGBoost · ARIMA · Baseline — Auto selects the best"),
            (c4, "📥", "Download",
             "Export historical data, forecasts & full report as CSV / TXT"),
        ]:
            col.markdown(
                f"""
                <div class="si-card" style="text-align:center;padding:1.2rem;">
                    <div style="font-size:1.8rem;margin-bottom:0.4rem;">{icon}</div>
                    <div style="font-weight:700;color:#E6EDF3;
                                font-size:0.9rem;">{title}</div>
                    <div class="si-card-sub" style="margin-top:0.3rem;
                                font-size:0.78rem;">{desc}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.stop()

# ===========================================================================
# ── RUN ANALYSIS ─────────────────────────────────────────────────────────────
# ===========================================================================
currency = CURRENCY_SYMBOL.get(market, "$")

(raw_df, df, predictions, model_used,
 comparison_df, all_predictions, X, y_seq, scaler) = run_analysis(
    ticker, market, hist_period, days_to_predict, selected_model,
)

# Abort early if data fetch failed
if raw_df is None:
    st.stop()

# Fetch company metadata (non-blocking — errors return safe defaults)
with st.spinner("Fetching company information…"):
    info = cached_info(ticker)

current_price   = float(df["Close"].iloc[-1])
prev_price      = float(df["Close"].iloc[-2]) if len(df) > 1 else current_price
daily_chg       = current_price - prev_price
daily_chg_pct   = (daily_chg / prev_price * 100) if prev_price != 0 else 0.0
last_volume     = int(df["Volume"].iloc[-1]) if "Volume" in df.columns else None
last_rsi        = float(df["RSI"].iloc[-1]) if "RSI" in df.columns else None

# 52-week high/low — prefer live info dict; fall back to DataFrame window
week52_high = info.get("week_52_high") or float(df["High"].tail(252).max())
week52_low  = info.get("week_52_low")  or float(df["Low"].tail(252).min())

# ===========================================================================
# ── SECTION 2: STOCK SUMMARY CARD ────────────────────────────────────────────
# ===========================================================================
st.markdown('<p class="si-section">📌 Stock Summary</p>', unsafe_allow_html=True)

company_name = info.get("company_name", ticker)
delta_cls    = _delta_class(daily_chg)
chg_arrow    = "▲" if daily_chg > 0 else ("▼" if daily_chg < 0 else "—")

# Row 1 — name + price block
r1c1, r1c2, r1c3, r1c4, r1c5 = st.columns([2.2, 1.4, 1.4, 1.4, 1.4], gap="small")

with r1c1:
    st.markdown(
        f"""
        <div class="si-card">
          <div class="si-card-label">Company</div>
          <div class="si-card-value" style="font-size:1.1rem;">
            {company_name}
          </div>
          <div class="si-card-sub">
            {ticker} &nbsp;·&nbsp;
            {info.get('exchange','N/A')} &nbsp;·&nbsp;
            {info.get('sector','N/A')}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r1c2:
    st.markdown(
        f"""
        <div class="si-card">
          <div class="si-card-label">Latest Close</div>
          <div class="si-card-value">{_fmt_currency(current_price, currency)}</div>
          <div class="si-card-sub">{info.get('currency','')}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r1c3:
    st.markdown(
        f"""
        <div class="si-card">
          <div class="si-card-label">Daily Change</div>
          <div class="si-card-value">
            <span class="{delta_cls}">
              {chg_arrow} {_fmt_currency(abs(daily_chg), currency)}
            </span>
          </div>
          <div class="si-card-sub">
            <span class="{delta_cls}">{daily_chg_pct:+.2f}%</span>
            &nbsp;vs prev close
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r1c4:
    st.markdown(
        f"""
        <div class="si-card">
          <div class="si-card-label">52-Week Range</div>
          <div class="si-card-value"
               style="font-size:0.95rem;letter-spacing:-0.3px;">
            {_fmt_currency(week52_low, currency)}
            <span style="color:{COLOR_NEUTRAL};font-weight:400;
                         font-size:0.8rem;"> – </span>
            {_fmt_currency(week52_high, currency)}
          </div>
          <div class="si-card-sub">Low · High</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with r1c5:
    vol_str = f"{last_volume:,}" if last_volume else "N/A"
    avg_vol = info.get("avg_volume")
    avg_vol_str = _fmt_large(avg_vol) if avg_vol else "N/A"
    st.markdown(
        f"""
        <div class="si-card">
          <div class="si-card-label">Volume</div>
          <div class="si-card-value" style="font-size:1rem;">{vol_str}</div>
          <div class="si-card-sub">Avg 10d: {avg_vol_str}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Row 2 — supplementary stats
st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
r2c1, r2c2, r2c3, r2c4, r2c5 = st.columns(5, gap="small")

for col, label, value in [
    (r2c1, "Market Cap",    _fmt_large(info.get("market_cap"))),
    (r2c2, "P/E Ratio",     f"{info['pe_ratio']:.2f}" if info.get("pe_ratio") else "N/A"),
    (r2c3, "EPS (TTM)",     f"{currency}{info['eps']:.2f}" if info.get("eps") else "N/A"),
    (r2c4, "Dividend Yield",_fmt_pct(info.get("dividend_yield"), scale=100)),
    (r2c5, "Beta",          f"{info['beta']:.2f}" if info.get("beta") else "N/A"),
]:
    col.markdown(
        f"""
        <div class="si-card" style="padding:0.65rem 1rem;">
          <div class="si-card-label">{label}</div>
          <div style="font-size:1rem;font-weight:700;
                      color:#E6EDF3;margin-top:0.1rem;">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

# ===========================================================================
# ── SECTION 3: PRICE CHART ────────────────────────────────────────────────────
# ===========================================================================
st.markdown('<p class="si-section">📊 Price Chart</p>', unsafe_allow_html=True)

display_days = min(HISTORICAL_DISPLAY_DAYS, len(df))
price_fig = _build_price_chart(
    df, ticker, currency,
    show_sma20, show_sma50, show_sma200, show_ema20, show_bb, show_volume,
    display_days=display_days,
)
st.plotly_chart(price_fig, use_container_width=True)

# ===========================================================================
# ── SECTION 4: FORECAST SUMMARY ──────────────────────────────────────────────
# ===========================================================================
if predictions:
    st.markdown(
        '<p class="si-section">🔮 Forecast Summary</p>',
        unsafe_allow_html=True,
    )

    final_price   = float(predictions[-1])
    total_chg_f   = final_price - current_price
    total_pct_f   = (total_chg_f / current_price * 100) if current_price != 0 else 0
    fc_delta_cls  = _delta_class(total_chg_f)
    fc_arrow      = "▲" if total_chg_f > 0 else ("▼" if total_chg_f < 0 else "—")

    fc1, fc2, fc3, fc4 = st.columns(4, gap="small")

    with fc1:
        st.markdown(
            f"""
            <div class="si-card">
              <div class="si-card-label">Current Price</div>
              <div class="si-card-value">
                {_fmt_currency(current_price, currency)}
              </div>
              <div class="si-card-sub">As of last close</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc2:
        st.markdown(
            f"""
            <div class="si-card">
              <div class="si-card-label">Day {days_to_predict} Forecast</div>
              <div class="si-card-value">
                {_fmt_currency(final_price, currency)}
              </div>
              <div class="si-card-sub">{selected_model}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc3:
        st.markdown(
            f"""
            <div class="si-card">
              <div class="si-card-label">Expected Move</div>
              <div class="si-card-value">
                <span class="{fc_delta_cls}">
                  {fc_arrow} {_fmt_currency(abs(total_chg_f), currency)}
                </span>
              </div>
              <div class="si-card-sub">
                <span class="{fc_delta_cls}">{total_pct_f:+.2f}%</span>
                &nbsp;over {days_to_predict}d
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with fc4:
        st.markdown(
            f"""
            <div class="si-card">
              <div class="si-card-label">Horizon / Model</div>
              <div class="si-card-value" style="font-size:1rem;">
                {days_to_predict} days
              </div>
              <div class="si-card-sub">{model_used}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # RSI signal badge
    if last_rsi is not None:
        sig = _signal_badge(last_rsi)
        st.markdown(
            f"<div style='font-size:0.82rem;margin-top:0.3rem;'>"
            f"RSI Signal: {sig} &nbsp;(current RSI = <b>{last_rsi:.1f}</b>)"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ===========================================================================
    # ── SECTION 5: FORECAST CHART ─────────────────────────────────────────────
    # ===========================================================================
    st.markdown(
        '<p class="si-section">📈 Forecast Chart</p>',
        unsafe_allow_html=True,
    )

    fc_col, fc_table_col = st.columns([3, 1], gap="medium")

    with fc_col:
        forecast_fig = _build_forecast_chart(
            df, predictions, ticker, currency, days_to_predict,
        )
        st.plotly_chart(forecast_fig, use_container_width=True)

    with fc_table_col:
        st.markdown(
            f"<div style='font-size:0.78rem;font-weight:700;"
            f"color:#E6EDF3;margin-bottom:0.6rem;'>Day-by-Day Forecast</div>",
            unsafe_allow_html=True,
        )
        ref = current_price
        for i, price in enumerate(predictions, start=1):
            p = float(price)
            d = p - ref
            d_pct = (d / ref * 100) if ref != 0 else 0
            cls = "si-up" if d >= 0 else "si-down"
            arrow = "▲" if d > 0 else "▼"
            st.markdown(
                f"""
                <div class="fc-row">
                  <span class="fc-label">Day {i}</span>
                  <span class="fc-val">
                    {_fmt_currency(p, currency)}&nbsp;
                    <span class="{cls}" style="font-size:0.75rem;">
                      {arrow}{abs(d_pct):.1f}%
                    </span>
                  </span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            ref = p

# ===========================================================================
# ── SECTION 6: TECHNICAL ANALYSIS ────────────────────────────────────────────
# ===========================================================================
st.markdown(
    '<p class="si-section">🔬 Technical Analysis</p>',
    unsafe_allow_html=True,
)

ta_tab1, ta_tab2, ta_tab3, ta_tab4, ta_tab5, ta_tab6 = st.tabs([
    "RSI", "MACD", "Moving Averages", "Bollinger Bands",
    "Stochastic", "Volatility & ATR",
])

with ta_tab1:
    rsi_now = last_rsi
    rsi_sig = _signal_badge(rsi_now)
    c_rsi1, c_rsi2 = st.columns([3, 1], gap="medium")
    with c_rsi1:
        st.plotly_chart(_build_rsi_chart(df), use_container_width=True)
    with c_rsi2:
        st.markdown(
            f"""
            <div class="si-card" style="margin-top:0.5rem;">
              <div class="si-card-label">Current RSI</div>
              <div class="si-card-value">{f'{rsi_now:.1f}' if rsi_now else 'N/A'}</div>
              <div class="si-card-sub" style="margin-top:0.4rem;">{rsi_sig}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            """
            <div class="si-card" style="margin-top:0.6rem;
                 font-size:0.76rem;line-height:1.6;">
              <b>RSI Guide</b><br>
              🔴 ≥ 70 · Overbought<br>
              🟡 30–70 · Neutral<br>
              🟢 ≤ 30 · Oversold
            </div>
            """,
            unsafe_allow_html=True,
        )

with ta_tab2:
    st.plotly_chart(_build_macd_chart(df), use_container_width=True)
    # MACD current values
    macd_df = compute_macd(df)
    macd_now   = float(macd_df["MACD"].iloc[-1])
    signal_now = float(macd_df["MACD_Signal"].iloc[-1])
    hist_now   = float(macd_df["MACD_Hist"].iloc[-1])
    m1, m2, m3 = st.columns(3, gap="small")
    for col, lbl, val in [
        (m1, "MACD",     macd_now),
        (m2, "Signal",   signal_now),
        (m3, "Histogram",hist_now),
    ]:
        cls = "si-up" if val >= 0 else "si-down"
        col.markdown(
            f"""
            <div class="si-card" style="text-align:center;">
              <div class="si-card-label">{lbl}</div>
              <div class="si-card-value">
                <span class="{cls}">{val:+.4f}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ta_tab3:
    st.plotly_chart(_build_ma_chart(df, currency), use_container_width=True)
    ma_cols = st.columns(4, gap="small")
    for col, label, key in [
        (ma_cols[0], "SMA 20",  "SMA20"),
        (ma_cols[1], "SMA 50",  "SMA50"),
        (ma_cols[2], "SMA 200", "SMA200"),
        (ma_cols[3], "EMA 20",  "EMA20"),
    ]:
        val = float(df[key].iloc[-1]) if key in df.columns else None
        price_vs = ((current_price - val) / val * 100) if val else None
        v_cls = _delta_class(price_vs or 0)
        col.markdown(
            f"""
            <div class="si-card" style="text-align:center;">
              <div class="si-card-label">{label}</div>
              <div class="si-card-value" style="font-size:1rem;">
                {_fmt_currency(val, currency)}
              </div>
              <div class="si-card-sub">
                Price vs MA:&nbsp;
                <span class="{v_cls}">
                  {f'{price_vs:+.2f}%' if price_vs is not None else 'N/A'}
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ta_tab4:
    st.plotly_chart(_build_bb_chart(df, currency), use_container_width=True)
    bb_upper = float(df["BB_upper"].iloc[-1]) if "BB_upper" in df.columns else None
    bb_lower = float(df["BB_lower"].iloc[-1]) if "BB_lower" in df.columns else None
    bb_mid   = float(df["BB_mid"].iloc[-1])   if "BB_mid"   in df.columns else None
    bb_width = ((bb_upper - bb_lower) / bb_mid * 100) if (bb_upper and bb_lower and bb_mid) else None
    b1, b2, b3, b4 = st.columns(4, gap="small")
    for col, lbl, val in [
        (b1, "Upper Band",     _fmt_currency(bb_upper, currency)),
        (b2, "Middle (SMA20)", _fmt_currency(bb_mid,   currency)),
        (b3, "Lower Band",     _fmt_currency(bb_lower, currency)),
        (b4, "Band Width",     f"{bb_width:.2f}%" if bb_width else "N/A"),
    ]:
        col.markdown(
            f"""
            <div class="si-card" style="text-align:center;">
              <div class="si-card-label">{lbl}</div>
              <div class="si-card-value" style="font-size:1rem;">{val}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ta_tab5:
    st.plotly_chart(_build_stoch_chart(df), use_container_width=True)
    stoch_df = compute_stochastic(df)
    k_now = float(stoch_df["Stoch_K"].iloc[-1]) if "Stoch_K" in stoch_df.columns else None
    d_now = float(stoch_df["Stoch_D"].iloc[-1]) if "Stoch_D" in stoch_df.columns else None
    s1, s2 = st.columns(2, gap="small")
    for col, lbl, val in [(s1, "%K", k_now), (s2, "%D (Signal)", d_now)]:
        if val is not None:
            sig = "🔴 Overbought" if val >= 80 else ("🟢 Oversold" if val <= 20 else "🟡 Neutral")
        else:
            sig = "N/A"
        col.markdown(
            f"""
            <div class="si-card" style="text-align:center;">
              <div class="si-card-label">{lbl}</div>
              <div class="si-card-value">{f'{val:.1f}' if val is not None else 'N/A'}</div>
              <div class="si-card-sub">{sig}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

with ta_tab6:
    col_v, col_a = st.columns(2, gap="medium")
    with col_v:
        st.markdown("**20-Day Annualised Volatility**")
        st.plotly_chart(_build_volatility_chart(df), use_container_width=True)
        vol_df = compute_volatility(df)
        vol_now = float(vol_df["Volatility"].dropna().iloc[-1]) \
            if "Volatility" in vol_df.columns else None
        st.markdown(
            f"""
            <div class="si-card" style="text-align:center;margin-top:0.3rem;">
              <div class="si-card-label">Current Volatility</div>
              <div class="si-card-value">
                {f'{vol_now:.2f}%' if vol_now else 'N/A'}
              </div>
              <div class="si-card-sub">20-day annualised</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col_a:
        st.markdown("**Average True Range (ATR-14)**")
        st.plotly_chart(_build_atr_chart(df, currency), use_container_width=True)
        atr_df = compute_atr(df)
        atr_now = float(atr_df["ATR"].dropna().iloc[-1]) \
            if "ATR" in atr_df.columns else None
        st.markdown(
            f"""
            <div class="si-card" style="text-align:center;margin-top:0.3rem;">
              <div class="si-card-label">Current ATR</div>
              <div class="si-card-value">
                {_fmt_currency(atr_now, currency) if atr_now else 'N/A'}
              </div>
              <div class="si-card-sub">14-period EMA of True Range</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ===========================================================================
# ── SECTION 7: MODEL PERFORMANCE ─────────────────────────────────────────────
# ===========================================================================
if X is not None and y_seq is not None and predictions:
    st.markdown(
        '<p class="si-section">🎯 Model Performance</p>',
        unsafe_allow_html=True,
    )

    # ── Per-model metrics for the active forecast ─────────────────────────
    with st.spinner("Computing model performance metrics…"):
        metrics = _compute_model_metrics(df, scaler, X, y_seq)

    p1, p2, p3, p4 = st.columns(4, gap="small")
    for col, lbl, val, desc in [
        (p1, "MAE",
         f"{_fmt_currency(metrics['MAE'], currency)}" if metrics["MAE"] else "N/A",
         "Mean Absolute Error"),
        (p2, "RMSE",
         f"{_fmt_currency(metrics['RMSE'], currency)}" if metrics["RMSE"] else "N/A",
         "Root Mean Squared Error"),
        (p3, "MAPE",
         f"{metrics['MAPE']:.2f}%" if metrics["MAPE"] else "N/A",
         "Mean Absolute Percentage Error"),
        (p4, "R²",
         f"{metrics['R2']:.4f}" if metrics["R2"] is not None else "N/A",
         "Coefficient of Determination"),
    ]:
        col.markdown(
            f"""
            <div class="perf-card">
              <div class="perf-label">{lbl}</div>
              <div class="perf-value">{val}</div>
              <div class="perf-desc">{desc}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if metrics.get("n_test"):
        st.caption(
            f"LSTM in-sample metrics · last {metrics['n_test']} sequences "
            f"(20% pseudo hold-out) · lower MAE/RMSE/MAPE and R² → 1 is better."
        )

    # ── Model comparison table (Auto mode or always shown) ────────────────
    st.markdown(
        '<p class="si-section">📊 Model Comparison</p>',
        unsafe_allow_html=True,
    )

    _show_comparison = comparison_df is not None

    if not _show_comparison:
        # Single-model run: run a lightweight comparison in the background
        with st.spinner("Running model comparison (chronological validation)…"):
            try:
                comparison_df, _ = run_all_models(
                    df["Close"].rename("Close"),
                    steps=days_to_predict,
                    val_size=MODEL_VAL_SIZE,
                )
                _show_comparison = True
            except Exception as exc:
                st.info(f"Model comparison unavailable: {exc}")

    if _show_comparison and comparison_df is not None and not comparison_df.empty:
        # Identify the best model row
        best_name_cmp = (
            comparison_df[comparison_df["Status"] == "OK"].iloc[0]["Model"]
            if not comparison_df[comparison_df["Status"] == "OK"].empty
            else None
        )

        # Build display DataFrame
        disp_cols = ["Model", "MAE", "RMSE", "MAPE", "R2", "n_test", "Status"]
        disp_df   = comparison_df[[c for c in disp_cols if c in comparison_df.columns]].copy()

        def _fmt_val(v, suffix=""):
            try:
                return f"{float(v):.4f}{suffix}"
            except (TypeError, ValueError):
                return "N/A"

        for num_col, sfx in [("MAE",""), ("RMSE",""), ("MAPE","%"), ("R2","")]:
            if num_col in disp_df.columns:
                disp_df[num_col] = disp_df[num_col].apply(lambda v: _fmt_val(v, sfx))

        # Add Best indicator column
        disp_df.insert(0, "🏆", disp_df["Model"].apply(
            lambda m: "✅ Best" if m == best_name_cmp else ""
        ))

        st.dataframe(
            disp_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                "🏆":     st.column_config.TextColumn("", width="small"),
                "Model":  st.column_config.TextColumn("Model"),
                "MAE":    st.column_config.TextColumn("MAE ↓"),
                "RMSE":   st.column_config.TextColumn("RMSE ↓"),
                "MAPE":   st.column_config.TextColumn("MAPE ↓"),
                "R2":     st.column_config.TextColumn("R² ↑"),
                "n_test": st.column_config.TextColumn("Val Samples"),
                "Status": st.column_config.TextColumn("Status"),
            },
        )

        if best_name_cmp:
            st.success(
                f"✅ **Best model by RMSE:** {best_name_cmp}  "
                f"{'(currently selected)' if best_name_cmp == model_used else ''}"
            )

        st.caption(
            f"Chronological train/validation split · {MODEL_VAL_SIZE} held-out "
            "trading days · ↓ lower is better · ↑ higher is better. "
            "Metrics reflect in-sample validation performance and do **not** "
            "guarantee future forecast accuracy."
        )

    # ── All-model forecast overlay (if we have multi-model predictions) ────
    if len(all_predictions) > 1:
        st.markdown(
            '<p class="si-section">📈 All-Model Forecast Overlay</p>',
            unsafe_allow_html=True,
        )
        last_date  = df.index[-1]
        pred_dates = list(pd.bdate_range(start=last_date, periods=days_to_predict + 1)[1:])

        palette = [COLOR_PRIMARY, COLOR_UP, COLOR_DOWN, COLOR_WARNING,
                   COLOR_RSI, COLOR_SMA50, COLOR_SMA200, COLOR_EMA20]

        fig_overlay = go.Figure()
        # Historical close (last 30 days)
        hist_sl = df["Close"].iloc[-30:]
        fig_overlay.add_trace(go.Scatter(
            x=hist_sl.index, y=hist_sl.values,
            name="Historical Close",
            line=dict(color=COLOR_NEUTRAL, width=1.8),
        ))
        fig_overlay.add_vline(
            x=last_date, line_dash="dot",
            line_color=COLOR_NEUTRAL, line_width=1,
        )
        for idx, (mname, mpreds) in enumerate(all_predictions.items()):
            is_best = mname == model_used
            fig_overlay.add_trace(go.Scatter(
                x=pred_dates[:len(mpreds)],
                y=[float(p) for p in mpreds],
                name=f"{mname}{'  ✅' if is_best else ''}",
                line=dict(
                    color=palette[idx % len(palette)],
                    width=3 if is_best else 1.5,
                    dash="solid" if is_best else "dot",
                ),
            ))
        fig_overlay.update_layout(
            template=CHART_TEMPLATE,
            height=CHART_HEIGHT_FORECAST,
            hovermode="x unified",
            xaxis_rangeslider_visible=False,
            yaxis_title=f"Price ({currency})",
            margin=dict(l=0, r=0, t=30, b=0),
            legend=dict(orientation="h", yanchor="bottom", y=1.01,
                        xanchor="right", x=1, font=dict(size=11)),
        )
        st.plotly_chart(fig_overlay, use_container_width=True)
        st.caption(
            "Bold / solid line = selected/best model. "
            "All forecasts are experimental — not financial advice."
        )

# ===========================================================================
# ── SECTION 8: HISTORICAL DATA ───────────────────────────────────────────────
# ===========================================================================
st.markdown(
    '<p class="si-section">📂 Historical Data</p>',
    unsafe_allow_html=True,
)

with st.expander(
    f"Show OHLCV data — {ticker}  ({len(raw_df)} trading days)",
    expanded=False,
):
    display_df = raw_df[["Open", "High", "Low", "Close", "Volume"]].copy()
    display_df.index = display_df.index.strftime("%Y-%m-%d")
    display_df.index.name = "Date"

    # Colour Close column red/green
    def _style_close(val):
        return ""  # Streamlit Styler gradients are too slow for large DFs

    st.dataframe(
        display_df.sort_index(ascending=False),
        use_container_width=True,
        height=380,
    )
    st.caption(
        f"Showing {len(display_df)} rows · Period: {hist_period_label} · "
        f"Source: {DATA_SOURCE}"
    )

# ===========================================================================
# ── SECTION 9: DOWNLOADS ─────────────────────────────────────────────────────
# ===========================================================================
st.markdown(
    '<p class="si-section">📥 Download</p>',
    unsafe_allow_html=True,
)

dl1, dl2, dl3 = st.columns(3, gap="small")

with dl1:
    csv_hist = _make_hist_csv(raw_df)
    st.download_button(
        label="📊 Historical Data (CSV)",
        data=csv_hist,
        file_name=f"{ticker}_historical_{hist_period}.csv",
        mime="text/csv",
        use_container_width=True,
    )
    st.caption(f"{len(raw_df)} rows · OHLCV")

with dl2:
    if predictions:
        csv_fc = _make_forecast_csv(
            ticker, current_price, predictions, df.index[-1], currency,
        )
        st.download_button(
            label="🔮 Forecast Data (CSV)",
            data=csv_fc,
            file_name=f"{ticker}_forecast_{days_to_predict}d.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.caption(f"{days_to_predict} day forecast · {selected_model}")
    else:
        st.button(
            "🔮 Forecast Data (CSV)",
            disabled=True,
            use_container_width=True,
        )
        st.caption("Run forecast to enable")

with dl3:
    metrics_for_report = metrics if (X is not None and predictions) else None
    report_txt = _make_report_text(
        ticker, info, current_price, predictions or [],
        metrics_for_report, currency, hist_period,
        model_used, days_to_predict,
    )
    st.download_button(
        label="📄 Full Report (TXT)",
        data=report_txt,
        file_name=f"{ticker}_stockintel_report_{datetime.now().strftime('%Y%m%d')}.txt",
        mime="text/plain",
        use_container_width=True,
    )
    st.caption("Summary · Forecast · Model metrics · Disclaimer")

# ===========================================================================
# ── SECTION 10: DISCLAIMER ───────────────────────────────────────────────────
# ===========================================================================
st.markdown(
    f'<div class="si-disclaimer">{DISCLAIMER}</div>',
    unsafe_allow_html=True,
)

# Attribution footer
st.markdown(
    f"""
    <div style="text-align:center;font-size:0.7rem;
                color:{COLOR_NEUTRAL};margin-top:1.8rem;
                padding:1rem 0 0.5rem 0;
                border-top:1px solid {COLOR_BORDER};">
      Built by <strong style="color:#E6EDF3;">{AUTHOR_NAME}</strong>
      &nbsp;·&nbsp; {APP_NAME} v{APP_VERSION}
      &nbsp;·&nbsp;
      <a href="{GITHUB_URL}" target="_blank"
         style="color:{COLOR_PRIMARY};text-decoration:none;">
        GitHub
      </a>
      &nbsp;·&nbsp;
      Original project:
      <a href="{ORIGINAL_REPO_URL}" target="_blank"
         style="color:{COLOR_PRIMARY};text-decoration:none;">
        {ORIGINAL_PROJECT_NAME}
      </a>
      by {ORIGINAL_AUTHOR} (MIT License)
    </div>
    """,
    unsafe_allow_html=True,
)
