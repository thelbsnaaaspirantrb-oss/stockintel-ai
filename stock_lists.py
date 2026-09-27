"""
stock_lists.py
--------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Purpose
-------
Single source of truth for all supported ticker symbols.
Import ``indian_stocks`` and ``us_stocks`` from this module wherever a
stock list is needed — do NOT duplicate these lists in other files.

Coverage
--------
indian_stocks : Nifty 50 + Nifty Next 50  (~100 NSE tickers, suffix .NS)
us_stocks     : S&P 500 top 50 + Nasdaq top 50  (~100 US tickers)

All symbols use Yahoo Finance ticker conventions.

Original project this module is derived from:
    ModelHub by BadakalaYashwanth
    https://github.com/BadakalaYashwanth/ModelHub
    Licensed under the MIT License — see LICENSE for details.
"""

# ---------------------------------------------------------------------------
# Indian Equities — Nifty 50 + Nifty Next 50
# (Yahoo Finance NSE format: TICKER.NS)
# ---------------------------------------------------------------------------
indian_stocks: list[str] = [
    # ── Nifty 50 ─────────────────────────────────────────────────────────────
    "ADANIENT.NS",   "ADANIPORTS.NS", "ASIANPAINT.NS", "AXISBANK.NS",
    "BAJAJ-AUTO.NS", "BAJFINANCE.NS", "BAJAJFINSV.NS", "BPCL.NS",
    "BHARTIARTL.NS", "BRITANNIA.NS",  "CIPLA.NS",      "COALINDIA.NS",
    "DIVISLAB.NS",   "DRREDDY.NS",    "EICHERMOT.NS",  "GRASIM.NS",
    "HCLTECH.NS",    "HDFCBANK.NS",   "HDFCLIFE.NS",   "HEROMOTOCO.NS",
    "HINDALCO.NS",   "HINDUNILVR.NS", "ICICIBANK.NS",  "INDUSINDBK.NS",
    "INFY.NS",       "ITC.NS",        "JSWSTEEL.NS",   "KOTAKBANK.NS",
    "LT.NS",         "M&M.NS",        "MARUTI.NS",     "NESTLEIND.NS",
    "NTPC.NS",       "ONGC.NS",       "POWERGRID.NS",  "RELIANCE.NS",
    "SBILIFE.NS",    "SBIN.NS",       "SUNPHARMA.NS",  "TATACONSUM.NS",
    "TATAMOTORS.NS", "TATASTEEL.NS",  "TECHM.NS",      "TITAN.NS",
    "TCS.NS",        "ULTRACEMCO.NS", "UPL.NS",        "WIPRO.NS",
    "HINDPETRO.NS",  "BAJAJHLDNG.NS",

    # ── Nifty Next 50 ────────────────────────────────────────────────────────
    "ABB.NS",        "ADANIGREEN.NS", "ADANITRANS.NS", "ALKEM.NS",
    "AMBUJACEM.NS",  "AUROPHARMA.NS", "BANKBARODA.NS", "BERGEPAINT.NS",
    "BIOCON.NS",     "BOSCHLTD.NS",   "CANBK.NS",      "CHOLAFIN.NS",
    "DABUR.NS",      "DLF.NS",        "GAIL.NS",       "GODREJCP.NS",
    "HAVELLS.NS",    "ICICIPRULI.NS", "IGL.NS",        "INDIGO.NS",
    "L&TFH.NS",      "LICI.NS",       "MCDOWELL-N.NS", "MOTHERSUMI.NS",
    "NMDC.NS",       "PEL.NS",        "PGHH.NS",       "PIDILITIND.NS",
    "PIIND.NS",      "PNB.NS",        "RECLTD.NS",     "SAIL.NS",
    "SHREECEM.NS",   "SIEMENS.NS",    "SRF.NS",        "TORNTPHARM.NS",
    "TRENT.NS",      "TVSMOTOR.NS",   "UBL.NS",        "VEDL.NS",
    "VOLTAS.NS",     "YESBANK.NS",    "ZEEL.NS",       "IDFCFIRSTB.NS",
    "INDUSTOWER.NS",
]

# ---------------------------------------------------------------------------
# US Equities — S&P 500 top 50 + Nasdaq top 50
# (Yahoo Finance standard US format: TICKER)
# ---------------------------------------------------------------------------
us_stocks: list[str] = [
    # ── S&P 500 top 50 ───────────────────────────────────────────────────────
    "AAPL",  "MSFT",  "AMZN",  "GOOGL", "META",  "TSLA",  "NVDA",  "BRK.B",
    "UNH",   "JNJ",   "JPM",   "V",     "PG",    "MA",    "HD",    "XOM",
    "LLY",   "CVX",   "ABBV",  "AVGO",  "PEP",   "KO",    "MRK",   "COST",
    "WMT",   "BAC",   "ADBE",  "CRM",   "ACN",   "TMO",   "INTC",  "NKE",
    "LIN",   "ABT",   "NEE",   "MCD",   "DHR",   "WFC",   "TXN",   "UNP",
    "MDT",   "MS",    "QCOM",  "HON",   "PM",    "LOW",   "IBM",   "AMGN",
    "SBUX",  "INTU",

    # ── Nasdaq top 50 ────────────────────────────────────────────────────────
    "AMD",   "PYPL",  "BKNG",  "AMAT",  "CSCO",  "ADP",   "ISRG",  "VRTX",
    "GILD",  "ADI",   "REGN",  "ZM",    "MAR",   "LRCX",  "IDXX",  "ASML",
    "PANW",  "CTAS",  "CDNS",  "FISV",  "SNPS",  "EXC",   "EA",    "ILMN",
    "ROST",  "NXPI",  "ORLY",  "KLAC",  "FAST",  "MNST",  "CHTR",  "PAYX",
    "MTCH",  "BIIB",  "WBA",   "ALGN",  "KHC",   "BIDU",  "NTES",  "DOCU",
    "OKTA",  "TEAM",  "CRWD",  "ZS",    "SPLK",  "DDOG",  "MDB",   "PLTR",
    "ROKU",  "U",
]
