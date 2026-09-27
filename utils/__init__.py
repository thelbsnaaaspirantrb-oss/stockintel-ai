"""
utils
-----
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Package containing helper modules for data acquisition and preprocessing.

Modules
-------
data_fetcher  — yfinance wrappers with retry / rate-limit mitigation.
                Public: fetch_indian_stock(), fetch_us_stock()

preprocess    — Technical indicator computation and LSTM sequence builder.
                Public: add_technical_indicators(), preprocess_for_lstm()
"""
