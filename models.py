"""
models.py
---------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

Purpose
-------
Defines the neural-network architectures used for stock price forecasting.
Currently exposes a 2-layer LSTM model. Additional architectures (GRU,
Transformer) are planned for Phase 4 of the roadmap.

Public API
----------
    build_lstm_model(input_shape) -> keras.Sequential

Original project this module is derived from:
    ModelHub by BadakalaYashwanth
    https://github.com/BadakalaYashwanth/ModelHub
    Licensed under the MIT License — see LICENSE for details.
"""

from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential


def build_lstm_model(input_shape: tuple) -> Sequential:
    """
    Build and compile a 2-layer stacked LSTM model for stock price prediction.

    Architecture
    ------------
    Input  →  LSTM(50, return_sequences=True)
           →  Dropout(0.2)
           →  LSTM(50, return_sequences=False)
           →  Dropout(0.2)
           →  Dense(25)
           →  Dense(1)          ← Predicted next-day close price (scaled 0–1)

    The model is compiled with the Adam optimiser and Mean Squared Error loss,
    which is standard for single-step regression on normalised targets.

    Args:
        input_shape (tuple): Shape of one input sample, e.g. ``(60, 1)``
            for a 60-day univariate sequence. The first dimension is the
            sequence length; the second is the number of features.

    Returns:
        keras.Sequential: Compiled LSTM model ready for ``model.fit()``
            or ``model.predict()``.

    Example:
        >>> model = build_lstm_model((60, 1))
        >>> model.summary()
        ...
        Total params: ~28,000
    """
    model = Sequential(
        [
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1),
        ],
        name="stockintel_lstm",
    )
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model
