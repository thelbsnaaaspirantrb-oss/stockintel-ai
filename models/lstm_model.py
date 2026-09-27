"""
models/lstm_model.py
--------------------
StockIntel AI — Intelligent Stock Analytics & Forecasting Platform
Author : Rohan Bondre

LSTM forecasting model — wraps the pre-trained Keras model in the
common BaseForecaster interface.

Design
------
- Loads lstm_model.h5 from the project root (path configurable).
- Uses MinMaxScaler on Close prices, as per the original training pipeline.
- Validation: builds sliding-window sequences from the val portion and
  runs batch inference — faster than the original one-by-one loop.
- Recursive prediction: each predicted (scaled) value is rolled into the
  sequence window for the next step.
- tensorflow import is deferred to __init__ so the module is importable
  without TF installed; ImportError is raised at instantiation time.

Note on inference scaler
------------------------
The scaler is re-fit on the full available series at fit() time.
This is consistent with the original app.py approach.  A future
improvement is to persist the training scaler alongside the weights.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from models.base_model import BaseForecaster

# ---------------------------------------------------------------------------
# Deferred TF import
# ---------------------------------------------------------------------------
_TF_OK: bool = True
try:
    from tensorflow.keras.models import load_model as _keras_load_model
except ImportError:
    _TF_OK = False
    _keras_load_model = None  # type: ignore[assignment]

_DEFAULT_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "lstm_model.h5",
)


class LSTMModel(BaseForecaster):
    """
    Pre-trained 2-layer LSTM wrapped in the BaseForecaster interface.

    Args:
        model_path      : Path to lstm_model.h5.  Defaults to project root.
        sequence_length : Input window length (must match training — 60).

    Raises:
        ImportError : If tensorflow is not installed.
        FileNotFoundError : If the .h5 file does not exist.

    Example
    -------
    >>> model = LSTMModel()
    >>> model.fit(close_series)
    >>> preds = model.predict(steps=7)
    """

    name: str = "LSTM"

    def __init__(
        self,
        model_path: str = _DEFAULT_MODEL_PATH,
        sequence_length: int = 60,
    ) -> None:
        if not _TF_OK:
            raise ImportError(
                "tensorflow is required for LSTMModel. "
                "Install it with: pip install tensorflow"
            )
        super().__init__()
        self._model_path      = model_path
        self._sequence_length = sequence_length
        self._keras_model: Any = None
        self._scaler: MinMaxScaler | None = None
        self._last_sequence: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, series: pd.Series, val_size: int = 30) -> "LSTMModel":
        """
        Load the pre-trained model, scale the series, build sequences,
        and compute validation metrics.

        Args:
            series   : Chronological Close-price Series.
            val_size : Tail observations withheld for validation.

        Returns:
            self
        """
        min_len = self._sequence_length + val_size + 1
        if len(series) < min_len:
            raise ValueError(
                f"Series length ({len(series)}) must be >= "
                f"sequence_length + val_size + 1 = {min_len}."
            )

        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"LSTM model file not found: {self._model_path}\n"
                "Run scripts/train_model.py to generate it."
            )

        # Load the Keras model (cached by @st.cache_resource in app.py
        # for the Streamlit context; here we load fresh for standalone use)
        self._keras_model = _keras_load_model(self._model_path)

        # Fit scaler on training observations only; transform the full series
        train_vals = series.iloc[:-val_size].to_numpy(dtype=float).reshape(-1, 1)
        all_vals = series.to_numpy(dtype=float).reshape(-1, 1)
        self._scaler = MinMaxScaler(feature_range=(0, 1))
        self._scaler.fit(train_vals)
        scaled = self._scaler.transform(all_vals).flatten()

        # Build sliding windows
        seqs, targets = [], []
        for i in range(self._sequence_length, len(scaled)):
            seqs.append(scaled[i - self._sequence_length : i])
            targets.append(scaled[i])
        X_all = np.array(seqs).reshape(-1, self._sequence_length, 1)
        y_all = np.array(targets)

        # Chronological split — validation is the tail
        X_val = X_all[-val_size:]
        y_val_scaled = y_all[-val_size:].reshape(-1, 1)

        # Batch inference on validation
        y_pred_scaled = self._keras_model.predict(X_val, verbose=0)
        y_val  = self._scaler.inverse_transform(y_val_scaled).flatten()
        y_pred = self._scaler.inverse_transform(y_pred_scaled).flatten()

        # Store last training sequence for recursive prediction
        self._last_sequence = X_all[-(val_size + 1)].copy()

        self._val_y_true = y_val
        self._val_y_pred = y_pred
        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    def predict(self, steps: int) -> list[float]:
        """
        Recursively forecast ``steps`` future closing prices.

        Args:
            steps : Forecast horizon.

        Returns:
            list[float] of length ``steps``.
        """
        self._require_fitted()

        seq = self._last_sequence.copy().reshape(1, self._sequence_length, 1)
        predictions: list[float] = []

        for _ in range(steps):
            pred_scaled = self._keras_model.predict(seq, verbose=0)
            price = float(
                self._scaler.inverse_transform(pred_scaled)[0][0]
            )
            predictions.append(price)
            seq = np.roll(seq, -1, axis=1)
            seq[0, -1, 0] = pred_scaled[0][0]

        return predictions

    # ------------------------------------------------------------------
    def evaluate(self) -> dict[str, Any]:
        """Return chronological validation-set metrics."""
        self._require_fitted()
        return self._metrics(self._val_y_true, self._val_y_pred)
