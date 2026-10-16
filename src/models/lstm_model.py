"""Configurable LSTM forecaster (primary model)."""
from __future__ import annotations

from . import require_keras


def build_lstm_model(input_steps: int, n_features: int, horizon: int, units: int = 64,
                     num_layers: int = 1, dropout: float = 0.2, learning_rate: float = 1e-3):
    """Stacked LSTM -> Dense(horizon). Direct multi-step forecast of the next ``horizon`` steps.

    Input : (batch, input_steps, n_features)   Output: (batch, horizon)
    Adam optimiser, MSE loss.
    """
    if num_layers not in (1, 2):
        raise ValueError("num_layers must be 1 or 2")
    _, keras = require_keras()
    inputs = keras.Input(shape=(input_steps, n_features), name="sequence")
    x = inputs
    for i in range(num_layers):
        x = keras.layers.LSTM(units, return_sequences=(i < num_layers - 1), name=f"lstm_{i + 1}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)
    outputs = keras.layers.Dense(horizon, name="forecast")(x)
    model = keras.Model(inputs, outputs, name="powerpulse_lstm")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model
