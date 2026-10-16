"""Multilayer-perceptron comparison model."""
from __future__ import annotations

from typing import Sequence

from . import require_keras


def build_mlp_model(n_features: int, horizon: int, hidden_units: Sequence[int] = (128, 64),
                    dropout: float = 0.0, learning_rate: float = 1e-3):
    """Dense(ReLU) stack -> Dense(horizon) on lagged-target + calendar/weather/occupancy features.

    Input : (batch, n_features)   Output: (batch, horizon)
    Adam optimiser, MSE loss.
    """
    _, keras = require_keras()
    inputs = keras.Input(shape=(n_features,), name="features")
    x = inputs
    for i, units in enumerate(hidden_units):
        x = keras.layers.Dense(int(units), activation="relu", name=f"dense_{i + 1}")(x)
        if dropout > 0:
            x = keras.layers.Dropout(dropout, name=f"dropout_{i + 1}")(x)
    outputs = keras.layers.Dense(horizon, name="forecast")(x)
    model = keras.Model(inputs, outputs, name="powerpulse_mlp")
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse", metrics=["mae"])
    return model
