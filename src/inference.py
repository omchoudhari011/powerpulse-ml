"""Shared prediction helpers (used by train, evaluate and predict)."""
from __future__ import annotations

import numpy as np


def inputs_for_model(model_name: str, split) -> np.ndarray:
    return split.X_seq if model_name == "lstm" else split.X_tab


def predict_original_units(model, preprocessor, X: np.ndarray, clip_non_negative: bool = True,
                           batch_size: int = 256) -> np.ndarray:
    """Model output (scaled) -> original units (kW / kWh), shape (N, H)."""
    if len(X) == 0:
        return np.empty((0, 0))
    scaled = np.asarray(model.predict(X, batch_size=batch_size, verbose=0))
    out = preprocessor.inverse_target(scaled)
    return np.clip(out, 0.0, None) if clip_non_negative else out
