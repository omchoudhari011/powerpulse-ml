"""Model builders. TensorFlow is imported lazily so data code/tests work without it."""
from __future__ import annotations

import os
import random
from typing import Any, Dict

import numpy as np


def require_keras():
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required to build, train or load the LSTM/MLP models. "
            "Install it with: pip install -r requirements.txt (Python 3.10-3.12 recommended)."
        ) from exc
    return tf, tf.keras


def set_global_seed(seed: int, deterministic_ops: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import tensorflow as tf
    except ImportError:
        return
    tf.keras.utils.set_random_seed(seed)
    if deterministic_ops:
        tf.config.experimental.enable_op_determinism()


def fit_model(model, X_train, Y_train, X_val, Y_val, *, batch_size: int, max_epochs: int,
              patience: int, shuffle: bool = True, verbose: int = 1) -> Dict[str, Any]:
    """Train with Adam/MSE (already compiled) and early stopping on validation loss."""
    _, keras = require_keras()
    stopper = keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True)
    h = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=max_epochs,
                  batch_size=batch_size, shuffle=shuffle, callbacks=[stopper], verbose=verbose)
    history = {k: [float(v) for v in vals] for k, vals in h.history.items()}
    val_loss = history["val_loss"]
    return {
        "history": history,
        "epochs_run": len(val_loss),
        "best_epoch": int(np.argmin(val_loss)) + 1,
        "best_val_loss_scaled": float(np.min(val_loss)),
        "stopped_early": len(val_loss) < max_epochs,
    }
