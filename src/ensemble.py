"""Weighted ensemble that links the LSTM and MLP: forecast = w * LSTM + (1 - w) * MLP.

The weight ``w`` is fitted on the VALIDATION period only (grid search over 0.00 ... 1.00 on the
selection metric), never on test. Because w = 0 and w = 1 are on the grid, the ensemble's validation
score is never worse than the better single model; whether that holds on unseen data is shown by the
test metrics and ``selection.test_ranking_agrees`` in the evaluation report.

The ensemble also carries an empirical prediction interval: per forecast step, the 10th and 90th
percentiles of its validation errors (actual - forecast) are added to the forecast, giving an 80 %
range. Its coverage on the test period is measured and reported.

Fitted by ``python -m src.evaluate`` and saved to ``artifacts/ensemble/ensemble.json``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .artifacts import to_jsonable
from .config import ARTIFACTS_DIR
from .errors import ArtifactError
from .metrics import mae, rmse, wape

ENSEMBLE_NAME = "ensemble"
COMPONENTS = ("lstm", "mlp")
ENSEMBLE_FILE = "ensemble.json"
INTERVAL_LEVEL = 80                         # percent of actuals expected inside [lower, upper]
WEIGHT_STEP = 0.01
_METRICS = {"mae": mae, "rmse": rmse, "wape": wape}


def fit_weights(y_true: np.ndarray, preds: Dict[str, np.ndarray], metric: str = "rmse",
                step: float = WEIGHT_STEP) -> Dict[str, float]:
    """Weight of each component (sums to 1) that minimises ``metric`` on the given (validation) data."""
    a, b = COMPONENTS
    score_fn = _METRICS[metric]
    best_w, best_score = 1.0, np.inf
    for w in np.round(np.arange(0.0, 1.0 + step / 2, step), 6):
        score = score_fn(y_true, w * preds[a] + (1 - w) * preds[b])
        if score is not None and score < best_score - 1e-12:
            best_w, best_score = float(w), score
    return {a: best_w, b: round(1.0 - best_w, 6)}


def combine(preds: Dict[str, np.ndarray], weights: Dict[str, float]) -> np.ndarray:
    return sum(weights[m] * np.asarray(preds[m], dtype=float) for m in weights)


def interval_offsets(y_true: np.ndarray, y_pred: np.ndarray, level: float = INTERVAL_LEVEL) -> Dict[str, list]:
    """Per-step error quantiles; forecast + offset gives the lower / upper bound."""
    err = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)       # (N, H)
    tail = (100 - level) / 2
    return {"lower": np.percentile(err, tail, axis=0).tolist(),
            "upper": np.percentile(err, 100 - tail, axis=0).tolist()}


def apply_interval(pred: np.ndarray, offsets: Dict[str, list], clip_non_negative: bool = True):
    lower = pred + np.asarray(offsets["lower"])
    upper = pred + np.asarray(offsets["upper"])
    if clip_non_negative:
        lower = np.clip(lower, 0.0, None)
    return lower, np.maximum(upper, lower)


def interval_coverage(y_true: np.ndarray, pred: np.ndarray, offsets: Dict[str, list],
                      clip_non_negative: bool = True) -> float:
    lower, upper = apply_interval(pred, offsets, clip_non_negative)
    y = np.asarray(y_true, dtype=float)
    return float(np.mean((y >= lower) & (y <= upper)) * 100.0)


def ensemble_dir(root=None) -> Path:
    return Path(root or ARTIFACTS_DIR) / ENSEMBLE_NAME


def save_ensemble(info: Dict[str, Any], root=None) -> Path:
    d = ensemble_dir(root)
    d.mkdir(parents=True, exist_ok=True)
    path = d / ENSEMBLE_FILE
    path.write_text(json.dumps(to_jsonable(info), indent=2), encoding="utf-8")
    return path


def load_ensemble(root=None) -> Dict[str, Any]:
    path = ensemble_dir(root) / ENSEMBLE_FILE
    if not path.exists():
        raise ArtifactError(f"No fitted ensemble at {path}. Train both models, then run "
                            "python -m src.evaluate --csv <training csv> to fit it.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ArtifactError(f"Ensemble file {path} is corrupt: {exc}") from exc
