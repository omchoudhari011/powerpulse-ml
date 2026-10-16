"""Calendar features, lag features and leakage-safe sample (window) construction.

Conventions
-----------
* A *sample* is identified by its forecast origin row ``t`` (the last observed row).
* Inputs only use rows ``<= t``; targets are rows ``t+1 .. t+H``.
* "Lag k" means the k-th most recent observation at the origin: ``y[t-(k-1)]``, so lag 1 is
  the latest observed value. Using lags relative to the origin guarantees the same lag set
  is valid for every horizon step of a direct multi-step forecast.
"""
from __future__ import annotations

import math
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .config import Config

CALENDAR_COLUMNS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "month_sin", "month_cos"]


# ------------------------------------------------------------------ time resolution
def hours_to_steps(hours: float, interval_seconds: float) -> int:
    """Convert a duration in hours to a step count for the dataset's sampling interval.

    For intervals longer than one hour (e.g. daily data) "hours" cannot be honoured, so the
    value is interpreted as a number of steps (24 -> 24 observations).
    """
    if interval_seconds <= 3600:
        steps = round(hours * 3600.0 / interval_seconds)
    else:
        steps = round(hours)
    return max(1, int(steps))


def resolve_steps(cfg: Config, interval_seconds: float) -> Dict:
    lags = sorted({hours_to_steps(h, interval_seconds) for h in cfg.features.lag_hours})
    return {
        "interval_seconds": int(round(interval_seconds)),
        "mode": "time" if interval_seconds <= 3600 else "steps",
        "input_steps": hours_to_steps(cfg.train.input_hours, interval_seconds),
        "horizon_steps": hours_to_steps(cfg.train.horizon_hours, interval_seconds),
        "lag_steps": lags,
    }


def required_rows(cfg: Config, interval_seconds: float) -> int:
    """Lower bound on regular-grid rows needed for the configured split sizes."""
    s = resolve_steps(cfg, interval_seconds)
    L, H, d = s["input_steps"], s["horizon_steps"], cfg.data
    test_frac = 1.0 - d.train_frac - d.val_frac
    return max(
        math.ceil((d.min_train_samples + H + L - 1) / d.train_frac),
        math.ceil((d.min_val_samples + H - 1) / d.val_frac),
        math.ceil((d.min_test_samples + H - 1) / test_frac),
    )


# ------------------------------------------------------------------ calendar features
def calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    hour = index.hour + index.minute / 60.0
    dow = index.dayofweek.to_numpy()
    month = index.month.to_numpy()
    out = pd.DataFrame(index=index)
    out["hour_sin"] = np.sin(2 * np.pi * np.asarray(hour) / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * np.asarray(hour) / 24.0)
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend"] = (dow >= 5).astype(float)
    out["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    out["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)
    return out[CALENDAR_COLUMNS]


def build_feature_frame(clean_frame: pd.DataFrame, preprocessor, use_calendar: bool = True) -> pd.DataFrame:
    """Scaled [target, exogenous..., calendar...] frame on the regular grid (column 0 = target)."""
    scaled = preprocessor.transform(clean_frame)
    if use_calendar:
        scaled = pd.concat([scaled, calendar_features(clean_frame.index)], axis=1)
    return scaled


# ------------------------------------------------------------------ windows
def make_sequences(values: np.ndarray, origins: np.ndarray, input_steps: int) -> np.ndarray:
    """(N, input_steps, F) windows ending at each origin (inclusive)."""
    idx = origins[:, None] + np.arange(-input_steps + 1, 1)[None, :]
    return values[idx]


def make_targets(series: np.ndarray, origins: np.ndarray, horizon: int) -> np.ndarray:
    """(N, horizon) values at rows t+1 .. t+H."""
    idx = origins[:, None] + np.arange(1, horizon + 1)[None, :]
    return series[idx]


def make_tabular(values: np.ndarray, origins: np.ndarray, lag_steps: Sequence[int]) -> np.ndarray:
    """MLP inputs: target lags (column 0 of ``values``) + all other features at the origin."""
    lag_cols = [values[origins - (k - 1), 0] for k in lag_steps]
    others = values[origins, 1:]
    return np.column_stack(lag_cols + [others]) if others.shape[1] else np.column_stack(lag_cols)


def tabular_feature_names(lag_steps: Sequence[int], seq_feature_names: Sequence[str]) -> List[str]:
    return [f"target_lag_{k}" for k in lag_steps] + list(seq_feature_names[1:])


def valid_origins(values: np.ndarray, target_observed: np.ndarray, input_steps: int,
                  horizon: int, lag_steps: Sequence[int]) -> np.ndarray:
    """Origins whose input window, lag features and target window are all usable.

    * input window / exogenous features must contain no NaN (forward-filled gaps are fine);
    * lag values of the target must not be NaN;
    * every target value ``t+1..t+H`` must be a *genuinely observed* reading (never an
      imputed one), so metrics are computed against real data only.
    """
    n = len(values)
    warmup = max(input_steps, max(lag_steps))
    origins = np.arange(warmup - 1, n - horizon)
    if len(origins) == 0:
        return origins
    row_bad = np.isnan(values).any(axis=1).astype(int)
    c_bad = np.concatenate([[0], np.cumsum(row_bad)])
    unobserved = (~target_observed).astype(int)
    c_un = np.concatenate([[0], np.cumsum(unobserved)])

    win_bad = c_bad[origins + 1] - c_bad[origins - input_steps + 1]
    tgt_unobs = c_un[origins + horizon + 1] - c_un[origins + 1]
    ok = (win_bad == 0) & (tgt_unobs == 0)
    target_nan = np.isnan(values[:, 0])
    for k in lag_steps:
        ok &= ~target_nan[origins - (k - 1)]
    return origins[ok]


def assign_splits(origins: np.ndarray, horizon: int, train_end: int, val_end: int, n_rows: int) -> Dict[str, np.ndarray]:
    """Chronological split of origins. A sample belongs to a split only if its *whole target
    window* lies inside that split's rows; its inputs may reach back into earlier rows (history),
    never forward."""
    bounds = {"train": (0, train_end), "validation": (train_end, val_end), "test": (val_end, n_rows)}
    return {
        name: origins[(origins + 1 >= s) & (origins + horizon <= e - 1)]
        for name, (s, e) in bounds.items()
    }


def build_inference_inputs(values: np.ndarray, origin_idx: int, input_steps: int, lag_steps: Sequence[int]):
    """Inputs for a single forecast origin (no target window required)."""
    from .errors import InsufficientDataError

    warmup = max(input_steps, max(lag_steps))
    if origin_idx < warmup - 1:
        raise InsufficientDataError(
            f"Need at least {warmup} rows of history before the forecast origin "
            f"(longest lag / input window), but only {origin_idx + 1} are available."
        )
    o = np.array([origin_idx])
    seq = make_sequences(values, o, input_steps)
    tab = make_tabular(values, o, lag_steps)
    lag_target = [values[origin_idx - (k - 1), 0] for k in lag_steps]
    if np.isnan(seq).any() or np.isnan(tab).any() or np.isnan(lag_target).any():
        raise InsufficientDataError(
            "The history window before the forecast origin contains missing values that could "
            "not be forward-filled (gap longer than max_gap_hours). Supply more complete recent data."
        )
    return seq.astype(np.float32), tab.astype(np.float32)
