"""Forecast metrics. Zero denominators and empty inputs are handled explicitly (None, never fake)."""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

_EPS = 1e-9


def _arr(y_true, y_pred) -> Tuple[np.ndarray, np.ndarray]:
    a, b = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: y_true {a.shape} vs y_pred {b.shape}")
    return a, b


def mae(y_true, y_pred) -> Optional[float]:
    a, b = _arr(y_true, y_pred)
    return float(np.mean(np.abs(a - b))) if a.size else None


def rmse(y_true, y_pred) -> Optional[float]:
    a, b = _arr(y_true, y_pred)
    return float(np.sqrt(np.mean((a - b) ** 2))) if a.size else None


def wape(y_true, y_pred) -> Optional[float]:
    """sum|y - yhat| / sum|y| in percent. None when sum|y| ~ 0 (mathematically undefined)."""
    a, b = _arr(y_true, y_pred)
    if not a.size:
        return None
    denom = float(np.sum(np.abs(a)))
    if denom < _EPS:
        return None
    return float(np.sum(np.abs(a - b)) / denom * 100.0)


def compute_metrics(y_true, y_pred) -> Dict[str, Any]:
    a, b = _arr(y_true, y_pred)
    out: Dict[str, Any] = {"n_values": int(a.size), "mae": mae(a, b), "rmse": rmse(a, b), "wape_percent": wape(a, b)}
    if out["wape_percent"] is None:
        out["wape_note"] = "WAPE undefined (no data or sum of |actual| is zero)."
    return out


def mae_by_horizon(y_true, y_pred) -> list:
    a, b = _arr(y_true, y_pred)
    if a.ndim != 2 or not a.size:
        return []
    return [float(v) for v in np.mean(np.abs(a - b), axis=0)]


def peak_demand_arrays(y_true: np.ndarray, y_pred: np.ndarray, stride: int) -> Tuple[np.ndarray, ...]:
    """Peak of each non-overlapping forecast window (rows ``0, stride, 2*stride, ...``).

    Consecutive selected rows have origins >= ``stride`` apart, so their windows are disjoint.
    Returns (peak_true, peak_pred, argmax_true, argmax_pred).
    """
    a, b = _arr(y_true, y_pred)
    a, b = a[::stride], b[::stride]
    return a.max(axis=1), b.max(axis=1), a.argmax(axis=1), b.argmax(axis=1)


def peak_demand_metrics(y_true, y_pred, stride: int, unit: str) -> Dict[str, Any]:
    """Peak-demand error; only meaningful for demand in kW."""
    if unit != "kW":
        return {"available": False, "reason": f"Peak-demand error is only reported for demand_kw (unit is {unit})."}
    a = np.asarray(y_true, dtype=float)
    if a.ndim != 2 or a.shape[0] == 0:
        return {"available": False, "reason": "No test windows."}
    pt, pp, it, ip = peak_demand_arrays(a, y_pred, stride)
    err = pp - pt
    rel = np.where(pt > _EPS, np.abs(err) / np.maximum(pt, _EPS) * 100.0, np.nan)
    return {
        "available": True,
        "n_windows": int(len(pt)),
        "mean_abs_peak_error_kw": float(np.mean(np.abs(err))),
        "mean_signed_peak_error_kw": float(np.mean(err)),
        "mean_abs_peak_error_percent": None if np.isnan(rel).all() else float(np.nanmean(rel)),
        "mean_abs_peak_timing_error_steps": float(np.mean(np.abs(ip - it))),
    }
