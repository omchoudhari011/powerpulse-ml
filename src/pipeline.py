"""Orchestrates load -> validate -> clean -> split -> scale -> windows, identically for both models."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import EXOG_COLUMNS, TARGET_UNITS, Config
from .data_loader import load_csv, sha256_of_file
from .data_validation import CleanedData, clean_dataframe, validate_dataframe
from .errors import InsufficientDataError
from .feature_engineering import (assign_splits, build_feature_frame, make_sequences, make_tabular,
                                  make_targets, resolve_steps, required_rows, tabular_feature_names,
                                  valid_origins)
from .preprocessing import Preprocessor, chronological_split


@dataclass
class SplitData:
    name: str
    origins: np.ndarray
    origin_times: np.ndarray        # (N,)   datetime64
    target_times: np.ndarray        # (N, H) datetime64
    X_seq: np.ndarray               # (N, L, F)  LSTM input
    X_tab: np.ndarray               # (N, D)     MLP input
    Y_scaled: np.ndarray            # (N, H)
    y_true: np.ndarray              # (N, H) original units

    def __len__(self) -> int:
        return int(len(self.origins))


@dataclass
class PreparedData:
    splits: Dict[str, SplitData]
    preprocessor: Preprocessor
    steps: Dict[str, Any]
    target: str
    unit: str
    seq_feature_names: List[str]
    tab_feature_names: List[str]
    boundaries: Dict[str, Any]
    dropped_lags: List[int] = field(default_factory=list)
    report: Dict[str, Any] = field(default_factory=dict)
    clean_stats: Dict[str, Any] = field(default_factory=dict)
    csv_path: str = ""
    csv_sha256: str = ""
    data_signature: str = ""
    warnings: List[str] = field(default_factory=list)


def data_signature(cfg: Config, csv_sha256: str, target: str) -> str:
    """Fingerprint of everything that defines the samples/splits. Two models are comparable
    only if their signatures match."""
    payload = {"csv": csv_sha256, "target": target, "data": cfg.data.__dict__,
               "features": cfg.features.__dict__, "input_hours": cfg.train.input_hours,
               "horizon_hours": cfg.train.horizon_hours}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def prepare_dataset(clean: CleanedData, cfg: Config, interval_seconds: float, target: str,
                    preprocessor: Optional[Preprocessor] = None,
                    lag_steps: Optional[Sequence[int]] = None) -> PreparedData:
    """Build chronological train/validation/test samples for BOTH models from one cleaned frame."""
    d = cfg.data
    frame = clean.frame
    n = len(frame)
    steps = resolve_steps(cfg, interval_seconds)
    L, H = steps["input_steps"], steps["horizon_steps"]
    warnings: List[str] = []

    need = required_rows(cfg, interval_seconds)
    if n < need:
        raise InsufficientDataError(
            f"Insufficient data: {n} regular time steps available, at least {need} required "
            f"(input={L} steps, horizon={H} steps, min samples train/val/test="
            f"{d.min_train_samples}/{d.min_val_samples}/{d.min_test_samples}).")
    a, b = chronological_split(n, d.train_frac, d.val_frac)

    # lags that the training period can support (longest lag must leave enough train samples)
    dropped: List[int] = []
    if lag_steps is None:
        max_lag_allowed = a - H - d.min_train_samples + 1
        lag_steps = [k for k in steps["lag_steps"] if k <= max_lag_allowed]
        dropped = [k for k in steps["lag_steps"] if k > max_lag_allowed]
        if dropped:
            warnings.append(f"Lag(s) {dropped} (steps) dropped: not enough history to use them while keeping "
                            f"{d.min_train_samples} training samples.")
        if L > max_lag_allowed or not lag_steps:
            raise InsufficientDataError(
                f"Insufficient data: input window of {L} steps leaves fewer than {d.min_train_samples} "
                f"training samples ({a} train rows, horizon {H}).")
    lag_steps = sorted(int(k) for k in lag_steps)
    steps = {**steps, "lag_steps": lag_steps}

    exog = [c for c in EXOG_COLUMNS if c in frame.columns]
    if preprocessor is None:
        pre = Preprocessor(target, exog).fit(frame.iloc[:a])      # TRAIN rows only
    else:
        pre = preprocessor
    feat = build_feature_frame(frame, pre, cfg.features.use_calendar)
    seq_names = list(feat.columns)
    tab_names = tabular_feature_names(lag_steps, seq_names)
    values = feat.to_numpy(dtype=float)

    origins = valid_origins(values, clean.target_observed.to_numpy(), L, H, lag_steps)
    by_split = assign_splits(origins, H, a, b, n)
    raw_target = frame[target].to_numpy(dtype=float)
    times = frame.index.to_numpy()

    splits: Dict[str, SplitData] = {}
    for name, o in by_split.items():
        splits[name] = SplitData(
            name=name, origins=o, origin_times=times[o],
            target_times=times[o[:, None] + np.arange(1, H + 1)[None, :]] if len(o) else np.empty((0, H), dtype=times.dtype),
            X_seq=make_sequences(values, o, L).astype(np.float32) if len(o) else np.empty((0, L, values.shape[1]), np.float32),
            X_tab=make_tabular(values, o, lag_steps).astype(np.float32) if len(o) else np.empty((0, len(tab_names)), np.float32),
            Y_scaled=make_targets(values[:, 0], o, H).astype(np.float32) if len(o) else np.empty((0, H), np.float32),
            y_true=make_targets(raw_target, o, H) if len(o) else np.empty((0, H)),
        )

    minimum = {"train": d.min_train_samples, "validation": d.min_val_samples, "test": d.min_test_samples}
    short = {k: (len(splits[k]), v) for k, v in minimum.items() if len(splits[k]) < v}
    if short:
        detail = ", ".join(f"{k}: {have} usable samples (need {want})" for k, (have, want) in short.items())
        raise InsufficientDataError(
            f"Insufficient usable samples after excluding windows with missing/imputed targets or inputs -> {detail}. "
            f"Provide more complete data or relax min_*_samples / input_hours / horizon_hours.")

    idx = frame.index
    boundaries = {
        "interval_seconds": int(round(interval_seconds)),
        "n_rows": int(n),
        "train": {"rows": int(a), "start": str(idx[0]), "end": str(idx[a - 1]), "n_samples": len(splits["train"])},
        "validation": {"rows": int(b - a), "start": str(idx[a]), "end": str(idx[b - 1]), "n_samples": len(splits["validation"])},
        "test": {"rows": int(n - b), "start": str(idx[b]), "end": str(idx[-1]), "n_samples": len(splits["test"])},
    }
    return PreparedData(splits=splits, preprocessor=pre, steps=steps, target=target, unit=TARGET_UNITS[target],
                        seq_feature_names=seq_names, tab_feature_names=tab_names, boundaries=boundaries,
                        dropped_lags=dropped, clean_stats=dict(clean.stats), warnings=warnings)


def load_and_prepare(csv_path, cfg: Config, interval_seconds: Optional[float] = None,
                     preprocessor: Optional[Preprocessor] = None, lag_steps: Optional[Sequence[int]] = None,
                     target: Optional[str] = None) -> PreparedData:
    """CSV -> PreparedData. Raises clear errors for missing file / invalid data / too little data."""
    cfg.validate()
    if target:
        cfg.data.target = target
    df, info = load_csv(csv_path, cfg.data)
    report = validate_dataframe(df, info, cfg, interval_seconds=interval_seconds)
    report.raise_if_errors()
    interval = float(interval_seconds or report.interval_seconds)
    exog = [c for c in EXOG_COLUMNS if c in report.columns_present]
    clean = clean_dataframe(df, interval, cfg, report.target, exog)
    prepared = prepare_dataset(clean, cfg, interval, report.target, preprocessor=preprocessor, lag_steps=lag_steps)
    prepared.report = report.to_dict()
    prepared.warnings = list(report.warnings) + prepared.warnings
    prepared.csv_path = str(csv_path)
    prepared.csv_sha256 = sha256_of_file(csv_path)
    prepared.data_signature = data_signature(cfg, prepared.csv_sha256, report.target)
    return prepared
