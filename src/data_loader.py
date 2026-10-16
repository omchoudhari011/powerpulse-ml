"""Load a real building-energy CSV. No data is ever generated here."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .config import EXOG_COLUMNS, KNOWN_COLUMNS, TARGET_COLUMNS, DataConfig
from .errors import DatasetError

_STATUS_MAP = {
    "on": 1.0, "off": 0.0, "true": 1.0, "false": 0.0, "yes": 1.0, "no": 0.0,
    "running": 1.0, "stopped": 0.0, "active": 1.0, "inactive": 0.0, "1": 1.0, "0": 0.0,
}


def sha256_of_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _status_value(v) -> float:
    if v is None or pd.isna(v):
        return np.nan
    if isinstance(v, (bool, np.bool_, int, float, np.integer, np.floating)):
        return float(v)
    t = str(v).strip().lower()
    if t in _STATUS_MAP:
        return _STATUS_MAP[t]
    try:
        return float(t)
    except ValueError:
        return np.nan


def _parse_timestamps(series: pd.Series, fmt: Optional[str]) -> Tuple[pd.Series, str]:
    note = ""
    parsed = pd.to_datetime(series, errors="coerce", format=fmt)
    if not pd.api.types.is_datetime64_any_dtype(parsed):  # mixed UTC offsets -> object dtype
        parsed = pd.to_datetime(series, errors="coerce", format=fmt, utc=True)
    if getattr(parsed.dt, "tz", None) is not None:
        note = f"Timestamps were timezone-aware ({parsed.dt.tz}); converted to naive wall-clock time."
        parsed = parsed.dt.tz_localize(None)
    return parsed, note


def load_csv(path, data_cfg: Optional[DataConfig] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """Read ``path`` and return ``(frame, info)``.

    ``frame`` has a parsed ``timestamp`` column plus whichever known columns exist, coerced to
    numeric. Nothing is cleaned or reordered here; see ``data_validation`` for that.
    """
    cfg = data_cfg or DataConfig()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Dataset not found: {p}\nPlace your real building-energy CSV in the data/ folder "
            f"(see data/README.md for the expected columns) and pass its path with --csv."
        )
    if not p.is_file():
        raise DatasetError(f"Dataset path is not a file: {p}")
    try:
        raw = pd.read_csv(p)
    except pd.errors.EmptyDataError as exc:
        raise DatasetError(f"Dataset file is empty: {p}") from exc
    except (UnicodeDecodeError, pd.errors.ParserError) as exc:
        raise DatasetError(f"Could not parse {p} as CSV: {exc}") from exc

    raw.columns = [str(c).strip().lower().replace(" ", "_") for c in raw.columns]
    ts_col = cfg.timestamp_column
    info: Dict[str, Any] = {
        "rows_read": int(len(raw)),
        "columns_in_file": list(raw.columns),
        "columns_ignored": [c for c in raw.columns if c not in KNOWN_COLUMNS],
        "timestamp_note": "",
        "unparseable_timestamps": 0,
        "non_numeric_values": {},
    }
    if raw.empty:
        raise DatasetError(f"Dataset has a header but no rows: {p}")
    if ts_col not in raw.columns:
        raise DatasetError(
            f"Required column '{ts_col}' not found. Columns in file: {list(raw.columns)}"
        )

    parsed, note = _parse_timestamps(raw[ts_col], cfg.timestamp_format)
    info["timestamp_note"] = note
    bad_ts = parsed.isna()
    info["unparseable_timestamps"] = int(bad_ts.sum())
    if bad_ts.all():
        raise DatasetError(
            f"No value in column '{ts_col}' could be parsed as a timestamp. "
            f"Example values: {raw[ts_col].head(3).tolist()}. Use timestamp_format in the config if needed."
        )

    frame = pd.DataFrame({ts_col: parsed})
    for col in TARGET_COLUMNS + EXOG_COLUMNS:
        if col not in raw.columns:
            continue
        if col == "equipment_status":
            values = raw[col].astype(object).map(_status_value).astype(float)
        else:
            values = pd.to_numeric(raw[col], errors="coerce").astype(float)
        introduced = int((raw[col].notna() & values.isna()).sum())
        if introduced:
            info["non_numeric_values"][col] = introduced
        frame[col] = values
    frame = frame.loc[~bad_ts].reset_index(drop=True)
    return frame, info
