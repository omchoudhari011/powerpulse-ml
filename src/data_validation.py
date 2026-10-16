"""Dataset validation and leakage-safe cleaning.

Run from the command line::

    python -m src.data_validation --csv data\\my_building.csv
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .config import (EXOG_COLUMNS, KNOWN_COLUMNS, TARGET_COLUMNS, TARGET_UNITS, TIMESTAMP_COLUMN,
                     Config, DataConfig, add_config_arguments, config_from_args)
from .errors import DatasetError, InsufficientDataError, PowerPulseError

_MISSING_NOTES = {
    "demand_kw": "no electrical-demand column; the target falls back to energy_kwh (kWh) if available",
    "energy_kwh": "no energy-consumption column (only needed if demand_kw is also absent)",
    "temperature": "weather feature not used (optional)",
    "humidity": "weather feature not used (optional)",
    "occupancy": "occupancy feature not used (optional)",
    "equipment_status": "equipment-status feature not used (optional)",
}


@dataclass
class ValidationReport:
    n_rows: int = 0
    start: Optional[str] = None
    end: Optional[str] = None
    columns_in_file: List[str] = field(default_factory=list)
    columns_present: List[str] = field(default_factory=list)
    columns_missing: Dict[str, str] = field(default_factory=dict)
    columns_ignored: List[str] = field(default_factory=list)
    target: Optional[str] = None
    target_unit: Optional[str] = None
    interval_seconds: Optional[int] = None
    is_sorted: bool = True
    unparseable_timestamps: int = 0
    n_duplicate_rows: int = 0
    n_duplicate_timestamps: int = 0
    n_irregular_intervals: int = 0
    n_gaps: int = 0
    largest_gap_seconds: float = 0.0
    n_missing_grid_rows: int = 0
    missing_values: Dict[str, int] = field(default_factory=dict)
    invalid_values: Dict[str, int] = field(default_factory=dict)
    non_numeric_values: Dict[str, int] = field(default_factory=dict)
    expected_grid_rows: int = 0
    required_rows: int = 0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ok"] = self.ok
        return d

    def raise_if_errors(self) -> None:
        if self.errors:
            exc = InsufficientDataError if any(e.startswith("Insufficient") for e in self.errors) else DatasetError
            raise exc("Dataset validation failed:\n  - " + "\n  - ".join(self.errors))

    def summary(self) -> str:
        lines = [
            f"Rows: {self.n_rows}   Period: {self.start} -> {self.end}",
            f"Sampling interval (median): {self.interval_seconds} s",
            f"Columns present : {self.columns_present}",
            f"Columns missing : {self.columns_missing or 'none'}",
            f"Columns ignored : {self.columns_ignored or 'none'}",
            f"Target          : {self.target} ({self.target_unit})",
            f"Duplicate timestamps: {self.n_duplicate_timestamps} ({self.n_duplicate_rows} rows)",
            f"Irregular intervals: {self.n_irregular_intervals}; gaps > 1.5x interval: {self.n_gaps} "
            f"(largest {self.largest_gap_seconds:.0f} s); missing grid rows: {self.n_missing_grid_rows}",
            f"Missing values  : {self.missing_values}",
            f"Invalid readings: {self.invalid_values}",
            f"Non-numeric     : {self.non_numeric_values or 'none'}",
            f"Rows required (lower bound) for configured splits: {self.required_rows}",
        ]
        lines += [f"WARNING: {w}" for w in self.warnings]
        lines += [f"ERROR:   {e}" for e in self.errors]
        lines.append("RESULT: " + ("PASSED" if self.ok else "FAILED"))
        return "\n".join(lines)


def invalid_masks(df: pd.DataFrame, data_cfg: DataConfig) -> Dict[str, pd.Series]:
    """Boolean masks of physically invalid readings (inf, negative energy/demand, ...)."""
    masks: Dict[str, pd.Series] = {}
    lo, hi = data_cfg.temperature_range
    for col in TARGET_COLUMNS + EXOG_COLUMNS:
        if col not in df.columns:
            continue
        s = df[col].astype(float)
        m = pd.Series(np.isinf(s.to_numpy()), index=s.index)
        if col in ("demand_kw", "energy_kwh", "occupancy", "equipment_status"):
            m |= s < 0
        elif col == "humidity":
            m |= (s < 0) | (s > 100)
        elif col == "temperature":
            m |= (s < lo) | (s > hi)
        masks[col] = m
    return masks


def validate_dataframe(df: pd.DataFrame, load_info: Dict[str, Any], cfg: Config,
                       interval_seconds: Optional[float] = None, check_sufficiency: bool = True,
                       required_columns: Optional[Sequence[str]] = None) -> ValidationReport:
    from .feature_engineering import required_rows

    rep = ValidationReport()
    ts_col = cfg.data.timestamp_column
    rep.n_rows = int(len(df))
    rep.columns_in_file = list(load_info.get("columns_in_file", df.columns))
    rep.columns_ignored = list(load_info.get("columns_ignored", []))
    rep.unparseable_timestamps = int(load_info.get("unparseable_timestamps", 0))
    rep.non_numeric_values = dict(load_info.get("non_numeric_values", {}))
    if load_info.get("timestamp_note"):
        rep.warnings.append(load_info["timestamp_note"])
    if rep.unparseable_timestamps:
        rep.warnings.append(f"{rep.unparseable_timestamps} row(s) had unparseable timestamps and were dropped.")
    if df.empty:
        rep.errors.append("No rows with valid timestamps.")
        return rep

    # ---- columns
    present = [c for c in KNOWN_COLUMNS if c != TIMESTAMP_COLUMN and c in df.columns and df[c].notna().any()]
    rep.columns_present = present
    for c in KNOWN_COLUMNS:
        if c == TIMESTAMP_COLUMN or c in present:
            continue
        why = "column is entirely empty" if c in df.columns else "column not in file"
        rep.columns_missing[c] = f"{why}: {_MISSING_NOTES[c]}"
    if rep.columns_ignored:
        rep.warnings.append(f"Unrecognised columns ignored: {rep.columns_ignored}")

    # ---- target selection
    requested = cfg.data.target
    if requested:
        if requested not in present:
            rep.errors.append(f"Requested target '{requested}' is missing or empty. Present: {present}")
        else:
            rep.target = requested
    else:
        rep.target = next((c for c in TARGET_COLUMNS if c in present), None)
        if rep.target is None:
            rep.errors.append(f"No forecasting target found: need 'demand_kw' or 'energy_kwh'. Present: {present}")
    if rep.target:
        rep.target_unit = TARGET_UNITS[rep.target]
        if rep.target == "energy_kwh":
            rep.warnings.append("Target is energy_kwh (energy consumed per interval, kWh), not demand_kw (kW). "
                                "Forecasts are labelled kWh and must not be read as instantaneous power.")
    if required_columns:
        absent = [c for c in required_columns if c not in present]
        if absent:
            rep.errors.append(f"Columns required by the trained model are missing/empty in this CSV: {absent}")

    # ---- ordering, duplicates, interval
    ts = df[ts_col]
    rep.start, rep.end = str(ts.min()), str(ts.max())
    rep.is_sorted = bool(ts.is_monotonic_increasing)
    if not rep.is_sorted:
        rep.warnings.append("Rows were not in chronological order; they will be sorted.")
    dup = ts.duplicated(keep=False)
    rep.n_duplicate_rows = int(dup.sum())
    rep.n_duplicate_timestamps = int(ts[dup].nunique())
    if rep.n_duplicate_timestamps:
        rep.warnings.append(f"{rep.n_duplicate_timestamps} duplicate timestamp(s); resolved with policy "
                            f"'{cfg.data.duplicate_policy}'.")
    uniq = ts.drop_duplicates().sort_values()
    if len(uniq) < 3:
        rep.errors.append("Fewer than 3 distinct timestamps; cannot determine the sampling interval.")
        return rep
    diffs = uniq.diff().dt.total_seconds().dropna().to_numpy()
    median = float(np.median(diffs))
    if median <= 0:
        rep.errors.append("Could not determine a positive sampling interval.")
        return rep
    interval = float(interval_seconds) if interval_seconds else median
    rep.interval_seconds = max(1, int(round(interval)))
    rep.n_irregular_intervals = int((np.abs(diffs - interval) > max(1.0, 0.01 * interval)).sum())
    rep.n_gaps = int((diffs > 1.5 * interval).sum())
    rep.largest_gap_seconds = float(diffs.max())
    span = (uniq.iloc[-1] - uniq.iloc[0]).total_seconds()
    rep.expected_grid_rows = int(round(span / rep.interval_seconds)) + 1
    rep.n_missing_grid_rows = max(0, rep.expected_grid_rows - len(uniq))
    if rep.expected_grid_rows > 50_000_000:
        rep.errors.append(f"Time span / interval implies {rep.expected_grid_rows} rows; dataset looks malformed.")
    if rep.n_irregular_intervals:
        rep.warnings.append(f"{rep.n_irregular_intervals} irregular interval(s) (median {rep.interval_seconds} s); "
                            f"timestamps are snapped to a regular grid and absent rows become missing values.")
    if rep.n_gaps:
        rep.warnings.append(f"{rep.n_gaps} gap(s) longer than 1.5x the interval (largest "
                            f"{rep.largest_gap_seconds / 3600:.1f} h). Gaps up to {cfg.data.max_gap_hours} h are "
                            f"forward-filled for *inputs* only; training/evaluation targets are never imputed.")
    if interval > 3600:
        rep.warnings.append(f"Sampling interval ({rep.interval_seconds} s) is longer than 1 h: input_hours, "
                            f"horizon_hours and lag_hours are interpreted as numbers of steps.")

    # ---- missing / invalid readings
    masks = invalid_masks(df, cfg.data)
    for c in present:
        rep.missing_values[c] = int(df[c].isna().sum())
        n_bad = int(masks[c].sum())
        if n_bad:
            rep.invalid_values[c] = n_bad
    if rep.invalid_values:
        rep.warnings.append(f"Invalid readings set to missing: {rep.invalid_values}")
    if any(rep.missing_values.values()):
        rep.warnings.append("Missing values are only forward-filled (never back-filled or interpolated) so no "
                            "future information leaks into earlier rows.")

    if rep.target:
        valid_target = df[rep.target].notna() & ~masks[rep.target]
        if valid_target.sum() == 0:
            rep.errors.append(f"Target '{rep.target}' has no valid readings.")
        elif df.loc[valid_target, rep.target].nunique() <= 1:
            rep.errors.append(f"Target '{rep.target}' is constant; nothing to forecast.")

    # ---- sufficiency (lower bound; the exact check happens when samples are built)
    if check_sufficiency and rep.interval_seconds:
        rep.required_rows = required_rows(cfg, rep.interval_seconds)
        if rep.expected_grid_rows < rep.required_rows:
            rep.errors.append(
                f"Insufficient data: the dataset spans {rep.expected_grid_rows} time steps but at least "
                f"{rep.required_rows} are needed for the configured input window, horizon and "
                f"train/validation/test sizes. Provide a longer history or shorten input/horizon.")
    return rep


class CleanedData:
    """Regular-grid frame + mask of genuinely observed target readings."""

    def __init__(self, frame: pd.DataFrame, target_observed: pd.Series, stats: Dict[str, Any]):
        self.frame = frame
        self.target_observed = target_observed
        self.stats = stats


def clean_dataframe(df: pd.DataFrame, interval_seconds: float, cfg: Config, target: str,
                    exog_columns: Sequence[str]) -> CleanedData:
    """Invalid -> NaN, dedupe, snap to a regular grid, causal forward-fill (limited).

    Only past values are ever used to fill a gap (no back-fill, no interpolation, no rolling
    statistic over the future), so cleaning cannot leak future data into earlier rows.
    """
    ts_col = cfg.data.timestamp_column
    interval = max(1, int(round(interval_seconds)))
    cols = [target] + [c for c in exog_columns if c != target]
    d = df[[ts_col] + cols].copy()

    n_invalid = 0
    for col, mask in invalid_masks(d, cfg.data).items():
        n_invalid += int(mask.sum())
        d.loc[mask, col] = np.nan

    freq = f"{interval}s"
    snapped = d[ts_col].dt.round(freq)
    n_off_grid = int((snapped != d[ts_col]).sum())
    d[ts_col] = snapped
    grouped = d.groupby(ts_col, sort=True)
    policy = cfg.data.duplicate_policy
    d = grouped.mean() if policy == "mean" else (grouped.first() if policy == "first" else grouped.last())

    full_index = pd.date_range(d.index.min(), d.index.max(), freq=freq)
    if len(full_index) > 50_000_000:
        raise DatasetError("Regular time grid would exceed 50M rows; dataset looks malformed.")
    d = d.reindex(full_index)
    d.index.name = "timestamp"

    observed = d[target].notna()
    max_gap_steps = max(1, int(round(cfg.data.max_gap_hours * 3600.0 / interval)))
    filled = d.ffill(limit=max_gap_steps)
    stats = {
        "grid_rows": int(len(filled)),
        "invalid_readings_set_to_nan": n_invalid,
        "off_grid_timestamps_snapped": n_off_grid,
        "target_missing_before_fill": int((~observed).sum()),
        "target_imputed_by_ffill": int((observed.eq(False) & filled[target].notna()).sum()),
        "target_still_missing": int(filled[target].isna().sum()),
        "max_ffill_steps": max_gap_steps,
    }
    return CleanedData(filled, observed, stats)


def main(argv=None) -> int:
    from .data_loader import load_csv

    parser = argparse.ArgumentParser(description="Validate a building-energy CSV for PowerPulse forecasting.")
    parser.add_argument("--csv", required=True, help="Path to the dataset CSV.")
    parser.add_argument("--report-json", help="Optionally save the validation report as JSON.")
    add_config_arguments(parser)
    args = parser.parse_args(argv)
    try:
        cfg = config_from_args(args)
        df, info = load_csv(args.csv, cfg.data)
        rep = validate_dataframe(df, info, cfg)
    except (PowerPulseError, FileNotFoundError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(rep.summary())
    if args.report_json:
        Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report_json).write_text(json.dumps(rep.to_dict(), indent=2, default=str), encoding="utf-8")
    return 0 if rep.ok else 1


if __name__ == "__main__":
    sys.exit(main())
