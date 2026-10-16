"""Turn the files in outputs/ into a short plain-text summary for the language model.

The summary is small and relevant to almost every question, so it is always sent in full rather than
retrieved. Every number in it is read from the files written by src.evaluate and src.predict.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

import pandas as pd

from ..config import ARTIFACTS_DIR, OUTPUTS_DIR
from ..report import DUMMY_DATASET, latest_report, render_report

MAX_TABLE_ROWS = 48


def _fmt(value, digits: int = 2) -> str:
    return "n/a" if value is None or pd.isna(value) else f"{value:.{digits}f}"


def latest_forecast_file(output_dir) -> Optional[Path]:
    files = sorted(Path(output_dir).glob("forecast_*.csv"), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None


def summarize_metrics(output_dir, artifacts_dir) -> List[str]:
    path = Path(output_dir) / "evaluation_metrics.json"
    if not path.exists():
        return ["Model evaluation: not available (run python -m src.evaluate)."]
    report = json.loads(path.read_text(encoding="utf-8"))
    unit = report.get("unit", "")
    selection = report.get("selection", {})
    best = selection.get("preferred_model")
    test = report.get("boundaries", {}).get("test", {})
    note = f"; {selection['note'].rstrip('.')}" if selection.get("note") else ""
    lines = [f"Model evaluation (report generated {report.get('generated_at', 'n/a')}):",
             f"- Target: {report.get('target_variable')} ({unit}); test period {test.get('start')} -> {test.get('end')}.",
             f"- Preferred model: {best or 'none'} (chosen by {selection.get('criterion', 'n/a')}{note})."]
    for name, m in report.get("models", {}).items():
        for split in ("validation", "test"):
            s = m.get(split, {})
            lines.append(f"- {name.upper()} {split}: MAE {_fmt(s.get('mae'))} {unit}, "
                         f"RMSE {_fmt(s.get('rmse'))} {unit}, WAPE {_fmt(s.get('wape_percent'))}%.")
        by_step = m.get("test_mae_by_horizon_step") or []
        if by_step:
            lines.append(f"- {name.upper()} test MAE grows from {_fmt(by_step[0])} {unit} one step ahead "
                         f"to {_fmt(by_step[-1])} {unit} {len(by_step)} steps ahead.")
        if m.get("weights"):
            lines.append(f"- {name.upper()} = " + " + ".join(f"{w:.2f} x {c.upper()}" for c, w in m["weights"].items())
                         + " (weights fitted on the validation period).")
        pi = m.get("prediction_interval") or {}
        if pi.get("test_coverage_percent") is not None:
            lines.append(f"- {name.upper()} {pi['level_percent']}% forecast range contained "
                         f"{_fmt(pi['test_coverage_percent'], 1)}% of actual test readings.")
        peak = m.get("test_peak_demand") or {}
        if peak.get("available"):
            lines.append(f"- {name.upper()} daily peak: mean absolute error {_fmt(peak.get('mean_abs_peak_error_kw'))} kW "
                         f"({_fmt(peak.get('mean_abs_peak_error_percent'), 1)}%), mean signed error "
                         f"{_fmt(peak.get('mean_signed_peak_error_kw'))} kW (negative = peak under-predicted), "
                         f"peak timing off by {_fmt(peak.get('mean_abs_peak_timing_error_steps'), 1)} steps on average.")
    for warning in report.get("dataset_warnings") or []:
        lines.append(f"- Dataset warning: {warning}")
    if report.get("caveat"):
        lines.append(f"- Caveat: {report['caveat']}")
    # the ensemble has no metadata.json of its own; its components were trained on the same CSV
    meta_path = next((Path(artifacts_dir) / n / "metadata.json" for n in (best, "mlp", "lstm")
                      if n and (Path(artifacts_dir) / n / "metadata.json").exists()), None)
    if meta_path is not None:
        csv_path = json.loads(meta_path.read_text(encoding="utf-8")).get("csv_path", "")
        if csv_path:
            note = (" This is the bundled SYNTHETIC demo dataset, not real building data."
                    if Path(csv_path.replace("\\", "/")).name == DUMMY_DATASET else "")
            lines.append(f"- Training data: {csv_path}.{note}")
    return lines


def summarize_forecast(output_dir) -> List[str]:
    """The latest forecast report (src/report.py) when one exists, else a summary of the latest forecast CSV."""
    report_path = latest_report(output_dir)
    if report_path is not None:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            return [f"Latest forecast report ({report_path.name}):", render_report(report)]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass  # unreadable/older report: fall back to the CSV below
    path = latest_forecast_file(output_dir)
    if path is None:
        return ["Latest forecast: not available (run python -m src.predict)."]
    df = pd.read_csv(path, parse_dates=["timestamp"])
    if df.empty:
        return [f"Latest forecast: {path.name} is empty."]
    unit, target = df["unit"].iloc[0], df["target_variable"].iloc[0]
    pred = df["predicted_value"]
    hi, lo = pred.idxmax(), pred.idxmin()
    lines = [f"Latest forecast ({path.name}, model {df['model_name'].iloc[0]} {df['model_version'].iloc[0]}, "
             f"generated {df['forecast_generated_at'].iloc[0]}):",
             f"- Covers {df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]} ({len(df)} steps of {target}).",
             f"- Peak {_fmt(pred[hi])} {unit} at {df['timestamp'][hi]}; lowest {_fmt(pred[lo])} {unit} at "
             f"{df['timestamp'][lo]}; mean {_fmt(pred.mean())} {unit}."]
    if len(df) > 1:
        hours = (df["timestamp"].iloc[1] - df["timestamp"].iloc[0]).total_seconds() / 3600
        if unit == "kW":
            lines.append(f"- Implied energy over the period: {_fmt(pred.sum() * hours, 0)} kWh.")
        elif unit == "kWh":
            lines.append(f"- Total over the period: {_fmt(pred.sum(), 0)} kWh.")
    actual = df["actual_value"]
    if actual.notna().any():
        mae = (actual - pred).abs().mean()
        lines.append(f"- This is a backtest: actual readings exist for {int(actual.notna().sum())} steps; "
                     f"MAE {_fmt(mae)} {unit}.")
    else:
        lines.append("- Future forecast: no actual readings yet.")
    lines.append(f"- Values (timestamp, predicted{', actual' if actual.notna().any() else ''}, {unit}):")
    for _, row in df.head(MAX_TABLE_ROWS).iterrows():
        extra = f", {_fmt(row['actual_value'], 1)}" if actual.notna().any() else ""
        lines.append(f"  {row['timestamp']:%Y-%m-%d %H:%M}, {_fmt(row['predicted_value'], 1)}{extra}")
    if len(df) > MAX_TABLE_ROWS:
        lines.append(f"  ... {len(df) - MAX_TABLE_ROWS} more steps not shown.")
    return lines


def build_forecast_context(output_dir=None, artifacts_dir=None) -> str:
    output_dir, artifacts_dir = Path(output_dir or OUTPUTS_DIR), Path(artifacts_dir or ARTIFACTS_DIR)
    return "\n".join(summarize_metrics(output_dir, artifacts_dir) + [""] + summarize_forecast(output_dir))
