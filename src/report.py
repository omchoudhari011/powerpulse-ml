"""Forecast report: one structured, documented format for every model (lstm, mlp, ensemble).

``build_forecast_report`` turns a forecast table into a JSON-ready dict (schema below);
``render_report`` turns that dict into Markdown, which is printed by src.predict, saved next to the
JSON, and handed to the question-answering assistant (src/rag).

Report schema (``schema_version`` 1)::

    report_type, schema_version, generated_at, target_variable, unit
    model:      name, version, components [{name, version, weight}]
    forecast:   origin, start, end, steps, interval_minutes, kind ("future" | "backtest")
    summary:    peak {value, timestamp}, low {value, timestamp}, mean, total_energy_kwh
    prediction_interval: level_percent, method, test_coverage_percent      (null if not available)
    accuracy:   source, test_period, validation {mae, rmse, wape_percent}, test {...},
                test_peak_demand {...}                                      (null if not evaluated)
    backtest:   n_actual, mae, rmse                                         (null for a future forecast)
    data:       training_csv, synthetic
    values:     [{timestamp, predicted, lower, upper, lstm, mlp, actual}]   (missing values are null)
    caveats:    [str]
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .artifacts import to_jsonable

REPORT_TYPE = "powerpulse_forecast"
SCHEMA_VERSION = 1
DUMMY_DATASET = "dummy_building.csv"


def _num(v, digits: int = 3) -> Optional[float]:
    return None if v is None or pd.isna(v) else round(float(v), digits)


def _ts(t) -> str:
    return pd.Timestamp(t).strftime("%Y-%m-%d %H:%M")


def accuracy_from_evaluation(evaluation: Optional[Dict[str, Any]], model_name: str,
                             model_version: str) -> Optional[Dict[str, Any]]:
    """Accuracy block for this exact model version, or None when the report is missing or stale."""
    if not evaluation:
        return None
    m = evaluation.get("models", {}).get(model_name)
    if not m or m.get("model_version") != model_version:
        return None
    test = evaluation.get("boundaries", {}).get("test", {})
    keep = ("mae", "rmse", "wape_percent")
    return {"source": "outputs/evaluation_metrics.json",
            "test_period": {"start": test.get("start"), "end": test.get("end")},
            "validation": {k: _num(m["validation"].get(k)) for k in keep},
            "test": {k: _num(m["test"].get(k)) for k in keep},
            "test_peak_demand": m.get("test_peak_demand")}


def build_forecast_report(df: pd.DataFrame, model_name: str, model_version: str, origin, interval_seconds: float,
                          components: Optional[List[Dict[str, Any]]] = None,
                          interval_info: Optional[Dict[str, Any]] = None,
                          evaluation: Optional[Dict[str, Any]] = None, training_csv: str = "") -> Dict[str, Any]:
    unit, target = df["unit"].iloc[0], df["target_variable"].iloc[0]
    pred = df["predicted_value"].to_numpy(dtype=float)
    actual = df["actual_value"].to_numpy(dtype=float)
    hi, lo = int(np.argmax(pred)), int(np.argmin(pred))
    hours = interval_seconds / 3600.0
    total_kwh = pred.sum() * hours if unit == "kW" else pred.sum() if unit == "kWh" else None
    observed = ~np.isnan(actual)

    def col(name):
        return df[name].to_numpy(dtype=float) if name in df.columns else np.full(len(df), np.nan)

    lower, upper, lstm, mlp = col("lower_value"), col("upper_value"), col("lstm_predicted_value"), col("mlp_predicted_value")
    values = [{"timestamp": _ts(t), "predicted": _num(pred[i]), "lower": _num(lower[i]), "upper": _num(upper[i]),
               "lstm": _num(lstm[i]), "mlp": _num(mlp[i]), "actual": _num(actual[i])}
              for i, t in enumerate(df["timestamp"])]

    synthetic = Path(training_csv.replace("\\", "/")).name == DUMMY_DATASET if training_csv else False
    caveats = ["Forecasts are estimates; use the accuracy figures and the range (if given) to judge how far to trust them."]
    if synthetic:
        caveats.append("The models were trained on the bundled SYNTHETIC demo dataset; results do not describe a real building.")
    accuracy = accuracy_from_evaluation(evaluation, model_name, model_version)
    if evaluation is not None and accuracy is None:
        caveats.append("outputs/evaluation_metrics.json does not match this model version; rerun python -m src.evaluate.")
    if evaluation and evaluation.get("caveat"):
        caveats.append(evaluation["caveat"])

    return to_jsonable({
        "report_type": REPORT_TYPE, "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target_variable": target, "unit": unit,
        "model": {"name": model_name, "version": model_version,
                  "components": components or [{"name": model_name, "version": model_version, "weight": 1.0}]},
        "forecast": {"origin": _ts(origin), "start": values[0]["timestamp"], "end": values[-1]["timestamp"],
                     "steps": len(df), "interval_minutes": _num(interval_seconds / 60, 1),
                     "kind": "backtest" if observed.any() else "future"},
        "summary": {"peak": {"value": _num(pred[hi]), "timestamp": values[hi]["timestamp"]},
                    "low": {"value": _num(pred[lo]), "timestamp": values[lo]["timestamp"]},
                    "mean": _num(pred.mean()), "total_energy_kwh": _num(total_kwh, 1)},
        "prediction_interval": interval_info,
        "accuracy": accuracy,
        "backtest": None if not observed.any() else {
            "n_actual": int(observed.sum()),
            "mae": _num(np.mean(np.abs(actual[observed] - pred[observed]))),
            "rmse": _num(np.sqrt(np.mean((actual[observed] - pred[observed]) ** 2)))},
        "data": {"training_csv": training_csv or None, "synthetic": synthetic},
        "values": values,
        "caveats": caveats,
    })


def _f(v, digits: int = 1) -> str:
    return "-" if v is None else f"{v:.{digits}f}"


def render_report(r: Dict[str, Any]) -> str:
    """Markdown rendering of a forecast report (also readable as plain text in a terminal)."""
    u, m, fc, s = r["unit"], r["model"], r["forecast"], r["summary"]
    blend = " + ".join(f"{c['weight']:.2f} x {c['name'].upper()}" for c in m["components"])
    lines = [f"# PowerPulse forecast - {r['target_variable']} ({u})", "",
             f"- **Model:** {m['name'].upper()} `{m['version']}`" + (f" = {blend}" if len(m["components"]) > 1 else ""),
             f"- **Period:** {fc['start']} -> {fc['end']} ({fc['steps']} steps of {_f(fc['interval_minutes'], 0)} min, "
             f"{fc['kind']} forecast from origin {fc['origin']})",
             f"- **Generated:** {r['generated_at']}", "", "## Summary", "", "| Item | Value |", "|---|---|",
             f"| Peak | {_f(s['peak']['value'])} {u} at {s['peak']['timestamp']} |",
             f"| Lowest | {_f(s['low']['value'])} {u} at {s['low']['timestamp']} |",
             f"| Mean | {_f(s['mean'])} {u} |"]
    if s.get("total_energy_kwh") is not None:
        lines.append(f"| Energy over the period | {_f(s['total_energy_kwh'], 0)} kWh |")
    pi = r.get("prediction_interval")
    if pi:
        cov = pi.get("test_coverage_percent")
        lines.append(f"| Forecast range | {pi['level_percent']}% range"
                     + (f", covered {cov:.1f}% of actuals on the test period" if cov is not None else "") + " |")
    if r.get("backtest"):
        b = r["backtest"]
        lines.append(f"| Backtest error | MAE {_f(b['mae'], 2)} {u}, RMSE {_f(b['rmse'], 2)} {u} ({b['n_actual']} actuals) |")

    a = r.get("accuracy")
    if a:
        lines += ["", f"## Accuracy of this model ({a['test_period']['start']} -> {a['test_period']['end']} test period)", "",
                  "| Split | MAE | RMSE | WAPE |", "|---|---|---|---|"]
        for split in ("validation", "test"):
            v = a[split]
            lines.append(f"| {split} | {_f(v['mae'], 2)} {u} | {_f(v['rmse'], 2)} {u} | {_f(v['wape_percent'], 2)}% |")
        peak = a.get("test_peak_demand") or {}
        if peak.get("available"):
            lines.append(f"\nDaily peak on test: off by {_f(peak['mean_abs_peak_error_kw'], 2)} kW on average "
                         f"({_f(peak['mean_abs_peak_error_percent'])}%), timing off by "
                         f"{_f(peak['mean_abs_peak_timing_error_steps'])} steps.")

    vals = r["values"]
    cols = [("Time", "timestamp", None), ("Forecast", "predicted", 1)]
    if any(v["lower"] is not None for v in vals):
        cols.append(("Range", "range", 1))
    for name in ("lstm", "mlp"):
        if any(v[name] is not None for v in vals) and len(m["components"]) > 1:
            cols.append((name.upper(), name, 1))
    if any(v["actual"] is not None for v in vals):
        cols.append(("Actual", "actual", 1))
    lines += ["", f"## Forecast values ({u})", "", "| " + " | ".join(c[0] for c in cols) + " |",
              "|" + "---|" * len(cols)]
    for v in vals:
        cells = []
        for _, key, digits in cols:
            if key == "timestamp":
                cells.append(v[key])
            elif key == "range":
                cells.append(f"{_f(v['lower'])} - {_f(v['upper'])}")
            else:
                cells.append(_f(v[key], digits))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "## Notes", ""] + [f"- {c}" for c in r["caveats"]]
    return "\n".join(lines) + "\n"


def save_report(report: Dict[str, Any], out_dir, stem: str) -> Dict[str, Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {"json": out_dir / f"{stem}.json", "md": out_dir / f"{stem}.md"}
    paths["json"].write_text(json.dumps(report, indent=2), encoding="utf-8")
    paths["md"].write_text(render_report(report), encoding="utf-8")
    return paths


def latest_report(output_dir) -> Optional[Path]:
    files = sorted(Path(output_dir).glob("forecast_report_*.json"), key=lambda f: f.stat().st_mtime)
    return files[-1] if files else None
