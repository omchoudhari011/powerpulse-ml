"""Generate a forecast with a trained model or with the LSTM+MLP ensemble.

    python -m src.predict --model ensemble --csv data\\recent_history.csv
    python -m src.predict --model best --csv data\\my_building.csv --origin "2024-03-01 00:00"

By default the forecast starts after the LAST timestamp in the CSV (a genuine future forecast, so
``actual_value`` is empty). ``--origin`` backtests from an earlier timestamp when actuals exist.

Writes ``forecast_<model>_<origin>.csv/.json`` (one row per step) and
``forecast_report_<model>_<origin>.json/.md`` (summary, accuracy, range and values in the format
documented in src/report.py), and prints the report.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from .artifacts import load_artifacts
from .config import ARTIFACTS_DIR, MODEL_NAMES, OUTPUTS_DIR, Config
from .data_loader import load_csv
from .data_validation import clean_dataframe, validate_dataframe
from .ensemble import ENSEMBLE_NAME, apply_interval, combine, load_ensemble
from .errors import ArtifactError, DatasetError, PowerPulseError
from .feature_engineering import build_feature_frame, build_inference_inputs
from .inference import predict_original_units
from .report import build_forecast_report, render_report, save_report

FORECAST_COLUMNS = ["timestamp", "actual_value", "predicted_value", "target_variable", "unit",
                    "model_name", "model_version", "forecast_generated_at"]
ENSEMBLE_COLUMNS = ["lower_value", "upper_value", "lstm_predicted_value", "mlp_predicted_value"]
FORECAST_MODELS = list(MODEL_NAMES) + [ENSEMBLE_NAME]


def resolve_best_model(output_dir) -> str:
    path = Path(output_dir) / "evaluation_metrics.json"
    if not path.exists():
        raise ArtifactError(f"--model best needs {path}; run python -m src.evaluate first.")
    best = json.loads(path.read_text(encoding="utf-8")).get("selection", {}).get("preferred_model")
    if not best:
        raise ArtifactError("The evaluation report has no preferred model (see selection.note). "
                            f"Choose one of {', '.join(FORECAST_MODELS)}.")
    return best


def model_forecast(model_name: str, csv_path, artifacts_root=None, origin: Optional[str] = None) -> Dict[str, Any]:
    """Forecast of ONE trained model (lstm or mlp) from the given origin, in original units."""
    loaded = load_artifacts(model_name, artifacts_root)
    meta, cfg = loaded.metadata, Config.from_dict(loaded.config)
    target, interval = meta["target"], meta["interval_seconds"]
    H, L, lags = meta["steps"]["horizon_steps"], meta["steps"]["input_steps"], meta["steps"]["lag_steps"]
    exog = loaded.preprocessor.exog_columns

    cfg.data.target = target
    df, info = load_csv(csv_path, cfg.data)
    report = validate_dataframe(df, info, cfg, interval_seconds=interval, check_sufficiency=False,
                                required_columns=[target] + exog)
    report.raise_if_errors()
    clean = clean_dataframe(df, interval, cfg, target, exog)

    frame = clean.frame
    if origin is None:
        idx = len(frame) - 1
    else:
        try:
            idx = frame.index.get_loc(pd.Timestamp(origin))
        except KeyError as exc:
            raise DatasetError(f"--origin {origin} is not a timestamp on the dataset's regular "
                               f"{interval}s grid ({frame.index[0]} -> {frame.index[-1]}).") from exc
    values = build_feature_frame(frame, loaded.preprocessor, cfg.features.use_calendar).to_numpy(dtype=float)
    X_seq, X_tab = build_inference_inputs(values, idx, L, lags)
    X = X_seq if model_name == "lstm" else X_tab
    pred = predict_original_units(loaded.model, loaded.preprocessor, X, cfg.train.clip_non_negative)[0]

    origin_time = frame.index[idx]
    times = pd.date_range(origin_time + pd.Timedelta(seconds=interval), periods=H, freq=f"{interval}s")
    actual = np.full(H, np.nan)
    for k in range(H):                       # only genuinely observed readings count as actuals
        j = idx + 1 + k
        if j < len(frame) and clean.target_observed.iloc[j]:
            actual[k] = frame[target].iloc[j]
    return {"pred": pred, "actual": actual, "times": times, "origin_time": origin_time, "meta": meta,
            "interval": interval, "clip": cfg.train.clip_non_negative}


def _read_evaluation(output_dir) -> Optional[Dict[str, Any]]:
    path = Path(output_dir) / "evaluation_metrics.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def generate_forecast(model_name: str, csv_path, artifacts_root=None, output_dir=None,
                      origin: Optional[str] = None, save: bool = True, return_report: bool = False):
    """Forecast table (FORECAST_COLUMNS, plus ENSEMBLE_COLUMNS for the ensemble); optionally also the report."""
    out_dir = Path(output_dir or OUTPUTS_DIR)
    interval_info, components = None, None
    if model_name == ENSEMBLE_NAME:
        ens = load_ensemble(artifacts_root)
        parts = {m: model_forecast(m, csv_path, artifacts_root, origin) for m in ens["weights"]}
        for m, part in parts.items():
            if part["meta"]["model_version"] != ens["component_versions"][m]:
                raise ArtifactError(f"'{m}' was retrained after the ensemble was fitted. Rerun "
                                    "python -m src.evaluate --csv <training csv> to refit it.")
        base = next(iter(parts.values()))
        pred = combine({m: p["pred"] for m, p in parts.items()}, ens["weights"])
        lower, upper = apply_interval(pred, ens["interval_offsets"], base["clip"])
        version = ens["model_version"]
        components = [{"name": m, "version": ens["component_versions"][m], "weight": w}
                      for m, w in ens["weights"].items()]
        interval_info = {"level_percent": ens["interval_level_percent"],
                         "method": ens["prediction_interval"]["method"],
                         "test_coverage_percent": ens["prediction_interval"]["test_coverage_percent"]}
    else:
        base = model_forecast(model_name, csv_path, artifacts_root, origin)
        parts, pred, version = {model_name: base}, base["pred"], base["meta"]["model_version"]
    meta = base["meta"]

    out = pd.DataFrame({
        "timestamp": base["times"], "actual_value": base["actual"], "predicted_value": pred,
        "target_variable": meta["target"], "unit": meta["unit"], "model_name": model_name,
        "model_version": version,
        "forecast_generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    })[FORECAST_COLUMNS]
    if model_name == ENSEMBLE_NAME:
        out["lower_value"], out["upper_value"] = lower, upper
        for m in ("lstm", "mlp"):
            out[f"{m}_predicted_value"] = parts[m]["pred"]

    report = build_forecast_report(out, model_name, version, base["origin_time"], base["interval"], components,
                                   interval_info, _read_evaluation(out_dir), meta.get("csv_path", ""))
    if save:
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = f"{model_name}_{base['origin_time']:%Y%m%dT%H%M}"
        out.to_csv(out_dir / f"forecast_{tag}.csv", index=False)
        out.to_json(out_dir / f"forecast_{tag}.json", orient="records", date_format="iso", indent=2)
        save_report(report, out_dir, f"forecast_report_{tag}")
    return (out, report) if return_report else out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Generate a forecast from a trained PowerPulse model or the ensemble.")
    p.add_argument("--model", required=True, choices=FORECAST_MODELS + ["best"])
    p.add_argument("--csv", required=True, help="CSV containing recent history (>= longest lag, e.g. 7+ days).")
    p.add_argument("--origin", help="Forecast origin timestamp (default: last row of the CSV).")
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    p.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    p.add_argument("--format", choices=["report", "table", "json"], default="report",
                   help="What to print: formatted report (default), raw table, or the report as JSON.")
    args = p.parse_args(argv)
    try:
        name = resolve_best_model(args.output_dir) if args.model == "best" else args.model
        out, report = generate_forecast(name, args.csv, args.artifacts_dir, args.output_dir, args.origin,
                                        return_report=True)
    except (PowerPulseError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    if args.format == "json":
        print(json.dumps(report, indent=2))
    elif args.format == "table":
        print(out.to_string(index=False))
    else:
        print(render_report(report))
    print(f"Saved forecast CSV/JSON and report JSON/Markdown to {args.output_dir}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
