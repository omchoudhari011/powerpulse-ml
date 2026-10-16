"""Evaluate and compare the trained LSTM, MLP and their weighted ensemble on the SAME held-out test period.

    python -m src.evaluate --csv data\\my_building.csv

When both models are evaluated, the ensemble weight and its 80 % prediction interval are fitted on the
validation period and saved to artifacts/ensemble (see src/ensemble.py). The preferred model is chosen
from VALIDATION performance only; test metrics are reported, not used for selection.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd

from .artifacts import load_artifacts, to_jsonable
from .config import ARTIFACTS_DIR, MODEL_NAMES, OUTPUTS_DIR, Config
from .data_loader import sha256_of_file
from .ensemble import (COMPONENTS, ENSEMBLE_NAME, INTERVAL_LEVEL, combine, fit_weights, interval_coverage,
                       interval_offsets, save_ensemble)
from .errors import ArtifactError, PowerPulseError
from .inference import inputs_for_model, predict_original_units
from .metrics import compute_metrics, mae_by_horizon, peak_demand_arrays, peak_demand_metrics
from .pipeline import load_and_prepare


# ------------------------------------------------------------------ prediction tables
def predictions_table(split, y_pred: np.ndarray, target: str, unit: str, model_name: str,
                      model_version: str, generated_at: str) -> pd.DataFrame:
    """Long-format forecast table (one row per origin x horizon step) in the required schema."""
    n, h = y_pred.shape
    return pd.DataFrame({
        "timestamp": pd.to_datetime(split.target_times.reshape(-1)),
        "actual_value": split.y_true.reshape(-1),
        "predicted_value": y_pred.reshape(-1),
        "target_variable": target,
        "unit": unit,
        "model_name": model_name,
        "model_version": model_version,
        "forecast_generated_at": generated_at,
        "forecast_origin": np.repeat(pd.to_datetime(split.origin_times).to_numpy(), h),
        "horizon_step": np.tile(np.arange(1, h + 1), n),
    })


# ------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def plot_actual_vs_predicted(test, preds: Dict[str, np.ndarray], target: str, unit: str, out_dir: Path,
                             max_days: int = 14) -> List[Path]:
    plt = _plt()
    H = test.y_true.shape[1]
    horizons = sorted({1, H})
    paths = []
    for tag, days in ((f"first_{max_days}d", max_days), ("full_test", None)):
        fig, axes = plt.subplots(len(horizons), 1, figsize=(13, 4 * len(horizons)), squeeze=False)
        for ax, h in zip(axes[:, 0], horizons):
            t = pd.to_datetime(test.target_times[:, h - 1])
            mask = np.ones(len(t), bool) if days is None else np.asarray(t <= t[0] + pd.Timedelta(days=days))
            lw = 0.6 if days is None else 1.2
            ax.plot(t[mask], test.y_true[mask, h - 1], color="black", lw=lw, label="Actual")
            for name, p in preds.items():
                ax.plot(t[mask], p[mask, h - 1], lw=lw, alpha=0.85, label=name.upper())
            ax.set_title(f"Actual vs predicted {target} - forecast step {h} of {H}")
            ax.set_ylabel(f"{target} ({unit})")
            ax.legend(loc="upper right")
            ax.grid(alpha=0.3)
        fig.tight_layout()
        path = out_dir / f"actual_vs_predicted_{tag}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def plot_error_over_time(test, preds, unit: str, interval_seconds: int, out_dir: Path) -> Path:
    plt = _plt()
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    t1 = pd.to_datetime(test.target_times[:, 0])
    window = max(1, int(round(86400 / interval_seconds)))
    for name, p in preds.items():
        axes[0].plot(t1, p[:, 0] - test.y_true[:, 0], lw=0.6, alpha=0.8, label=name.upper())
        per_origin = pd.Series(np.abs(p - test.y_true).mean(axis=1), index=pd.to_datetime(test.origin_times))
        axes[1].plot(per_origin.rolling(window, min_periods=1).mean(), lw=1.2, label=name.upper())
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_title("Signed forecast error (predicted - actual), step 1")
    axes[0].set_ylabel(unit)
    axes[1].set_title("Mean absolute error across the whole horizon (rolling 1-day mean)")
    axes[1].set_ylabel(unit)
    for ax in axes:
        ax.legend(loc="upper right")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "forecast_error_over_time.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_mae_by_horizon(test, preds, unit: str, interval_seconds: int, out_dir: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for name, p in preds.items():
        m = mae_by_horizon(test.y_true, p)
        ax.plot((np.arange(1, len(m) + 1) * interval_seconds) / 3600.0, m, marker="o", ms=3, label=name.upper())
    ax.set_xlabel("Forecast lead time (hours)")
    ax.set_ylabel(f"MAE ({unit})")
    ax.set_title("Test MAE by forecast lead time")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "mae_by_horizon.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_peak_demand(test, preds, out_dir: Path, stride: int) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6, 6))
    hi = 0.0
    for name, p in preds.items():
        pt, pp, _, _ = peak_demand_arrays(test.y_true, p, stride)
        ax.scatter(pt, pp, s=18, alpha=0.7, label=name.upper())
        hi = max(hi, float(pt.max()), float(pp.max()))
    ax.plot([0, hi], [0, hi], color="black", lw=0.8, label="Perfect")
    ax.set_xlabel("Actual peak demand per forecast window (kW)")
    ax.set_ylabel("Predicted peak (kW)")
    ax.set_title("Peak-demand prediction (non-overlapping test windows)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    path = out_dir / "peak_demand_error.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# ------------------------------------------------------------------ main routine
def run_evaluation(csv_path, model_names: Sequence[str] = MODEL_NAMES, artifacts_root=None,
                   output_dir=None, make_plots: bool = True) -> Dict[str, Any]:
    out_dir = Path(output_dir or OUTPUTS_DIR)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    loaded = {m: load_artifacts(m, artifacts_root) for m in model_names}
    ref = loaded[model_names[0]]
    sigs = {m: loaded[m].metadata["data_signature"] for m in model_names}
    if len(set(sigs.values())) > 1:
        raise ArtifactError("The models were trained on different data/split/feature settings, so they are not "
                            "comparable. Retrain both with: python -m src.train --model both --csv <file>")
    if sha256_of_file(csv_path) != ref.metadata["csv_sha256"]:
        raise ArtifactError("This CSV differs from the file the models were trained on; the held-out test period "
                            "would not match. Use the original CSV or retrain.")

    cfg = Config.from_dict(ref.config)
    prepared = load_and_prepare(csv_path, cfg, interval_seconds=ref.metadata["interval_seconds"],
                                preprocessor=ref.preprocessor, lag_steps=ref.metadata["steps"]["lag_steps"],
                                target=ref.metadata["target"])
    for m in model_names:
        expected = loaded[m].metadata["boundaries"]["test"]["n_samples"]
        if expected != len(prepared.splits["test"]):
            raise ArtifactError(f"Rebuilt test set ({len(prepared.splits['test'])} samples) does not match the one "
                                f"seen at training time ({expected}) for '{m}'. Retrain.")

    val, test = prepared.splits["validation"], prepared.splits["test"]
    H = prepared.steps["horizon_steps"]
    clip = cfg.train.clip_non_negative
    generated_at = datetime.now(timezone.utc).isoformat(timespec="seconds")

    results: Dict[str, Any] = {}
    val_preds: Dict[str, np.ndarray] = {}
    test_preds: Dict[str, np.ndarray] = {}
    for m in model_names:
        L = loaded[m]
        pv = predict_original_units(L.model, L.preprocessor, inputs_for_model(m, val), clip)
        pt = predict_original_units(L.model, L.preprocessor, inputs_for_model(m, test), clip)
        val_preds[m], test_preds[m] = pv, pt
        results[m] = {
            "model_version": L.metadata["model_version"],
            "validation": compute_metrics(val.y_true, pv),
            "test": compute_metrics(test.y_true, pt),
            "test_mae_by_horizon_step": mae_by_horizon(test.y_true, pt),
            "test_peak_demand": peak_demand_metrics(test.y_true, pt, H, prepared.unit),
            "training": {k: L.history[k] for k in ("epochs_run", "best_epoch", "stopped_early") if k in L.history},
        }
        predictions_table(test, pt, prepared.target, prepared.unit, m, L.metadata["model_version"],
                          generated_at).to_csv(out_dir / f"test_predictions_{m}.csv", index=False)

    metric = cfg.train.selection_metric
    # ---- ensemble: weight + interval fitted on VALIDATION only, then scored on test like the others
    if all(m in model_names for m in COMPONENTS):
        weights = fit_weights(val.y_true, val_preds, metric)
        pv, pt = combine(val_preds, weights), combine(test_preds, weights)
        offsets = interval_offsets(val.y_true, pv, INTERVAL_LEVEL)
        version = f"{ENSEMBLE_NAME}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
        test_preds[ENSEMBLE_NAME] = pt
        results[ENSEMBLE_NAME] = {
            "model_version": version,
            "weights": weights,
            "validation": compute_metrics(val.y_true, pv),
            "test": compute_metrics(test.y_true, pt),
            "test_mae_by_horizon_step": mae_by_horizon(test.y_true, pt),
            "test_peak_demand": peak_demand_metrics(test.y_true, pt, H, prepared.unit),
            "prediction_interval": {
                "level_percent": INTERVAL_LEVEL,
                "method": "per-step 10th/90th percentiles of validation errors added to the forecast",
                "validation_coverage_percent": interval_coverage(val.y_true, pv, offsets, clip),
                "test_coverage_percent": interval_coverage(test.y_true, pt, offsets, clip),
            },
            "note": f"Weights fitted on the validation period by {metric.upper()}; scored on test unchanged.",
        }
        predictions_table(test, pt, prepared.target, prepared.unit, ENSEMBLE_NAME, version,
                          generated_at).to_csv(out_dir / f"test_predictions_{ENSEMBLE_NAME}.csv", index=False)
        save_ensemble({
            "model_name": ENSEMBLE_NAME, "model_version": version, "created_at": generated_at,
            "weights": weights, "fitted_on": "validation", "criterion": f"validation_{metric}",
            "component_versions": {m: loaded[m].metadata["model_version"] for m in COMPONENTS},
            "data_signature": ref.metadata["data_signature"], "target": prepared.target, "unit": prepared.unit,
            "horizon_steps": H, "interval_level_percent": INTERVAL_LEVEL, "interval_offsets": offsets,
            "validation_metrics": results[ENSEMBLE_NAME]["validation"],
            "test_metrics": results[ENSEMBLE_NAME]["test"],
            "prediction_interval": results[ENSEMBLE_NAME]["prediction_interval"],
        }, artifacts_root)

    # ---- model selection on VALIDATION only
    key = "wape_percent" if metric == "wape" else metric
    val_scores = {m: results[m]["validation"][key] for m in results}
    usable = {m: v for m, v in val_scores.items() if v is not None}
    selection: Dict[str, Any] = {"criterion": f"validation_{metric}", "validation_scores": val_scores}
    if len(model_names) < 2:
        selection.update(preferred_model=None, note="Only one model evaluated; no comparison possible.")
    elif not usable:
        selection.update(preferred_model=None, note="Validation metric undefined for all models.")
    else:
        best = min(usable, key=usable.get)
        note = "Chosen from validation performance only."
        if best == ENSEMBLE_NAME:
            note += (" The ensemble weight was fitted on the same validation period, so its validation score "
                     "is slightly optimistic; test_ranking_agrees shows whether it also wins on unseen data.")
        selection.update(preferred_model=best, note=note)
        test_scores = {m: results[m]["test"][key] for m in results}
        if all(v is not None for v in test_scores.values()):
            selection["test_ranking_agrees"] = min(test_scores, key=test_scores.get) == best

    report = {
        "generated_at": generated_at, "target_variable": prepared.target, "unit": prepared.unit,
        "steps": prepared.steps, "boundaries": prepared.boundaries, "models": results, "selection": selection,
        "dataset_warnings": prepared.warnings,
        "caveat": "Metrics are measured on this dataset only and say nothing about other buildings or periods.",
    }
    (out_dir / "evaluation_metrics.json").write_text(json.dumps(to_jsonable(report), indent=2), encoding="utf-8")

    rows = [{"model": m, "split": s, "n_values": results[m][s]["n_values"], "mae": results[m][s]["mae"],
             "rmse": results[m][s]["rmse"], "wape_percent": results[m][s]["wape_percent"],
             "target": prepared.target, "unit": prepared.unit}
            for m in results for s in ("validation", "test")]
    pd.DataFrame(rows).to_csv(out_dir / "model_comparison.csv", index=False)

    if make_plots:
        plot_actual_vs_predicted(test, test_preds, prepared.target, prepared.unit, plots_dir)
        plot_error_over_time(test, test_preds, prepared.unit, prepared.steps["interval_seconds"], plots_dir)
        plot_mae_by_horizon(test, test_preds, prepared.unit, prepared.steps["interval_seconds"], plots_dir)
        if prepared.unit == "kW":
            plot_peak_demand(test, test_preds, plots_dir, H)
    return report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate and compare trained LSTM and MLP models and their ensemble.")
    p.add_argument("--csv", required=True, help="The same CSV the models were trained on.")
    p.add_argument("--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES))
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    p.add_argument("--output-dir", default=str(OUTPUTS_DIR))
    p.add_argument("--no-plots", action="store_true")
    args = p.parse_args(argv)
    try:
        rep = run_evaluation(args.csv, args.models, args.artifacts_dir, args.output_dir, not args.no_plots)
    except (PowerPulseError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    u = rep["unit"]
    print(f"Target: {rep['target_variable']} ({u})   Test period: {rep['boundaries']['test']['start']} -> "
          f"{rep['boundaries']['test']['end']}")
    for m, r in rep["models"].items():
        for split in ("validation", "test"):
            s = r[split]
            print(f"{m.upper():8s} {split:10s} MAE={s['mae']:.4f} RMSE={s['rmse']:.4f} WAPE%={s['wape_percent']}")
    ens = rep["models"].get(ENSEMBLE_NAME)
    if ens:
        pi = ens["prediction_interval"]
        print("Ensemble = " + " + ".join(f"{w:.2f} x {m.upper()}" for m, w in ens["weights"].items())
              + f"; {pi['level_percent']}% range covered {pi['test_coverage_percent']:.1f}% of test actuals")
    print("Selection:", rep["selection"])
    print(f"Reports written to {args.output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
