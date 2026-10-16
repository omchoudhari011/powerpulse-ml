"""Train the LSTM and/or MLP on a real CSV.

    python -m src.train --model lstm --csv data\\my_building.csv
    python -m src.train --model mlp  --csv data\\my_building.csv
    python -m src.train --model both --csv data\\my_building.csv
"""
from __future__ import annotations

import argparse
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Optional

import numpy as np

from . import models as model_utils
from .artifacts import save_artifacts
from .config import ARTIFACTS_DIR, MODEL_NAMES, Config, add_config_arguments, config_from_args
from .errors import PowerPulseError
from .inference import inputs_for_model, predict_original_units
from .metrics import compute_metrics
from .models import lstm_model, mlp_model
from .pipeline import PreparedData, load_and_prepare


def _library_versions() -> Dict[str, str]:
    import pandas, sklearn
    v = {"python": platform.python_version(), "numpy": np.__version__, "pandas": pandas.__version__,
         "scikit-learn": sklearn.__version__}
    try:
        import tensorflow as tf
        v["tensorflow"] = tf.__version__
    except ImportError:
        v["tensorflow"] = "not installed"
    return v


def build_model(model_name: str, prepared: PreparedData, cfg: Config):
    H = prepared.steps["horizon_steps"]
    lr = cfg.train.learning_rate
    if model_name == "lstm":
        return lstm_model.build_lstm_model(
            input_steps=prepared.steps["input_steps"], n_features=len(prepared.seq_feature_names), horizon=H,
            units=cfg.lstm.units, num_layers=cfg.lstm.num_layers, dropout=cfg.lstm.dropout, learning_rate=lr)
    if model_name == "mlp":
        return mlp_model.build_mlp_model(
            n_features=len(prepared.tab_feature_names), horizon=H, hidden_units=cfg.mlp.hidden_units,
            dropout=cfg.mlp.dropout, learning_rate=lr)
    raise ValueError(f"Unknown model '{model_name}'. Choose from {MODEL_NAMES}.")


def train_model(model_name: str, csv_path, cfg: Config, artifacts_root=None, verbose: int = 1,
                prepared: Optional[PreparedData] = None) -> Dict[str, Any]:
    """Train one model, save artifacts and return a summary (measured values only)."""
    cfg.validate()
    model_utils.set_global_seed(cfg.train.seed, cfg.train.deterministic_ops)
    prepared = prepared or load_and_prepare(csv_path, cfg)
    tr, va = prepared.splits["train"], prepared.splits["validation"]

    model = build_model(model_name, prepared, cfg)
    fit = model_utils.fit_model(
        model, inputs_for_model(model_name, tr), tr.Y_scaled, inputs_for_model(model_name, va), va.Y_scaled,
        batch_size=cfg.train.batch_size, max_epochs=cfg.train.max_epochs,
        patience=cfg.train.early_stopping_patience, shuffle=cfg.train.shuffle_train, verbose=verbose)

    pred_val = predict_original_units(model, prepared.preprocessor, inputs_for_model(model_name, va),
                                      cfg.train.clip_non_negative)
    val_metrics = compute_metrics(va.y_true, pred_val)
    version = f"{model_name}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    metadata = {
        "model_name": model_name, "model_version": version,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "target": prepared.target, "unit": prepared.unit,
        "interval_seconds": prepared.steps["interval_seconds"], "steps": prepared.steps,
        "seq_feature_names": prepared.seq_feature_names, "tab_feature_names": prepared.tab_feature_names,
        "exog_columns": prepared.preprocessor.exog_columns, "dropped_lags": prepared.dropped_lags,
        "boundaries": prepared.boundaries, "csv_path": prepared.csv_path, "csv_sha256": prepared.csv_sha256,
        "data_signature": prepared.data_signature, "seed": cfg.train.seed,
        "validation_metrics": val_metrics, "clean_stats": prepared.clean_stats,
        "dataset_warnings": prepared.warnings, "libraries": _library_versions(),
        "note": "validation_metrics are measured on the chronological validation period only; "
                "run src.evaluate for held-out test results.",
    }
    out = save_artifacts(model_name, model, prepared.preprocessor, cfg.to_dict(), fit, metadata, artifacts_root)
    return {"model_name": model_name, "model_version": version, "artifact_dir": str(out),
            "epochs_run": fit["epochs_run"], "best_epoch": fit["best_epoch"],
            "stopped_early": fit["stopped_early"], "validation_metrics": val_metrics,
            "n_samples": {k: len(v) for k, v in prepared.splits.items()}}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train PowerPulse LSTM / MLP forecasting models.")
    p.add_argument("--model", choices=list(MODEL_NAMES) + ["both"], required=True)
    p.add_argument("--csv", required=True, help="Path to the real building-energy CSV.")
    p.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR))
    p.add_argument("--quiet", action="store_true", help="Hide per-epoch training logs.")
    add_config_arguments(p)
    args = p.parse_args(argv)
    try:
        cfg = config_from_args(args)
        prepared = load_and_prepare(args.csv, cfg)
        for w in prepared.warnings:
            print(f"WARNING: {w}")
        print(f"Target: {prepared.target} ({prepared.unit}) | interval {prepared.steps['interval_seconds']} s | "
              f"input {prepared.steps['input_steps']} steps | horizon {prepared.steps['horizon_steps']} steps")
        print("Samples:", {k: len(v) for k, v in prepared.splits.items()})
        names = list(MODEL_NAMES) if args.model == "both" else [args.model]
        for name in names:
            print(f"\n=== Training {name.upper()} ===")
            s = train_model(name, args.csv, cfg, args.artifacts_dir, 0 if args.quiet else 1, prepared)
            m = s["validation_metrics"]
            print(f"{name}: epochs={s['epochs_run']} best_epoch={s['best_epoch']} | measured validation "
                  f"MAE={m['mae']:.4f} RMSE={m['rmse']:.4f} WAPE%={m['wape_percent']} ({prepared.unit})")
            print(f"Artifacts saved to {s['artifact_dir']}")
        print("\nNext: python -m src.evaluate --csv", args.csv)
        return 0
    except (PowerPulseError, FileNotFoundError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
