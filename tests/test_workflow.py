"""End-to-end PLUMBING tests (train -> artifacts -> evaluate -> predict) with a stub model.

The stub is a persistence forecaster standing in for Keras, so these tests verify data flow, artifact
handling, metrics, report/plot/forecast writing and error handling WITHOUT TensorFlow. They say nothing
about LSTM/MLP accuracy; see test_models.py for the real Keras tests (skipped when TF is missing).
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import artifacts, evaluate, predict, train
from src.config import Config
from src.errors import ArtifactError, DatasetError
from tests.helpers import make_frame, write_csv


class StubModel:
    def __init__(self, horizon):
        self.horizon = horizon

    def predict(self, X, batch_size=256, verbose=0):
        last = X[:, -1, 0] if X.ndim == 3 else X[:, 0]       # latest scaled target value
        return np.repeat(last[:, None], self.horizon, axis=1)


def stub_fit(model, *a, **k):
    return {"history": {"loss": [1.0], "val_loss": [1.0]}, "epochs_run": 1, "best_epoch": 1,
            "best_val_loss_scaled": 1.0, "stopped_early": True}


def stub_build(model_name, prepared, cfg):
    return StubModel(prepared.steps["horizon_steps"])


def stub_save(model, path):
    Path(path).write_text(json.dumps({"horizon": model.horizon}))


def stub_load(path):
    return StubModel(json.loads(Path(path).read_text())["horizon"])


class WorkflowBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.df = make_frame(24 * 60)
        self.csv = write_csv(self.df, self.root / "data.csv")
        self.art, self.out = self.root / "art", self.root / "out"
        patches = [mock.patch.object(train, "build_model", stub_build),
                   mock.patch("src.models.fit_model", stub_fit),
                   mock.patch.object(artifacts, "save_keras_model", stub_save),
                   mock.patch.object(artifacts, "load_keras_model", stub_load)]
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def train_both(self, cfg=None):
        cfg = cfg or Config()
        return [train.train_model(m, self.csv, cfg, self.art, verbose=0) for m in ("lstm", "mlp")]


class TestWorkflow(WorkflowBase):
    def test_train_writes_all_artifacts_and_metadata(self):
        res = self.train_both()
        for m in ("lstm", "mlp"):
            d = self.art / m
            for f in artifacts.FILES.values():
                self.assertTrue((d / f).exists(), f)
            meta = json.loads((d / "metadata.json").read_text())
            self.assertEqual((meta["target"], meta["unit"]), ("demand_kw", "kW"))
            self.assertIn("mae", meta["validation_metrics"])
        self.assertEqual(res[0]["n_samples"], res[1]["n_samples"])

    def test_evaluate_writes_reports_plots_and_schema(self):
        self.train_both()
        rep = evaluate.run_evaluation(self.csv, ("lstm", "mlp"), self.art, self.out)
        self.assertIn(rep["selection"]["preferred_model"], ("lstm", "mlp", "ensemble"))
        self.assertEqual(rep["selection"]["criterion"], "validation_rmse")
        for f in ("evaluation_metrics.json", "model_comparison.csv", "test_predictions_lstm.csv",
                  "test_predictions_mlp.csv"):
            self.assertTrue((self.out / f).exists(), f)
        for f in ("actual_vs_predicted_first_14d.png", "actual_vs_predicted_full_test.png",
                  "forecast_error_over_time.png", "mae_by_horizon.png", "peak_demand_error.png"):
            self.assertTrue((self.out / "plots" / f).exists(), f)
        tbl = pd.read_csv(self.out / "test_predictions_lstm.csv")
        for c in ["timestamp", "actual_value", "predicted_value", "target_variable", "unit", "model_name",
                  "model_version", "forecast_generated_at"]:
            self.assertIn(c, tbl.columns)
        self.assertFalse(tbl["actual_value"].isna().any())
        self.assertTrue(rep["models"]["lstm"]["test_peak_demand"]["available"])
        json.loads((self.out / "evaluation_metrics.json").read_text())      # valid JSON (no NaN tokens)

    def test_persistence_stub_metrics_match_manual_computation(self):
        self.train_both()
        rep = evaluate.run_evaluation(self.csv, ("lstm",), self.art, self.out, make_plots=False)
        tbl = pd.read_csv(self.out / "test_predictions_lstm.csv")
        manual = np.mean(np.abs(tbl["actual_value"] - tbl["predicted_value"]))
        self.assertAlmostEqual(rep["models"]["lstm"]["test"]["mae"], manual, places=6)

    def test_energy_target_has_no_peak_metric_and_kwh_label(self):
        df = make_frame(24 * 60, demand=False, energy=True)
        csv = write_csv(df, self.root / "e.csv")
        self.csv = csv
        self.train_both()
        rep = evaluate.run_evaluation(csv, ("lstm", "mlp"), self.art, self.out)
        self.assertEqual(rep["unit"], "kWh")
        self.assertFalse(rep["models"]["lstm"]["test_peak_demand"]["available"])
        self.assertFalse((self.out / "plots" / "peak_demand_error.png").exists())

    def test_forecast_future_and_backtest(self):
        self.train_both()
        f = predict.generate_forecast("lstm", self.csv, self.art, self.out)
        self.assertEqual(len(f), 24)
        self.assertTrue(f["actual_value"].isna().all())
        self.assertEqual(f["timestamp"].iloc[0], self.df["timestamp"].iloc[-1] + pd.Timedelta(hours=1))
        self.assertEqual(list(f.columns), predict.FORECAST_COLUMNS)
        self.assertTrue(list(self.out.glob("forecast_lstm_*.csv")) and list(self.out.glob("forecast_lstm_*.json")))
        origin = str(self.df["timestamp"].iloc[600])
        b = predict.generate_forecast("mlp", self.csv, self.art, self.out, origin=origin, save=False)
        np.testing.assert_allclose(b["actual_value"].to_numpy(), self.df["demand_kw"].iloc[601:625].to_numpy())
        self.assertAlmostEqual(b["predicted_value"].iloc[0], self.df["demand_kw"].iloc[600], places=3)  # stub = persistence

    def test_ensemble_links_models_end_to_end(self):
        self.train_both()
        rep = evaluate.run_evaluation(self.csv, ("lstm", "mlp"), self.art, self.out, make_plots=False)
        ens = rep["models"]["ensemble"]
        self.assertAlmostEqual(sum(ens["weights"].values()), 1.0)
        self.assertIn("ensemble", rep["selection"]["validation_scores"])
        self.assertTrue((self.art / "ensemble" / "ensemble.json").exists())
        self.assertTrue((self.out / "test_predictions_ensemble.csv").exists())
        self.assertIn("ensemble", set(pd.read_csv(self.out / "model_comparison.csv")["model"]))

        f, report = predict.generate_forecast("ensemble", self.csv, self.art, self.out, return_report=True)
        self.assertEqual(list(f.columns), predict.FORECAST_COLUMNS + predict.ENSEMBLE_COLUMNS)
        self.assertTrue((f["lower_value"] <= f["upper_value"]).all())
        w = ens["weights"]
        np.testing.assert_allclose(f["predicted_value"], w["lstm"] * f["lstm_predicted_value"]
                                   + w["mlp"] * f["mlp_predicted_value"])
        self.assertEqual(report["model"]["name"], "ensemble")
        self.assertIsNotNone(report["accuracy"])                    # matched to this ensemble version
        self.assertTrue(list(self.out.glob("forecast_report_ensemble_*.json")))
        self.assertTrue(list(self.out.glob("forecast_report_ensemble_*.md")))
        self.assertEqual(predict.main(["--model", "best", "--csv", self.csv, "--artifacts-dir", str(self.art),
                                       "--output-dir", str(self.out), "--format", "json"]), 0)

        meta_path = self.art / "lstm" / "metadata.json"            # simulate retraining one component
        meta = json.loads(meta_path.read_text())
        meta["model_version"] = "lstm-retrained"
        meta_path.write_text(json.dumps(meta))
        with self.assertRaises(ArtifactError):
            predict.generate_forecast("ensemble", self.csv, self.art, self.out, save=False)

    def test_forecast_errors_are_clear(self):
        self.train_both()
        with self.assertRaises(DatasetError):
            predict.generate_forecast("lstm", self.csv, self.art, self.out, origin="2023-01-02 00:30")
        short = write_csv(self.df.iloc[:50], self.root / "short.csv")
        with self.assertRaises(Exception) as cm:
            predict.generate_forecast("lstm", short, self.art, self.out)
        self.assertIn("history", str(cm.exception))
        no_temp = write_csv(self.df.drop(columns="temperature"), self.root / "nt.csv")
        with self.assertRaises(DatasetError):
            predict.generate_forecast("lstm", no_temp, self.art, self.out)

    def test_missing_artifacts_error_message(self):
        with self.assertRaises(ArtifactError) as cm:
            artifacts.load_artifacts("lstm", self.art)
        self.assertIn("python -m src.train", str(cm.exception))

    def test_incomparable_models_or_changed_csv_are_rejected(self):
        train.train_model("lstm", self.csv, Config(), self.art, verbose=0)
        other = Config()
        other.features.lag_hours = (1, 2, 3)
        train.train_model("mlp", self.csv, other, self.art, verbose=0)
        with self.assertRaises(ArtifactError):
            evaluate.run_evaluation(self.csv, ("lstm", "mlp"), self.art, self.out)
        self.train_both()
        changed = write_csv(make_frame(24 * 60, seed=9), self.root / "changed.csv")
        with self.assertRaises(ArtifactError):
            evaluate.run_evaluation(changed, ("lstm", "mlp"), self.art, self.out)

    def test_cli_exit_codes(self):
        self.assertEqual(train.main(["--model", "lstm", "--csv", str(self.root / "nope.csv"),
                                     "--artifacts-dir", str(self.art)]), 2)
        self.assertEqual(train.main(["--model", "both", "--csv", self.csv, "--artifacts-dir", str(self.art),
                                     "--quiet"]), 0)
        self.assertEqual(predict.main(["--model", "lstm", "--csv", self.csv, "--artifacts-dir", str(self.art),
                                       "--output-dir", str(self.out)]), 0)
        from src import data_validation
        self.assertEqual(data_validation.main(["--csv", self.csv]), 0)
        self.assertEqual(data_validation.main(["--csv", write_csv(make_frame(40), self.root / "s.csv")]), 1)

    def test_insufficient_data_blocks_training(self):
        tiny = write_csv(make_frame(80), self.root / "tiny.csv")
        with self.assertRaises(Exception) as cm:
            train.train_model("lstm", tiny, Config(), self.art, verbose=0)
        self.assertIn("Insufficient", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
