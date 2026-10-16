"""Tests for the LSTM+MLP ensemble maths and the forecast report format (no TensorFlow needed)."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.ensemble import (apply_interval, combine, fit_weights, interval_coverage, interval_offsets,
                          load_ensemble)
from src.errors import ArtifactError
from src.report import build_forecast_report, render_report


class TestEnsembleMaths(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = 50 + 10 * rng.standard_normal((500, 24))
        self.a = self.y + rng.normal(0, 4, self.y.shape)          # noisier model
        self.b = self.y + rng.normal(0, 2, self.y.shape)          # better model

    def test_weights_sum_to_one_and_favour_the_better_model(self):
        w = fit_weights(self.y, {"lstm": self.a, "mlp": self.b}, "rmse")
        self.assertAlmostEqual(w["lstm"] + w["mlp"], 1.0, places=6)
        self.assertGreater(w["mlp"], w["lstm"])
        # independent errors with variances 16 and 4 -> optimal lstm weight 4 / 20 = 0.2
        self.assertAlmostEqual(w["lstm"], 0.2, delta=0.05)

    def test_ensemble_never_worse_than_best_single_on_fitting_data(self):
        preds = {"lstm": self.a, "mlp": self.b}
        rmse = lambda p: np.sqrt(np.mean((self.y - p) ** 2))
        ens = combine(preds, fit_weights(self.y, preds, "rmse"))
        self.assertLessEqual(rmse(ens), min(rmse(self.a), rmse(self.b)) + 1e-9)

    def test_interval_covers_about_its_level(self):
        pred = combine({"lstm": self.a, "mlp": self.b}, {"lstm": 0.2, "mlp": 0.8})
        offsets = interval_offsets(self.y, pred, 80)
        self.assertEqual(len(offsets["lower"]), 24)
        self.assertAlmostEqual(interval_coverage(self.y, pred, offsets), 80.0, delta=1.0)
        lower, upper = apply_interval(np.zeros(24), offsets)
        self.assertTrue((lower >= 0).all() and (upper >= lower).all())       # clipped at zero

    def test_missing_ensemble_file_has_clear_error(self):
        with tempfile.TemporaryDirectory() as d, self.assertRaises(ArtifactError) as cm:
            load_ensemble(d)
        self.assertIn("python -m src.evaluate", str(cm.exception))


class TestForecastReport(unittest.TestCase):
    def frame(self, actual=None):
        t = pd.date_range("2024-01-01", periods=4, freq="1h")
        return pd.DataFrame({"timestamp": t, "actual_value": actual if actual is not None else [np.nan] * 4,
                             "predicted_value": [10.0, 40.0, 20.0, 30.0], "target_variable": "demand_kw",
                             "unit": "kW", "model_name": "ensemble", "model_version": "ens-1",
                             "forecast_generated_at": "x", "lower_value": [8.0, 35.0, 15.0, 25.0],
                             "upper_value": [12.0, 45.0, 25.0, 35.0], "lstm_predicted_value": [9.0, 41.0, 19.0, 29.0],
                             "mlp_predicted_value": [11.0, 39.0, 21.0, 31.0]})

    def test_report_summary_values_and_render(self):
        evaluation = {"models": {"ensemble": {"model_version": "ens-1",
                                              "validation": {"mae": 1, "rmse": 2, "wape_percent": 3},
                                              "test": {"mae": 4, "rmse": 5, "wape_percent": 6}}},
                      "boundaries": {"test": {"start": "a", "end": "b"}}}
        components = [{"name": "lstm", "version": "l", "weight": 0.4}, {"name": "mlp", "version": "m", "weight": 0.6}]
        r = build_forecast_report(self.frame(), "ensemble", "ens-1", pd.Timestamp("2023-12-31 23:00"), 3600,
                                  components, {"level_percent": 80, "method": "m", "test_coverage_percent": 82.0},
                                  evaluation, "data\\dummy_building.csv")
        self.assertEqual(r["summary"]["peak"], {"value": 40.0, "timestamp": "2024-01-01 01:00"})
        self.assertEqual(r["summary"]["total_energy_kwh"], 100.0)
        self.assertEqual(r["forecast"]["kind"], "future")
        self.assertEqual(r["accuracy"]["test"]["rmse"], 5.0)
        self.assertTrue(r["data"]["synthetic"])
        self.assertEqual(r["values"][1], {"timestamp": "2024-01-01 01:00", "predicted": 40.0, "lower": 35.0,
                                          "upper": 45.0, "lstm": 41.0, "mlp": 39.0, "actual": None})
        md = render_report(r)
        for text in ("0.40 x LSTM + 0.60 x MLP", "| Peak | 40.0 kW at 2024-01-01 01:00 |", "| Time | Forecast | Range | LSTM | MLP |",
                     "35.0 - 45.0", "SYNTHETIC"):
            self.assertIn(text, md)

    def test_backtest_and_stale_evaluation(self):
        r = build_forecast_report(self.frame(actual=[11.0, 38.0, 22.0, 30.0]), "ensemble", "ens-2",
                                  pd.Timestamp("2023-12-31 23:00"), 3600,
                                  evaluation={"models": {"ensemble": {"model_version": "ens-1"}}})
        self.assertEqual(r["forecast"]["kind"], "backtest")
        self.assertAlmostEqual(r["backtest"]["mae"], 1.25)
        self.assertIsNone(r["accuracy"])
        self.assertTrue(any("rerun python -m src.evaluate" in c for c in r["caveats"]))
        self.assertIn("| Actual |", render_report(r))


if __name__ == "__main__":
    unittest.main()
