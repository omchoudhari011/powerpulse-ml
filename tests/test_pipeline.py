import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.errors import InsufficientDataError
from src.feature_engineering import (calendar_features, hours_to_steps, make_tabular, resolve_steps,
                                     valid_origins)
from src.pipeline import load_and_prepare
from src.preprocessing import Preprocessor, chronological_split
from tests.helpers import make_frame, write_csv


def prepared_from(df, cfg=None):
    cfg = cfg or Config()
    with tempfile.TemporaryDirectory() as d:
        return load_and_prepare(write_csv(df, Path(d) / "x.csv"), cfg), cfg


class TestSplitAndScaling(unittest.TestCase):
    def test_chronological_split_boundaries(self):
        self.assertEqual(chronological_split(1000, 0.7, 0.15), (700, 850))
        with self.assertRaises(Exception):
            chronological_split(2, 0.7, 0.15)

    def test_scaler_is_fit_on_train_rows_only(self):
        p, _ = prepared_from(make_frame(24 * 60))
        n = p.boundaries["n_rows"]
        a = p.boundaries["train"]["rows"]
        raw = make_frame(24 * 60)["demand_kw"].to_numpy()
        self.assertAlmostEqual(p.preprocessor.scaler.mean_[0], raw[:a].mean(), places=6)
        self.assertNotAlmostEqual(p.preprocessor.scaler.mean_[0], raw.mean(), places=3)

    def test_preprocessor_roundtrip(self):
        p, _ = prepared_from(make_frame(24 * 60))
        with tempfile.TemporaryDirectory() as d:
            p.preprocessor.save(Path(d) / "p.joblib")
            q = Preprocessor.load(Path(d) / "p.joblib")
        x = np.array([10.0, 50.0])
        np.testing.assert_allclose(q.inverse_target((x - p.preprocessor.scaler.mean_[0]) / p.preprocessor.scaler.scale_[0]), x)


class TestSequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p, cls.cfg = prepared_from(make_frame(24 * 60))
        cls.raw = make_frame(24 * 60)["demand_kw"].to_numpy()

    def test_shapes(self):
        tr = self.p.splits["train"]
        L, H = self.p.steps["input_steps"], self.p.steps["horizon_steps"]
        self.assertEqual((L, H), (24, 24))
        self.assertEqual(tr.X_seq.shape, (len(tr), 24, len(self.p.seq_feature_names)))
        self.assertEqual(tr.X_tab.shape, (len(tr), len(self.p.tab_feature_names)))
        self.assertEqual(tr.Y_scaled.shape, (len(tr), 24))
        self.assertEqual(self.p.steps["lag_steps"], [1, 2, 3, 24, 48, 168])

    def test_alignment_of_inputs_targets_and_lags(self):
        pre, tr = self.p.preprocessor, self.p.splits["validation"]
        m, s = pre.scaler.mean_[0], pre.scaler.scale_[0]
        for i in (0, len(tr) // 2, len(tr) - 1):
            t = tr.origins[i]
            self.assertAlmostEqual(tr.X_seq[i, -1, 0], (self.raw[t] - m) / s, places=4)
            np.testing.assert_allclose(tr.y_true[i], self.raw[t + 1:t + 25])
            names = self.p.tab_feature_names
            self.assertAlmostEqual(tr.X_tab[i, names.index("target_lag_1")], (self.raw[t] - m) / s, places=4)
            self.assertAlmostEqual(tr.X_tab[i, names.index("target_lag_24")], (self.raw[t - 23] - m) / s, places=4)
            self.assertAlmostEqual(tr.X_tab[i, names.index("target_lag_168")], (self.raw[t - 167] - m) / s, places=4)

    def test_no_future_information_in_inputs(self):
        # Changing the data AFTER an origin must not change that sample's inputs.
        df = make_frame(24 * 60)
        p1, _ = prepared_from(df)
        t = int(p1.splits["validation"].origins[5])
        df2 = df.copy()
        df2.loc[t + 1:, ["demand_kw", "temperature", "occupancy"]] *= 3
        p2, _ = prepared_from(df2)
        j = list(p2.splits["validation"].origins).index(t)
        np.testing.assert_allclose(p1.splits["validation"].X_seq[5, :, :].shape, p2.splits["validation"].X_seq[j].shape)
        # scaler differs only through train rows (rows <= t+1 are not all train), so compare raw columns
        # that do not depend on scaling: calendar features + equipment_status
        cols = [i for i, n in enumerate(p1.seq_feature_names) if n in ("hour_sin", "is_weekend", "equipment_status")]
        np.testing.assert_allclose(p1.splits["validation"].X_seq[5][:, cols], p2.splits["validation"].X_seq[j][:, cols])

    def test_splits_are_chronological_and_disjoint(self):
        tr, va, te = (self.p.splits[k] for k in ("train", "validation", "test"))
        b = self.p.boundaries
        self.assertLessEqual(pd.Timestamp(tr.target_times.max()), pd.Timestamp(b["train"]["end"]))
        self.assertGreaterEqual(pd.Timestamp(va.target_times.min()), pd.Timestamp(b["validation"]["start"]))
        self.assertLessEqual(pd.Timestamp(va.target_times.max()), pd.Timestamp(b["validation"]["end"]))
        self.assertGreaterEqual(pd.Timestamp(te.target_times.min()), pd.Timestamp(b["test"]["start"]))
        self.assertLess(tr.target_times.max(), va.target_times.min())
        self.assertLess(va.target_times.max(), te.target_times.min())
        for s in (tr, va, te):
            self.assertTrue((np.diff(s.origins) > 0).all())

    def test_windows_never_target_imputed_or_missing_readings(self):
        df = make_frame(24 * 60)
        df.loc[500:530, "demand_kw"] = np.nan        # 31 h gap, only 3 h could be forward-filled
        df.loc[700, "demand_kw"] = np.nan            # 1 h gap -> imputed for inputs only
        p, _ = prepared_from(df)
        missing_times = set(make_frame(24 * 60)["timestamp"].iloc[[700] + list(range(500, 531))])
        for s in p.splits.values():
            got = set(pd.to_datetime(s.target_times.reshape(-1)))
            self.assertFalse(got & missing_times)
        self.assertGreater(p.clean_stats["target_imputed_by_ffill"], 0)

    def test_both_models_share_identical_samples(self):
        for s in self.p.splits.values():
            self.assertEqual(len(s.X_seq), len(s.X_tab))
            self.assertEqual(len(s.X_seq), len(s.y_true))

    def test_optional_features_not_required(self):
        p, _ = prepared_from(make_frame(24 * 60, weather=False, occupancy=False, equipment=False))
        self.assertEqual(p.preprocessor.exog_columns, [])
        self.assertEqual(p.seq_feature_names[0], "demand_kw")
        self.assertEqual(p.splits["train"].X_tab.shape[1], len(p.steps["lag_steps"]) + 7)

    def test_insufficient_data_raises(self):
        with self.assertRaises(InsufficientDataError):
            prepared_from(make_frame(60))

    def test_long_lag_dropped_when_history_is_short(self):
        p, _ = prepared_from(make_frame(24 * 14))
        self.assertIn(168, p.dropped_lags)
        self.assertNotIn(168, p.steps["lag_steps"])


class TestResolutionAndCalendar(unittest.TestCase):
    def test_fifteen_minute_data_scales_steps(self):
        s = resolve_steps(Config(), 900)
        self.assertEqual((s["input_steps"], s["horizon_steps"]), (96, 96))
        self.assertEqual(s["lag_steps"], [4, 8, 12, 96, 192, 672])

    def test_daily_data_interpreted_as_steps(self):
        self.assertEqual(hours_to_steps(24, 86400), 24)
        self.assertEqual(resolve_steps(Config(), 86400)["mode"], "steps")

    def test_calendar_features(self):
        idx = pd.date_range("2024-01-06 06:00", periods=2, freq="1D")   # Saturday, Sunday
        c = calendar_features(idx)
        self.assertEqual(c["is_weekend"].tolist(), [1.0, 1.0])
        self.assertAlmostEqual(c["hour_sin"].iloc[0], 1.0)
        self.assertTrue(((c.drop(columns="is_weekend").abs()) <= 1.0 + 1e-9).all().all())

    def test_end_to_end_prepare_on_15_minute_data(self):
        p, _ = prepared_from(make_frame(24 * 4 * 40, step="15min"))
        self.assertEqual(p.steps["interval_seconds"], 900)
        self.assertEqual(p.splits["train"].X_seq.shape[1], 96)
        self.assertEqual(p.splits["train"].Y_scaled.shape[1], 96)


if __name__ == "__main__":
    unittest.main()
