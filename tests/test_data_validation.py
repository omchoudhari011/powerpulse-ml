import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import Config
from src.data_loader import load_csv
from src.data_validation import clean_dataframe, validate_dataframe
from src.errors import DatasetError
from tests.helpers import make_frame, write_csv


def validate(df, cfg=None, **kw):
    cfg = cfg or Config()
    with tempfile.TemporaryDirectory() as d:
        path = write_csv(df, Path(d) / "x.csv")
        raw, info = load_csv(path, cfg.data)
    return raw, info, validate_dataframe(raw, info, cfg, **kw)


class TestLoader(unittest.TestCase):
    def test_missing_file_gives_clear_error(self):
        with self.assertRaises(FileNotFoundError) as cm:
            load_csv("does_not_exist.csv")
        self.assertIn("data/", str(cm.exception))

    def test_missing_timestamp_column(self):
        with tempfile.TemporaryDirectory() as d:
            p = write_csv(make_frame(48).rename(columns={"timestamp": "time"}), Path(d) / "a.csv")
            with self.assertRaises(DatasetError):
                load_csv(p)

    def test_unsorted_rows_are_flagged_and_sorted_by_cleaning(self):
        df = make_frame(24 * 20).sample(frac=1, random_state=1)
        raw, info, rep = validate(df)
        self.assertFalse(rep.is_sorted)
        clean = clean_dataframe(raw, rep.interval_seconds, Config(), rep.target, [])
        self.assertTrue(clean.frame.index.is_monotonic_increasing)


class TestValidation(unittest.TestCase):
    def test_target_prefers_demand_and_labels_kw(self):
        _, _, rep = validate(make_frame(24 * 40, energy=True))
        self.assertEqual((rep.target, rep.target_unit), ("demand_kw", "kW"))

    def test_energy_fallback_is_labelled_kwh(self):
        _, _, rep = validate(make_frame(24 * 40, demand=False, energy=True))
        self.assertEqual((rep.target, rep.target_unit), ("energy_kwh", "kWh"))
        self.assertTrue(any("kWh" in w for w in rep.warnings))

    def test_no_target_is_error(self):
        _, _, rep = validate(make_frame(24 * 40, demand=False))
        self.assertFalse(rep.ok)

    def test_missing_optional_columns_are_explained(self):
        _, _, rep = validate(make_frame(24 * 40, weather=False, occupancy=False, equipment=False))
        for c in ("temperature", "humidity", "occupancy", "equipment_status"):
            self.assertIn(c, rep.columns_missing)
        self.assertTrue(rep.ok)

    def test_duplicates_gaps_and_irregular_intervals(self):
        df = make_frame(24 * 40)
        df = pd.concat([df, df.iloc[[10]]]).drop(df.index[100:110])
        _, _, rep = validate(df)
        self.assertEqual(rep.n_duplicate_timestamps, 1)
        self.assertGreaterEqual(rep.n_gaps, 1)
        self.assertEqual(rep.n_missing_grid_rows, 10)
        self.assertEqual(rep.interval_seconds, 3600)

    def test_invalid_and_missing_readings_counted(self):
        df = make_frame(24 * 40)
        df.loc[5, "demand_kw"] = -3.0
        df.loc[6, "humidity"] = 140
        df.loc[7, "demand_kw"] = np.nan
        _, _, rep = validate(df)
        self.assertEqual(rep.invalid_values["demand_kw"], 1)
        self.assertEqual(rep.invalid_values["humidity"], 1)
        self.assertEqual(rep.missing_values["demand_kw"], 1)

    def test_non_numeric_values_reported(self):
        df = make_frame(24 * 40).astype({"demand_kw": object})
        df.loc[3, "demand_kw"] = "abc"
        _, _, rep = validate(df)
        self.assertEqual(rep.non_numeric_values["demand_kw"], 1)

    def test_constant_target_is_error(self):
        df = make_frame(24 * 40)
        df["demand_kw"] = 5.0
        self.assertFalse(validate(df)[2].ok)

    def test_insufficient_rows_is_error(self):
        rep = validate(make_frame(48))[2]
        self.assertFalse(rep.ok)
        self.assertTrue(any("Insufficient" in e for e in rep.errors))

    def test_required_columns_for_inference(self):
        _, _, rep = validate(make_frame(24 * 40, weather=False), required_columns=["demand_kw", "temperature"])
        self.assertFalse(rep.ok)


class TestCleaning(unittest.TestCase):
    def test_invalid_becomes_nan_and_duplicates_resolved(self):
        df = make_frame(24 * 10)
        df.loc[3, "demand_kw"] = -1
        dup = df.iloc[[20]].copy()
        dup["demand_kw"] = 999.0
        raw, _, rep = validate(pd.concat([df, dup]), check_sufficiency=False)
        c = clean_dataframe(raw, 3600, Config(), "demand_kw", [])
        self.assertTrue(c.frame.index.is_unique)
        self.assertEqual(c.frame["demand_kw"].iloc[20], 999.0)       # policy "last"
        self.assertFalse(c.target_observed.iloc[3])                   # invalid reading is not "observed"

    def test_forward_fill_is_causal_and_limited(self):
        df = make_frame(24 * 10, weather=False, occupancy=False, equipment=False)
        df.loc[10:11, "demand_kw"] = np.nan          # short gap -> filled with PAST value
        df.loc[50:60, "demand_kw"] = np.nan          # long gap (11 h) -> only 3 h filled
        df.loc[0, "demand_kw"] = np.nan              # leading NaN stays NaN (no back-fill)
        orig = make_frame(24 * 10, weather=False, occupancy=False, equipment=False)["demand_kw"]
        raw, _, _ = validate(df, check_sufficiency=False)
        c = clean_dataframe(raw, 3600, Config(), "demand_kw", [])
        f = c.frame["demand_kw"]
        self.assertTrue(np.isnan(f.iloc[0]))
        self.assertAlmostEqual(f.iloc[10], df["demand_kw"].iloc[9])
        self.assertAlmostEqual(f.iloc[11], df["demand_kw"].iloc[9])
        self.assertTrue(f.iloc[50:53].notna().all())
        self.assertTrue(f.iloc[53:61].isna().all())
        self.assertFalse(c.target_observed.iloc[50:61].any())
        self.assertNotAlmostEqual(f.iloc[10], orig.iloc[11])         # never uses future values


if __name__ == "__main__":
    unittest.main()
