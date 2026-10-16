"""Tests for tools/convert_bdg2.py on tiny files shaped like the BDG2 originals (no download)."""
import importlib.util
import io
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

import pandas as pd

from src.config import Config, PROJECT_ROOT
from src.data_loader import load_csv
from src.data_validation import validate_dataframe

spec = importlib.util.spec_from_file_location("convert_bdg2", PROJECT_ROOT / "tools" / "convert_bdg2.py")
convert_bdg2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(convert_bdg2)


class TestConvertBdg2(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.src = Path(self.tmp.name)
        t = pd.date_range("2016-01-01", periods=24 * 30, freq="1h")
        pd.DataFrame({"building_id": ["Hog_office_A", "Hog_lodging_B"], "site_id": ["Hog", "Hog"],
                      "primaryspaceusage": ["Office", "Lodging/residential"],
                      "sub_primaryspaceusage": ["Office", "Dorm"], "sqm": [100.0, 50.0],
                      "timezone": ["US/Central"] * 2}).to_csv(self.src / "metadata.csv", index=False)
        pd.DataFrame({"timestamp": t, "Hog_office_A": 50 + (t.hour >= 8) * 30 + t.dayofyear * 0.1,
                      "Hog_lodging_B": 10.0}).to_csv(self.src / "electricity_cleaned.csv", index=False)
        w = pd.DataFrame({"timestamp": t, "site_id": "Hog", "airTemperature": 20.0, "dewTemperature": 10.0})
        w.loc[5, "airTemperature"] = None
        pd.concat([w, pd.DataFrame({"timestamp": t, "site_id": "Other", "airTemperature": 99.0,
                                    "dewTemperature": 99.0})]).to_csv(self.src / "weather.csv", index=False)

    def tearDown(self):
        self.tmp.cleanup()

    def test_converted_file_matches_powerpulse_format_and_validates(self):
        out = self.src / "out.csv"
        with redirect_stdout(io.StringIO()):
            convert_bdg2.convert(self.src, "Hog_office_A", out)
        df = pd.read_csv(out)
        self.assertEqual(list(df.columns), ["timestamp", "energy_kwh", "demand_kw", "temperature", "humidity"])
        self.assertTrue((df["energy_kwh"] == df["demand_kw"]).all())
        self.assertEqual(df["temperature"].iloc[0], 20.0)                    # own site's weather, not "Other"
        self.assertAlmostEqual(df["humidity"].iloc[0], 52.6, delta=0.5)      # 20 C air, 10 C dew point
        self.assertTrue(pd.isna(df["humidity"].iloc[5]))                     # missing stays missing
        loaded, info = load_csv(out, Config().data)
        self.assertTrue(validate_dataframe(loaded, info, Config()).ok)

    def test_humidity_is_100_when_dew_point_equals_temperature(self):
        rh = convert_bdg2.relative_humidity(pd.Series([15.0]), pd.Series([15.0]))
        self.assertAlmostEqual(rh.iloc[0], 100.0)

    def test_list_keeps_only_complete_office_buildings(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            convert_bdg2.list_offices(self.src, 10)
        self.assertIn("Hog_office_A", buf.getvalue())
        self.assertNotIn("Hog_lodging_B", buf.getvalue())

    def test_unknown_building_is_rejected(self):
        with self.assertRaises(SystemExit):
            convert_bdg2.convert(self.src, "Nope", self.src / "x.csv")


if __name__ == "__main__":
    unittest.main()
