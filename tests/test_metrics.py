import unittest

import numpy as np

from src.metrics import compute_metrics, mae, peak_demand_metrics, rmse, wape


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y, p = np.array([10.0, 20.0, 30.0]), np.array([12.0, 18.0, 33.0])
        self.assertAlmostEqual(mae(y, p), 7 / 3)
        self.assertAlmostEqual(rmse(y, p), np.sqrt((4 + 4 + 9) / 3))
        self.assertAlmostEqual(wape(y, p), 7 / 60 * 100)

    def test_zero_denominator_returns_none(self):
        self.assertIsNone(wape(np.zeros(4), np.ones(4)))
        self.assertIn("wape_note", compute_metrics(np.zeros(4), np.ones(4)))

    def test_empty_input_is_handled(self):
        m = compute_metrics(np.array([]), np.array([]))
        self.assertEqual((m["n_values"], m["mae"], m["rmse"], m["wape_percent"]), (0, None, None, None))

    def test_shape_mismatch(self):
        with self.assertRaises(ValueError):
            mae(np.ones(3), np.ones(4))

    def test_peak_demand_error(self):
        yt = np.array([[1, 5, 2], [3, 4, 9], [1, 1, 1]], dtype=float)
        yp = np.array([[1, 4, 2], [3, 8, 7], [2, 2, 2]], dtype=float)
        r = peak_demand_metrics(yt, yp, stride=1, unit="kW")
        self.assertTrue(r["available"])
        self.assertAlmostEqual(r["mean_abs_peak_error_kw"], (1 + 1 + 1) / 3)
        self.assertAlmostEqual(r["mean_signed_peak_error_kw"], (-1 - 1 + 1) / 3)

    def test_peak_error_not_reported_for_kwh(self):
        self.assertFalse(peak_demand_metrics(np.ones((2, 3)), np.ones((2, 3)), 1, "kWh")["available"])

    def test_peak_windows_are_non_overlapping(self):
        yt = np.arange(30, dtype=float).reshape(10, 3)
        self.assertEqual(peak_demand_metrics(yt, yt, stride=3, unit="kW")["n_windows"], 4)


if __name__ == "__main__":
    unittest.main()
