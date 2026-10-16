"""Tests for the Streamlit UI: chart builders, and a headless run of the app (skipped without Streamlit)."""
import importlib.util
import unittest

from src.ui import charts

HAS_STREAMLIT = importlib.util.find_spec("streamlit") is not None


def make_report(model="ensemble", actual=False):
    components = ([{"name": "lstm", "version": "l", "weight": 0.4}, {"name": "mlp", "version": "m", "weight": 0.6}]
                  if model == "ensemble" else [{"name": model, "version": "v", "weight": 1.0}])
    values = [{"timestamp": f"2024-01-01 {h:02d}:00", "predicted": 10.0 + h, "lower": 8.0 + h if model == "ensemble" else None,
               "upper": 13.0 + h if model == "ensemble" else None, "lstm": 9.0 + h if model == "ensemble" else None,
               "mlp": 11.0 + h if model == "ensemble" else None, "actual": 10.5 + h if actual else None}
              for h in range(24)]
    return {"unit": "kW", "model": {"name": model, "version": "v", "components": components},
            "prediction_interval": {"level_percent": 80} if model == "ensemble" else None, "values": values}


class TestCharts(unittest.TestCase):
    def test_ensemble_frame_has_all_series_and_range(self):
        frame = charts.forecast_frame(make_report(actual=True))
        self.assertEqual(list(frame.columns), ["timestamp", "Ensemble", "LSTM", "MLP", "Actual", "Range low", "Range high"])

    def test_single_model_frame_uses_its_own_colour_slot(self):
        frame = charts.forecast_frame(make_report("lstm"))
        self.assertEqual(list(frame.columns), ["timestamp", "LSTM"])
        spec = charts.forecast_chart(make_report("lstm"), "light").to_dict()
        scales = [l["encoding"]["color"]["scale"] for l in spec["layer"] if "color" in l.get("encoding", {})]
        self.assertEqual(scales[0], {"domain": ["LSTM"], "range": [charts.PALETTES["light"]["LSTM"]]})

    def test_colours_follow_the_entity_in_both_modes(self):
        for mode in ("light", "dark"):
            spec = charts.forecast_chart(make_report(actual=True), mode).to_dict()
            scale = next(l["encoding"]["color"]["scale"] for l in spec["layer"] if "color" in l.get("encoding", {}))
            self.assertEqual(scale["domain"], ["Ensemble", "LSTM", "MLP", "Actual"])
            self.assertEqual(scale["range"], [charts.PALETTES[mode][s] for s in scale["domain"]])

    def test_mae_chart_from_evaluation(self):
        ev = {"unit": "kW", "steps": {"interval_seconds": 3600},
              "models": {"lstm": {"test_mae_by_horizon_step": [1, 2, 3]}, "mlp": {"test_mae_by_horizon_step": [1, 1.5, 2]},
                         "ensemble": {"test_mae_by_horizon_step": [0.9, 1.4, 1.9]}}}
        frame = charts.horizon_frame(ev)
        self.assertEqual(list(frame["Lead time (h)"]), [1.0, 2.0, 3.0])
        self.assertIsNotNone(charts.mae_by_horizon_chart(ev, "dark").to_dict())
        self.assertIsNone(charts.mae_by_horizon_chart({"models": {}}))


@unittest.skipUnless(HAS_STREAMLIT, "streamlit is not installed")
class TestApp(unittest.TestCase):
    def test_app_renders_every_tab_without_errors(self):
        from streamlit.testing.v1 import AppTest
        from src.config import PROJECT_ROOT
        at = AppTest.from_file(str(PROJECT_ROOT / "streamlit_app.py"), default_timeout=60).run()
        self.assertEqual([e.message for e in at.exception], [])
        self.assertEqual([t.label for t in at.tabs], ["Forecast", "Models", "Data & training"])


if __name__ == "__main__":
    unittest.main()
