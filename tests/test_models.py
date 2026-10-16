"""Real Keras tests. Skipped automatically when TensorFlow is not installed."""
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src import evaluate, predict, train
from src.config import Config
from tests.helpers import make_frame, write_csv

HAS_TF = importlib.util.find_spec("tensorflow") is not None


@unittest.skipUnless(HAS_TF, "TensorFlow is not installed")
class TestKerasModels(unittest.TestCase):
    def test_lstm_output_shape_and_config(self):
        from src.models.lstm_model import build_lstm_model
        for layers in (1, 2):
            m = build_lstm_model(24, 11, 24, units=8, num_layers=layers, dropout=0.2)
            self.assertEqual(m.output_shape, (None, 24))
            self.assertEqual(m.predict(np.zeros((3, 24, 11), np.float32), verbose=0).shape, (3, 24))
            self.assertEqual(len([l for l in m.layers if l.__class__.__name__ == "LSTM"]), layers)
        with self.assertRaises(ValueError):
            build_lstm_model(24, 11, 24, num_layers=3)

    def test_mlp_output_shape(self):
        from src.models.mlp_model import build_mlp_model
        m = build_mlp_model(20, 24, hidden_units=(128, 64))
        self.assertEqual(m.output_shape, (None, 24))
        self.assertEqual([l.units for l in m.layers if l.__class__.__name__ == "Dense"], [128, 64, 24])

    def test_tiny_end_to_end_train_evaluate_predict(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            csv = write_csv(make_frame(24 * 45), root / "x.csv")
            cfg = Config()
            cfg.train.max_epochs, cfg.lstm.units, cfg.mlp.hidden_units = 2, 8, (16, 8)
            for name in ("lstm", "mlp"):
                train.train_model(name, csv, cfg, root / "art", verbose=0)
            rep = evaluate.run_evaluation(csv, ("lstm", "mlp"), root / "art", root / "out")
            self.assertIn("preferred_model", rep["selection"])
            out = predict.generate_forecast("lstm", csv, root / "art", root / "out")
            self.assertEqual(len(out), 24)
            self.assertTrue(np.isfinite(out["predicted_value"]).all())


if __name__ == "__main__":
    unittest.main()
