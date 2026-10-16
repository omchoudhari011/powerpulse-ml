"""Tests for the RAG layer (src/rag). No network: the Anthropic client is replaced by a fake."""
import importlib.util
import io
import json
import os
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace

import pandas as pd

from src.errors import RAGError
from src.rag import ask
from src.rag.documents import discover_documents, load_chunks, read_blocks, split_text
from src.rag.forecast_context import build_forecast_context
from src.rag.index import KnowledgeIndex, ensure_index

HAS_ANTHROPIC = importlib.util.find_spec("anthropic") is not None

HVAC_DOC = """# HVAC manual

## Chiller schedule
The chiller starts at 06:00 on weekdays and causes the morning demand ramp.

## Lighting
LED retrofits cut lighting load by roughly a third.

```powershell
# not a heading: comment inside a code block
```
"""


ANSWER_JSON = json.dumps({
    "answer": "The chiller starts at 06:00 on weekdays [1].",
    "key_figures": [{"label": "Chiller start", "value": "06:00 on weekdays"}],
    "confidence": "high", "confidence_reason": "Stated directly in the HVAC manual.",
    "sources_used": [1], "caveats": ["Synthetic demo data."]})


class FakeClient:
    def __init__(self, stop_reason="end_turn", text=ANSWER_JSON):
        self.calls = []
        response = SimpleNamespace(stop_reason=stop_reason, model="claude-opus-5-5",
                                   content=[SimpleNamespace(type="text", text=text)],
                                   usage=SimpleNamespace(iterations=None))
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))
        self._response = response

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._response


class RagBase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.docs = self.root / "knowledge"
        self.docs.mkdir()
        (self.docs / "hvac.md").write_text(HVAC_DOC, encoding="utf-8")
        (self.docs / "tariff.txt").write_text("Peak tariff applies from 17:00 to 21:00 on weekdays.",
                                              encoding="utf-8")
        self.out = self.root / "outputs"
        self.out.mkdir()

    def tearDown(self):
        self.tmp.cleanup()

    def write_outputs(self):
        report = {"generated_at": "2024-01-01T00:00:00+00:00", "target_variable": "demand_kw", "unit": "kW",
                  "boundaries": {"test": {"start": "2023-12-01", "end": "2023-12-31"}},
                  "selection": {"criterion": "validation_rmse", "preferred_model": "mlp", "note": "Validation only."},
                  "models": {"mlp": {"validation": {"mae": 2.0, "rmse": 3.0, "wape_percent": 4.0},
                                     "test": {"mae": 2.5, "rmse": 3.5, "wape_percent": 4.5},
                                     "test_mae_by_horizon_step": [1.0, 2.0, 3.0],
                                     "test_peak_demand": {"available": True, "mean_abs_peak_error_kw": 5.0,
                                                          "mean_abs_peak_error_percent": 6.0,
                                                          "mean_signed_peak_error_kw": -1.0,
                                                          "mean_abs_peak_timing_error_steps": 1.5}}}}
        (self.out / "evaluation_metrics.json").write_text(json.dumps(report), encoding="utf-8")
        times = pd.date_range("2024-01-01", periods=4, freq="1h")
        pd.DataFrame({"timestamp": times, "actual_value": [None] * 4, "predicted_value": [10.0, 40.0, 20.0, 30.0],
                      "target_variable": "demand_kw", "unit": "kW", "model_name": "mlp",
                      "model_version": "mlp-test", "forecast_generated_at": "2024-01-01T00:00:00+00:00"}
                     ).to_csv(self.out / "forecast_mlp_20231231T2300.csv", index=False)


class TestDocuments(RagBase):
    def test_split_text_respects_word_limit(self):
        text = "\n\n".join(" ".join(["word"] * 30) for _ in range(5)) + "\n\n" + " ".join(["long"] * 130)
        chunks = split_text(text, max_words=60)
        self.assertTrue(all(len(c.split()) <= 60 for c in chunks))
        self.assertEqual(sum(len(c.split()) for c in chunks), 5 * 30 + 130)   # nothing lost

    def test_markdown_split_at_headings_but_not_inside_code(self):
        sections = [s for s, _ in read_blocks(self.docs / "hvac.md")]
        self.assertIn("Chiller schedule", sections)
        self.assertNotIn("not a heading: comment inside a code block", sections)

    def test_discover_skips_unsupported_and_hidden_files(self):
        (self.docs / "data.csv").write_text("a,b\n1,2", encoding="utf-8")
        (self.docs / ".cache").mkdir()
        (self.docs / ".cache" / "x.md").write_text("hidden", encoding="utf-8")
        names = [p.name for p in discover_documents([self.docs, self.root / "missing"])]
        self.assertEqual(names, ["hvac.md", "tariff.txt"])

    def test_chunks_carry_source_and_section(self):
        chunks = load_chunks([self.docs])
        chiller = [c for c in chunks if "chiller" in c.text.lower()]
        self.assertEqual(chiller[0].section, "Chiller schedule")
        self.assertTrue(chiller[0].label.endswith("hvac.md > Chiller schedule"))


class TestIndex(RagBase):
    def test_search_ranks_relevant_chunk_first(self):
        index = KnowledgeIndex.build([self.docs])
        self.assertIn("chiller", index.search("When does the chiller start?")[0].chunk.text.lower())
        self.assertIn("tariff", index.search("peak tariff hours")[0].chunk.text.lower())
        self.assertEqual(index.search("zzzz qqqq"), [])

    def test_empty_sources_raise(self):
        with self.assertRaises(RAGError):
            KnowledgeIndex.build([self.root / "nothing-here"])

    def test_ensure_index_reuses_then_rebuilds_on_change(self):
        idx_dir = self.root / "idx"
        _, rebuilt = ensure_index([self.docs], idx_dir)
        self.assertTrue(rebuilt)
        _, rebuilt = ensure_index([self.docs], idx_dir)
        self.assertFalse(rebuilt)
        (self.docs / "new.md").write_text("# Solar\nRooftop PV produces 40 kW at noon.", encoding="utf-8")
        index, rebuilt = ensure_index([self.docs], idx_dir)
        self.assertTrue(rebuilt)
        self.assertIn("pv", index.search("rooftop PV")[0].chunk.text.lower())

    def test_ensure_index_without_paths_keeps_saved_sources(self):
        idx_dir = self.root / "idx"
        ensure_index([self.docs / "tariff.txt"], idx_dir)
        index, rebuilt = ensure_index(None, idx_dir)
        self.assertFalse(rebuilt)
        self.assertEqual({c.source.split("/")[-1] for c in index.chunks}, {"tariff.txt"})
        later = os.stat(self.docs / "tariff.txt").st_mtime_ns + 10**9
        os.utime(self.docs / "tariff.txt", ns=(later, later))
        index, rebuilt = ensure_index(None, idx_dir)
        self.assertTrue(rebuilt)
        self.assertEqual({c.source.split("/")[-1] for c in index.chunks}, {"tariff.txt"})


class TestForecastContext(RagBase):
    def test_summary_reads_metrics_and_latest_forecast(self):
        self.write_outputs()
        text = build_forecast_context(self.out, self.root / "artifacts")
        self.assertIn("Preferred model: mlp", text)
        self.assertIn("MLP test: MAE 2.50 kW", text)
        self.assertIn("Peak 40.00 kW at 2024-01-01 01:00:00", text)
        self.assertIn("Implied energy over the period: 100 kWh", text)
        self.assertIn("Future forecast", text)

    def test_missing_outputs_are_reported_not_raised(self):
        text = build_forecast_context(self.out, self.root / "artifacts")
        self.assertIn("not available (run python -m src.evaluate)", text)
        self.assertIn("not available (run python -m src.predict)", text)


@unittest.skipUnless(HAS_ANTHROPIC, "anthropic is not installed")
class TestAsk(RagBase):
    def setUp(self):
        super().setUp()
        self.write_outputs()
        self.index = KnowledgeIndex.build([self.docs])

    def test_answer_sends_context_and_returns_structured_answer(self):
        client = FakeClient()
        answer = ask.answer_question("When does the chiller start?", self.index, client, k=2,
                                     output_dir=self.out, artifacts_dir=self.root / "artifacts")
        self.assertEqual(answer.data["confidence"], "high")
        call = client.calls[0]
        self.assertEqual(call["model"], ask.DEFAULT_MODEL)
        self.assertEqual(call["fallbacks"], "default")
        self.assertIn(ask.FALLBACK_BETA, call["betas"])
        self.assertEqual(call["output_config"]["format"]["schema"], ask.ANSWER_SCHEMA)
        text = ask.format_answer(answer)
        for heading in ("QUESTION", "ANSWER", "KEY FIGURES", "CONFIDENCE: HIGH", "CAVEATS", "SOURCES"):
            self.assertIn(heading, text)
        self.assertIn("[1] ", text)
        as_dict = answer.to_dict()
        self.assertEqual(as_dict["sources"][0]["index"], 1)
        self.assertEqual(as_dict["key_figures"][0]["label"], "Chiller start")
        content = call["messages"][0]["content"]
        self.assertIn("<forecast_summary>", content)
        self.assertIn("Chiller schedule", content)
        self.assertTrue(content.rstrip().endswith("Question: When does the chiller start?"))

    def test_refusal_truncation_and_bad_json_raise(self):
        for client in (FakeClient(stop_reason="refusal"), FakeClient(stop_reason="max_tokens"),
                       FakeClient(text="not json")):
            with self.assertRaises(RAGError):
                ask.answer_question("anything", self.index, client, output_dir=self.out)

    def test_context_prefers_forecast_report(self):
        from src.report import save_report
        report = {"report_type": "powerpulse_forecast", "unit": "kW", "target_variable": "demand_kw",
                  "generated_at": "now", "model": {"name": "ensemble", "version": "ensemble-x", "components": [
                      {"name": "lstm", "version": "a", "weight": 0.4}, {"name": "mlp", "version": "b", "weight": 0.6}]},
                  "forecast": {"origin": "o", "start": "s", "end": "e", "steps": 1, "interval_minutes": 60.0,
                               "kind": "future"},
                  "summary": {"peak": {"value": 9.0, "timestamp": "t"}, "low": {"value": 1.0, "timestamp": "t"},
                              "mean": 5.0, "total_energy_kwh": 5.0},
                  "prediction_interval": None, "accuracy": None, "backtest": None,
                  "values": [{"timestamp": "t", "predicted": 9.0, "lower": None, "upper": None, "lstm": 8.0,
                              "mlp": 9.6, "actual": None}], "caveats": ["c"]}
        save_report(report, self.out, "forecast_report_ensemble_x")
        text = build_forecast_context(self.out, self.root / "artifacts")
        self.assertIn("0.40 x LSTM + 0.60 x MLP", text)
        self.assertIn("forecast_report_ensemble_x.json", text)

    def test_retrieve_only_makes_no_call_and_empty_question_fails(self):
        client = FakeClient()
        answer = ask.answer_question("chiller", self.index, client, output_dir=self.out, retrieve_only=True)
        self.assertEqual(client.calls, [])
        self.assertIn("Question: chiller", answer.context)
        with self.assertRaises(RAGError):
            ask.answer_question("   ", self.index, client, output_dir=self.out)

    def test_cli_retrieve_only(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            code = ask.main(["How do I train the LSTM?", "--retrieve-only", "--index-dir", str(self.root / "idx"),
                             "--output-dir", str(self.out)])
        self.assertEqual(code, 0)
        self.assertIn("SOURCES", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
