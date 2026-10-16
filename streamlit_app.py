"""PowerPulse web UI (Streamlit). Run from the project root:

    streamlit run streamlit_app.py

Tabs: Forecast (run and view the 24 h forecast), Models (LSTM / MLP / ensemble comparison),
Data & training (upload, validate, train, evaluate). Everything calls the same functions as the
command-line tools.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import streamlit as st

from src.config import ARTIFACTS_DIR, DATA_DIR, MODEL_NAMES, OUTPUTS_DIR, PROJECT_ROOT, Config
from src.errors import PowerPulseError
from src.report import latest_report, render_report
from src.ui import charts

st.set_page_config(page_title="Forecast Agent", layout="wide")

MODE = "dark" if getattr(st.context.theme, "type", None) == "dark" else "light"
UPLOAD_DIR = OUTPUTS_DIR / "uploads"
MODEL_OPTIONS = {"Best (chosen by evaluation)": "best", "Ensemble (LSTM + MLP)": "ensemble",
                 "LSTM": "lstm", "MLP": "mlp"}
ERRORS = (PowerPulseError, FileNotFoundError, ImportError)


# ------------------------------------------------------------------ helpers
def load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None
    except (json.JSONDecodeError, OSError):
        return None


def evaluation() -> Optional[Dict[str, Any]]:
    return load_json(OUTPUTS_DIR / "evaluation_metrics.json")


def display(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def csv_choices() -> List[str]:
    files = sorted(DATA_DIR.glob("*.csv")) + sorted(UPLOAD_DIR.glob("*.csv"))
    return [display(f) for f in files]


def training_csv() -> Optional[Path]:
    """The CSV the current models were trained on (evaluation must use exactly that file)."""
    for name in MODEL_NAMES:
        meta = load_json(ARTIFACTS_DIR / name / "metadata.json")
        if meta and meta.get("csv_path"):
            p = Path(meta["csv_path"].replace("\\", "/"))
            return p if p.is_absolute() else PROJECT_ROOT / p
    return None


def save_upload(upload) -> Path:
    name = Path(upload.name).name
    if not name.lower().endswith(".csv"):
        raise PowerPulseError("Only .csv files can be uploaded.")
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    path = UPLOAD_DIR / name
    path.write_bytes(upload.getvalue())
    return path


def current_report() -> Optional[Dict[str, Any]]:
    if "report" not in st.session_state:
        path = latest_report(OUTPUTS_DIR)
        st.session_state["report"] = load_json(path) if path else None
    return st.session_state["report"]


def run_forecast(model_key: str, csv: str, origin: Optional[str]) -> Dict[str, Any]:
    from src.predict import generate_forecast, resolve_best_model
    name = resolve_best_model(OUTPUTS_DIR) if model_key == "best" else model_key
    _, report = generate_forecast(name, PROJECT_ROOT / csv, origin=origin or None, return_report=True)
    st.session_state["report"] = report
    return report


def fmt(v, digits: int = 1, unit: str = "") -> str:
    return "n/a" if v is None else f"{v:,.{digits}f}{(' ' + unit) if unit else ''}"


# ------------------------------------------------------------------ forecast tab
def render_forecast(report: Dict[str, Any]) -> None:
    u, m, fc, s = report["unit"], report["model"], report["forecast"], report["summary"]
    blend = " + ".join(f"{c['weight']:.2f} × {c['name'].upper()}" for c in m["components"])
    st.subheader(f"{fc['start']} → {fc['end']}")
    st.caption(f"{charts.MODEL_LABELS[m['name']]} `{m['version']}`"
               + (f" = {blend}" if len(m["components"]) > 1 else "")
               + f" · {fc['kind']} forecast from {fc['origin']} · {fc['steps']} steps of "
                 f"{fmt(fc['interval_minutes'], 0)} min")

    acc, pi = report.get("accuracy"), report.get("prediction_interval")
    k = st.columns(4)
    k[0].metric("Peak demand", fmt(s["peak"]["value"], 1, u), border=True)
    k[0].caption(f"at {s['peak']['timestamp']}")
    k[1].metric("Lowest", fmt(s["low"]["value"], 1, u), border=True)
    k[1].caption(f"at {s['low']['timestamp']}")
    if s.get("total_energy_kwh") is not None:
        k[2].metric("Energy over the period", fmt(s["total_energy_kwh"], 0, "kWh"), border=True)
        k[2].caption(f"mean {fmt(s['mean'], 1, u)}")
    else:
        k[2].metric("Mean", fmt(s["mean"], 1, u), border=True)
    if report.get("backtest"):
        k[3].metric("Backtest error (MAE)", fmt(report["backtest"]["mae"], 2, u), border=True)
        k[3].caption(f"{report['backtest']['n_actual']} actual readings")
    else:
        k[3].metric("Typical error (test MAE)", fmt(acc["test"]["mae"], 2, u) if acc else "n/a", border=True)
        k[3].caption((f"WAPE {fmt(acc['test']['wape_percent'], 1)}%" if acc else "run evaluation")
                     + (f" · {pi['level_percent']}% range held {fmt(pi['test_coverage_percent'], 1)}% of test actuals"
                        if pi and pi.get("test_coverage_percent") is not None else ""))

    st.altair_chart(charts.forecast_chart(report, MODE), width="stretch")
    if pi:
        st.caption(f"Shaded band: {pi['level_percent']}% forecast range ({pi['method']}).")

    with st.expander("Table view"):
        st.dataframe(charts.forecast_frame(report), hide_index=True, width="stretch",
                     column_config={"timestamp": st.column_config.DatetimeColumn("Time", format="YYYY-MM-DD HH:mm")})
    d = st.columns(3)
    tag = f"{m['name']}_{fc['origin'].replace('-', '').replace(':', '').replace(' ', 'T')}"
    d[0].download_button("Download report (JSON)", json.dumps(report, indent=2), f"forecast_report_{tag}.json",
                         "application/json", width="stretch")
    d[1].download_button("Download report (Markdown)", render_report(report), f"forecast_report_{tag}.md",
                         "text/markdown", width="stretch")
    d[2].download_button("Download values (CSV)", charts.forecast_frame(report).to_csv(index=False),
                         f"forecast_{tag}.csv", "text/csv", width="stretch")
    for note in report.get("caveats", []):
        st.caption(f"• {note}")


def forecast_tab() -> None:
    choices = csv_choices()
    c = st.columns([1.3, 1.7, 1.3, 0.9], vertical_alignment="bottom")
    model_label = c[0].selectbox("Model", list(MODEL_OPTIONS))
    trained = training_csv()
    default = choices.index(display(trained)) if trained and display(trained) in choices else 0
    csv = c[1].selectbox("History CSV", choices, index=default if choices else None,
                         help="Needs at least 7 days of recent readings. Upload files in the Data & training tab.")
    origin = c[2].text_input("Forecast origin (optional)", placeholder="last timestamp in the CSV",
                             help="An earlier timestamp, e.g. 2024-11-15 00:00, runs a backtest against actuals.")
    if c[3].button("Run forecast", type="primary", width="stretch", disabled=not choices):
        with st.spinner("Running the models..."):
            try:
                run_forecast(MODEL_OPTIONS[model_label], csv, origin.strip())
            except ERRORS as exc:
                st.error(str(exc))
    report = current_report()
    if report is None:
        st.info("No forecast yet. Train the models in the Data & training tab, then press Run forecast.")
    else:
        render_forecast(report)


# ------------------------------------------------------------------ models tab
def models_tab() -> None:
    ev = evaluation()
    csv = training_csv()
    if st.button("Re-run evaluation", disabled=csv is None,
                 help=f"Scores every model on the held-out test period of {display(csv) if csv else 'the training CSV'}."):
        with st.spinner("Evaluating LSTM, MLP and the ensemble..."):
            try:
                from src.evaluate import run_evaluation
                ev = run_evaluation(csv)
                st.success("Evaluation updated. Run a new forecast to use the refitted ensemble.")
            except ERRORS as exc:
                st.error(str(exc))
    if not ev:
        st.info("No evaluation yet. Train both models, then press Re-run evaluation.")
        return

    sel, u = ev["selection"], ev["unit"]
    test = ev["boundaries"]["test"]
    best = (sel.get("preferred_model") or "none").upper()
    st.markdown(f"**Preferred model: {best}** — chosen by {sel['criterion'].replace('_', ' ')}. "
                f"Test period {test['start']} → {test['end']}.")
    if sel.get("test_ranking_agrees") is False:
        st.warning("The model that wins on validation does not win on the test period.")

    rows = []
    for name, r in ev["models"].items():
        peak = r.get("test_peak_demand") or {}
        rows.append({"Model": charts.MODEL_LABELS.get(name, name.upper()),
                     f"Validation MAE ({u})": r["validation"]["mae"], f"Validation RMSE ({u})": r["validation"]["rmse"],
                     f"Test MAE ({u})": r["test"]["mae"], f"Test RMSE ({u})": r["test"]["rmse"],
                     "Test WAPE (%)": r["test"]["wape_percent"],
                     f"Daily peak error ({u})": peak.get("mean_abs_peak_error_kw") if peak.get("available") else None})
    num = st.column_config.NumberColumn(format="%.2f")
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={c: num for c in rows[0] if c != "Model"})

    ens = ev["models"].get("ensemble")
    if ens:
        pi = ens["prediction_interval"]
        st.caption("Ensemble = " + " + ".join(f"{w:.2f} × {m.upper()}" for m, w in ens["weights"].items())
                   + f" (weights fitted on the validation period). Its {pi['level_percent']}% range held "
                     f"{pi['test_coverage_percent']:.1f}% of test actuals.")

    st.subheader("Error by forecast lead time")
    chart = charts.mae_by_horizon_chart(ev, MODE)
    if chart is not None:
        st.altair_chart(chart, width="stretch")
        with st.expander("Table view"):
            st.dataframe(charts.horizon_frame(ev), hide_index=True, width="stretch",
                         column_config={c: num for c in charts.SERIES_ORDER})
    plots = sorted((OUTPUTS_DIR / "plots").glob("*.png"))
    if plots:
        with st.expander("Evaluation plots"):
            for p in plots:
                st.image(str(p), caption=p.stem.replace("_", " "))


# ------------------------------------------------------------------ data & training tab
def data_tab() -> None:
    st.subheader("1 · Choose data")
    upload = st.file_uploader("Upload a building-energy CSV", type="csv",
                              help="Columns: timestamp plus demand_kw or energy_kwh; temperature, humidity, "
                                   "occupancy and equipment_status are optional. See data/README.md.")
    if upload is not None:
        try:
            st.success(f"Saved to {display(save_upload(upload))}")
        except ERRORS as exc:
            st.error(str(exc))
    choices = csv_choices()
    if not choices:
        st.info("No CSV files in data/ yet; upload one above.")
        return
    csv = st.selectbox("Dataset", choices, key="data_csv")
    path = PROJECT_ROOT / csv

    st.subheader("2 · Validate")
    if st.button("Validate dataset"):
        try:
            from src.data_loader import load_csv
            from src.data_validation import validate_dataframe
            cfg = Config()
            df, info = load_csv(path, cfg.data)
            rep = validate_dataframe(df, info, cfg)
            (st.success if rep.ok else st.error)("Validation passed." if rep.ok else "Validation failed.")
            st.code(rep.summary(), language=None)
        except ERRORS as exc:
            st.error(str(exc))

    st.subheader("3 · Train, evaluate and forecast")
    epochs = st.number_input("Maximum epochs", min_value=5, max_value=500, value=100, step=5,
                             help="Early stopping usually ends training well before this.")
    st.caption("Trains the LSTM and the MLP, fits the ensemble on the validation period, scores all three on "
               "the test period and forecasts the 24 h after the last timestamp. Takes a few minutes on a CPU.")
    if st.button("Run full pipeline", type="primary"):
        with st.status("Running the pipeline...", expanded=True) as status:
            try:
                from src.evaluate import run_evaluation
                from src.pipeline import load_and_prepare
                from src.train import train_model
                cfg = Config()
                cfg.train.max_epochs = int(epochs)
                st.write("Preparing data...")
                prepared = load_and_prepare(path, cfg)
                for w in prepared.warnings:
                    st.warning(w)
                for name in MODEL_NAMES:
                    st.write(f"Training {name.upper()}...")
                    res = train_model(name, path, cfg, verbose=0, prepared=prepared)
                    st.write(f"{name.upper()}: {res['epochs_run']} epochs, validation MAE "
                             f"{res['validation_metrics']['mae']:.2f} {prepared.unit}")
                st.write("Evaluating LSTM, MLP and the ensemble...")
                ev = run_evaluation(path)
                st.write(f"Preferred model: {ev['selection']['preferred_model']}")
                st.write("Forecasting...")
                run_forecast("best", csv, None)
                status.update(label="Pipeline finished. See the Forecast and Models tabs.", state="complete")
            except ERRORS as exc:
                status.update(label="Pipeline failed.", state="error")
                st.error(str(exc))


# ------------------------------------------------------------------ page
st.title("Forecast Agent", anchor=False)
st.caption("Building energy demand forecasting with an LSTM, an MLP and their weighted ensemble.")

forecast, models, data = st.tabs(["Forecast", "Models", "Data & training"])
with forecast:
    forecast_tab()
with models:
    models_tab()
with data:
    data_tab()
