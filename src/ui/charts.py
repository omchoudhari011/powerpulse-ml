"""Altair charts for the Streamlit UI.

Colours follow the entity in every chart (Ensemble = blue, LSTM = orange, MLP = aqua, Actual = ink),
using the first three slots of a validated categorical palette, which pass colour-blind separation
checks pairwise in both light and dark mode. Each chart has a legend, a hover crosshair whose tooltip
lists every series at that time, and a table view in the app (aqua is below 3:1 contrast on white).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import altair as alt
import pandas as pd

PALETTES = {
    "light": {"Ensemble": "#2a78d6", "LSTM": "#eb6834", "MLP": "#1baf7a", "Actual": "#0b0b0b",
              "rule": "#898781", "surface": "#ffffff"},
    "dark": {"Ensemble": "#3987e5", "LSTM": "#d95926", "MLP": "#199e70", "Actual": "#f0efec",
             "rule": "#898781", "surface": "#0e1117"},
}
SERIES_ORDER = ["Ensemble", "LSTM", "MLP", "Actual"]
MODEL_LABELS = {"ensemble": "Ensemble", "lstm": "LSTM", "mlp": "MLP"}
HEIGHT = 360


def _palette(mode: str) -> Dict[str, str]:
    return PALETTES["dark" if mode == "dark" else "light"]


def _color(domain: List[str], pal: Dict[str, str]) -> alt.Color:
    return alt.Color("series:N", scale=alt.Scale(domain=domain, range=[pal[s] for s in domain]),
                     legend=alt.Legend(title=None, orient="top", symbolType="stroke", symbolStrokeWidth=2))


def _lines_with_crosshair(wide: pd.DataFrame, x: str, x_enc: alt.X, series: List[str], y_title: str,
                          pal: Dict[str, str], tooltip: List[alt.Tooltip]) -> List[alt.Chart]:
    """2px lines + a nearest-x hairline whose tooltip lists every series, + hover dots with a surface ring."""
    long = wide.melt(id_vars=[x], value_vars=series, var_name="series", value_name="value").dropna(subset=["value"])
    y_enc = alt.Y("value:Q", title=y_title, scale=alt.Scale(zero=True))
    color = _color(series, pal)
    hover = alt.selection_point(nearest=True, on="pointerover", fields=[x], empty=False, clear="pointerout")
    lines = alt.Chart(long).mark_line(strokeWidth=2, strokeCap="round", strokeJoin="round").encode(x_enc, y_enc, color)
    rule = (alt.Chart(wide).mark_rule(color=pal["rule"], strokeWidth=1)
            .encode(x_enc, opacity=alt.condition(hover, alt.value(1), alt.value(0)), tooltip=tooltip)
            .add_params(hover))
    dots = (alt.Chart(long).mark_point(filled=True, size=72, stroke=pal["surface"], strokeWidth=2, opacity=1)
            .encode(x_enc, y_enc, color).transform_filter(hover))
    return [lines, rule, dots]


def forecast_frame(report: Dict[str, Any]) -> pd.DataFrame:
    """Wide table of the report's values with display names as columns."""
    wide = pd.DataFrame(report["values"])
    wide["timestamp"] = pd.to_datetime(wide["timestamp"])
    main = MODEL_LABELS.get(report["model"]["name"], report["model"]["name"].upper())
    out = pd.DataFrame({"timestamp": wide["timestamp"], main: wide["predicted"]})
    if len(report["model"]["components"]) > 1:
        out["LSTM"], out["MLP"] = wide["lstm"], wide["mlp"]
    if wide["actual"].notna().any():
        out["Actual"] = wide["actual"]
    if wide["lower"].notna().any():
        out["Range low"], out["Range high"] = wide["lower"], wide["upper"]
    return out


def forecast_chart(report: Dict[str, Any], mode: str = "light") -> alt.LayerChart:
    pal, unit = _palette(mode), report["unit"]
    wide = forecast_frame(report)
    series = [s for s in SERIES_ORDER if s in wide.columns]
    main = MODEL_LABELS[report["model"]["name"]]
    one_day = wide["timestamp"].dt.date.nunique() == 1
    x_enc = alt.X("timestamp:T", title=None,
                  axis=alt.Axis(format="%H:%M" if one_day else "%d %b %H:%M", labelAngle=0, grid=False,
                                labelOverlap="greedy", labelSeparation=12))
    tooltip = [alt.Tooltip("timestamp:T", title="Time", format="%Y-%m-%d %H:%M")]
    tooltip += [alt.Tooltip(f"{s}:Q", title=f"{s} ({unit})", format=".1f") for s in series]
    layers: List[alt.Chart] = []
    if "Range low" in wide.columns:
        level = (report.get("prediction_interval") or {}).get("level_percent", 80)
        tooltip += [alt.Tooltip("Range low:Q", title=f"{level}% range low", format=".1f"),
                    alt.Tooltip("Range high:Q", title=f"{level}% range high", format=".1f")]
        layers.append(alt.Chart(wide).mark_area(opacity=0.18 if mode == "dark" else 0.12, color=pal[main])
                      .encode(x_enc, alt.Y("Range low:Q", title=unit), alt.Y2("Range high:Q")))
    layers += _lines_with_crosshair(wide, "timestamp", x_enc, series, unit, pal, tooltip)

    peak = wide.loc[wide[main].idxmax()]            # label the extreme only, in text ink
    peak_df = pd.DataFrame({"timestamp": [peak["timestamp"]], "value": [peak[main]],
                            "label": [f"Forecast peak {peak[main]:.1f} {unit}"]})
    layers.append(alt.Chart(peak_df).mark_point(filled=True, size=72, color=pal[main], stroke=pal["surface"],
                                                strokeWidth=2, opacity=1)
                  .encode(x_enc, alt.Y("value:Q")))
    # surface-coloured halo under the label keeps it legible where a line passes behind it
    label = dict(dy=-16, fontWeight="bold", fontSize=12)
    layers.append(alt.Chart(peak_df).mark_text(**label, stroke=pal["surface"], strokeWidth=5, strokeJoin="round",
                                               color=pal["surface"]).encode(x_enc, alt.Y("value:Q"), text="label:N"))
    layers.append(alt.Chart(peak_df).mark_text(**label).encode(x_enc, alt.Y("value:Q"), text="label:N"))
    return alt.layer(*layers).properties(height=HEIGHT)


def horizon_frame(evaluation: Dict[str, Any]) -> Optional[pd.DataFrame]:
    """Wide table: lead time (hours) x model test MAE."""
    hours = evaluation.get("steps", {}).get("interval_seconds", 3600) / 3600.0
    cols = {MODEL_LABELS.get(m, m.upper()): r.get("test_mae_by_horizon_step") or []
            for m, r in evaluation.get("models", {}).items()}
    cols = {k: v for k, v in cols.items() if v}
    if not cols:
        return None
    n = max(len(v) for v in cols.values())
    out = pd.DataFrame({"Lead time (h)": [round((i + 1) * hours, 2) for i in range(n)]})
    for name, values in cols.items():
        out[name] = pd.Series(values)
    return out


def mae_by_horizon_chart(evaluation: Dict[str, Any], mode: str = "light") -> Optional[alt.LayerChart]:
    wide = horizon_frame(evaluation)
    if wide is None:
        return None
    pal, unit = _palette(mode), evaluation.get("unit", "")
    series = [s for s in SERIES_ORDER if s in wide.columns]
    x = "Lead time (h)"
    x_enc = alt.X(f"{x}:Q", title="Forecast lead time (hours)", scale=alt.Scale(zero=False, nice=False),
                  axis=alt.Axis(grid=False, tickMinStep=1))
    tooltip = [alt.Tooltip(f"{x}:Q", title="Lead time (h)")]
    tooltip += [alt.Tooltip(f"{s}:Q", title=f"{s} MAE ({unit})", format=".2f") for s in series]
    return alt.layer(*_lines_with_crosshair(wide, x, x_enc, series, f"Test MAE ({unit})", pal, tooltip)
                     ).properties(height=300)
