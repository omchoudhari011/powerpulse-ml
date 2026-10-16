"""Generate a DUMMY (synthetic) building-energy CSV for demos and smoke-testing PowerPulse ML.

THIS DATA IS NOT REAL. It simulates a mid-size office building so you can run the whole pipeline
(validate -> train -> evaluate -> predict) end to end. Any accuracy measured on it says nothing about
real buildings. Replace it with your own CSV for real work.

Usage (from the project root):
    python tools/generate_dummy_dataset.py                       # writes data/dummy_building.csv
    python tools/generate_dummy_dataset.py --start 2023-01-01 --days 730 --seed 7 --out data/other.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

# Fixed public-holiday style dates (month, day) when the office is closed.
HOLIDAYS = {(1, 1), (1, 26), (3, 8), (8, 15), (10, 2), (11, 1), (12, 25)}


def _ar1(rng, n, phi, sigma):
    out = np.zeros(n)
    eps = rng.normal(0, sigma, n)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + eps[i]
    return out


def generate(start="2023-01-01", days=730, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=days * 24, freq="h")
    n = len(idx)
    hour = idx.hour.to_numpy()
    dow = idx.dayofweek.to_numpy()
    doy = idx.dayofyear.to_numpy()
    weekday = dow < 5
    holiday = np.array([(d.month, d.day) in HOLIDAYS for d in idx])
    workday = weekday & ~holiday

    # ---- weather (Celsius): seasonal + diurnal + slowly varying noise
    seasonal = 17 + 11 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    diurnal = 5 * np.sin(2 * np.pi * (hour - 9) / 24)
    temperature = seasonal + diurnal + _ar1(rng, n, 0.97, 0.6)
    humidity = 62 - 1.1 * (temperature - 17) + _ar1(rng, n, 0.95, 2.0) + rng.normal(0, 1.5, n)
    humidity = np.clip(humidity, 15, 98)

    # ---- occupancy (people in building)
    ramp = np.clip((hour - 7) / 2.0, 0, 1) * np.clip((19 - hour) / 2.0, 0, 1)   # smooth 7-9 up, 17-19 down
    lunch_dip = 1 - 0.18 * np.exp(-((hour - 13) ** 2) / 1.5)
    day_scale = np.where(workday, 1.0, np.where(weekday | holiday, 0.04, 0.06))
    day_scale = day_scale * np.where(dow == 4, 0.85, 1.0) * np.where(dow == 0, 0.95, 1.0)
    occ = 320 * ramp * lunch_dip * day_scale + rng.normal(0, 8, n) * (ramp > 0)
    cleaners = np.where((hour >= 19) & (hour < 22) & workday, 6, 0)
    occupancy = np.clip(np.round(occ + cleaners + rng.integers(0, 4, n)), 0, None).astype(int)

    # ---- equipment (HVAC plant) status: on during business hours, a few random outages
    equip = ((hour >= 6) & (hour < 20) & workday) | ((hour >= 8) & (hour < 14) & (dow == 5))
    cold_snap_or_heat = (temperature > 30) | (temperature < 3)          # plant runs longer in extremes
    equip = equip | (cold_snap_or_heat & (hour >= 5) & (hour < 22) & ~holiday)
    outage = np.zeros(n, dtype=bool)
    for s in rng.choice(n - 12, size=max(3, days // 90), replace=False):
        outage[s:s + rng.integers(2, 8)] = True
    equip = equip & ~outage
    equipment_status = np.where(equip, "on", "off")

    # ---- electrical demand (kW)
    base_load = 38 + 4 * np.sin(2 * np.pi * (hour - 15) / 24)           # servers, security, standby
    lighting = 0.06 * occupancy * (1 + 0.3 * (hour >= 17) * (hour < 20))
    plug = 0.11 * occupancy
    cooling = np.clip(temperature - 22, 0, None) * 4.2 * (1 + 0.012 * (humidity - 55)) * equip
    heating = np.clip(14 - temperature, 0, None) * 2.4 * equip
    fan_pump = 14 * equip
    startup = 12 * ((hour == 6) & equip)                                  # morning pre-cool/heat spike
    trend = 1 + 0.03 * np.arange(n) / n                                   # slow load growth
    demand = (base_load + lighting + plug + cooling + heating + fan_pump + startup) * trend
    demand = demand * (1 + _ar1(rng, n, 0.6, 0.015)) + rng.normal(0, 1.2, n)
    demand = np.clip(demand, 5, None)

    # energy over the hour (kWh) ~= average power over the hour, with metering noise
    energy = np.clip(demand * (1 + rng.normal(0, 0.01, n)), 0, None)

    return pd.DataFrame({
        "timestamp": idx.strftime("%Y-%m-%d %H:%M:%S"),
        "energy_kwh": energy.round(2),
        "demand_kw": demand.round(2),
        "temperature": temperature.round(1),
        "humidity": humidity.round(1),
        "occupancy": occupancy,
        "equipment_status": equipment_status,
    })


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--days", type=int, default=730)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(ROOT / "data" / "dummy_building.csv"))
    a = ap.parse_args(argv)
    df = generate(a.start, a.days, a.seed)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} hourly rows ({df.timestamp.iloc[0]} -> {df.timestamp.iloc[-1]}) to {out}")
    print("NOTE: this is DUMMY data for demos/smoke tests, not a real building.")


if __name__ == "__main__":
    main()
