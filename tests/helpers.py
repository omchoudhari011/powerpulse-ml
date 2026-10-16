"""TEST FIXTURES ONLY.

The synthetic frames below exist solely so unit tests can exercise the code. They are never used for
training or reporting accuracy, and the application never generates data.
"""
import numpy as np
import pandas as pd


def make_frame(n_hours=24 * 60, seed=0, step="1h", demand=True, energy=False, weather=True,
               occupancy=True, equipment=True, start="2023-01-02 00:00"):
    rng = np.random.default_rng(seed)
    idx = pd.date_range(start, periods=n_hours, freq=step)
    hod = idx.hour + idx.minute / 60.0
    base = 50 + 20 * np.sin(2 * np.pi * (hod - 6) / 24) + np.where(idx.dayofweek >= 5, -10, 5)
    df = pd.DataFrame({"timestamp": idx})
    if demand:
        df["demand_kw"] = base + rng.normal(0, 2, len(idx))
    if energy:
        df["energy_kwh"] = (base + rng.normal(0, 2, len(idx))) * 0.5
    if weather:
        df["temperature"] = 15 + 8 * np.sin(2 * np.pi * (hod - 9) / 24) + rng.normal(0, 1, len(idx))
        df["humidity"] = np.clip(60 + rng.normal(0, 5, len(idx)), 0, 100)
    if occupancy:
        df["occupancy"] = np.where((hod >= 8) & (hod < 18) & (idx.dayofweek < 5), 100, 5) + rng.integers(0, 5, len(idx))
    if equipment:
        df["equipment_status"] = np.where((hod >= 7) & (hod < 19), "on", "off")
    return df


def write_csv(df, path):
    df.to_csv(path, index=False)
    return str(path)
