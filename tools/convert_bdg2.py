"""Convert one building from the Building Data Genome Project 2 (BDG2) into a PowerPulse CSV.

BDG2: hourly meter readings from 1,636 real non-residential buildings, 2016-2017.
    https://github.com/buds-lab/building-data-genome-project-2
    Miller et al. (2020), Scientific Data 7:368, https://doi.org/10.1038/s41597-020-00687-2
    License: CC BY-SA 4.0 - credit the source and share derived files under the same license.

    # 1. download the three source files (~195 MB, checked against the repository's SHA-256 values)
    python tools\\convert_bdg2.py --download downloads\\bdg2
    # 2. list office buildings with complete data
    python tools\\convert_bdg2.py --source downloads\\bdg2 --list
    # 3. convert one building
    python tools\\convert_bdg2.py --source downloads\\bdg2 --building Hog_office_Lizzie --out data\\bdg2_hog_office_lizzie.csv

Output columns (all readings are real except humidity, which is derived):
    timestamp     local time, hourly (meters and weather both use the site's local time)
    energy_kwh    BDG2 cleaned electricity reading for that hour (kWh)
    demand_kw     average demand over the hour = energy_kwh / 1 h
    temperature   outdoor air temperature (C) from the site's weather station
    humidity      relative humidity (%) computed from air and dew-point temperature (Magnus formula)
Missing hours are left empty; src.data_validation reports them and forward-fills short gaps only.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

BASE_URL = "https://media.githubusercontent.com/media/buds-lab/building-data-genome-project-2/master/data"
FILES = {  # file -> (path in the repository, SHA-256 published in the repository's Git LFS pointer)
    "metadata.csv": ("metadata/metadata.csv", "992d0b29f24f96ad4332bc4dbb534b7bdd7dd2689aad093f94e93068ecddca02"),
    "weather.csv": ("weather/weather.csv", "a8189f1c6acdf3b9933a9e6354b8e7c1278cd56a7075929623a17565d44f04bd"),
    "electricity_cleaned.csv": ("meters/cleaned/electricity_cleaned.csv",
                                "b6ffc9b4dfcefe5c753594730a08ae822b0d50fec6815abb8f185591e6c630a3"),
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, (repo_path, expected) in FILES.items():
        target = directory / name
        if target.exists() and sha256(target) == expected:
            print(f"ok        {target}")
            continue
        print(f"download  {BASE_URL}/{repo_path}")
        urllib.request.urlretrieve(f"{BASE_URL}/{repo_path}", target)
        if sha256(target) != expected:
            target.unlink()
            raise SystemExit(f"ERROR: checksum mismatch for {name}; the download was removed.")
    print(f"Source files ready in {directory}")


def relative_humidity(temp_c: pd.Series, dew_c: pd.Series) -> pd.Series:
    """Magnus formula (Alduchov & Eskridge 1996 constants), clipped to 0-100 %."""
    a, b = 17.625, 243.04
    rh = 100 * np.exp(a * dew_c / (b + dew_c)) / np.exp(a * temp_c / (b + temp_c))
    return rh.clip(0, 100)


def longest_flat_run(s: pd.Series) -> int:
    s = s.dropna()
    return int(s.groupby((s.diff() != 0).cumsum()).size().max()) if len(s) else 0


def list_offices(source: Path, limit: int) -> None:
    meta = pd.read_csv(source / "metadata.csv")
    el = pd.read_csv(source / "electricity_cleaned.csv", parse_dates=["timestamp"]).set_index("timestamp")
    offices = meta[(meta.primaryspaceusage == "Office") & meta.building_id.isin(el.columns)]
    rows = []
    for _, m in offices.iterrows():
        s = el[m.building_id]
        rows.append({"building_id": m.building_id, "site": m.site_id, "timezone": m.timezone, "sqm": m.sqm,
                     "missing_%": round(s.isna().mean() * 100, 2), "zero_%": round((s == 0).mean() * 100, 2),
                     "mean_kwh": round(s.mean(), 1), "longest_flat_h": longest_flat_run(s)})
    table = pd.DataFrame(rows)
    # complete, non-zero, and no long stretches of a repeated value (a sign of filled gaps)
    good = table[(table["missing_%"] < 0.5) & (table["zero_%"] < 0.5) & (table["longest_flat_h"] <= 24)]
    print(f"{len(good)} of {len(table)} office buildings have complete data without long flat stretches:\n")
    print(good.sort_values(["missing_%", "longest_flat_h"]).head(limit).to_string(index=False))


def convert(source: Path, building: str, out: Path) -> pd.DataFrame:
    meta = pd.read_csv(source / "metadata.csv").set_index("building_id")
    if building not in meta.index:
        raise SystemExit(f"ERROR: unknown building_id {building!r}; use --list to see options.")
    site = meta.loc[building, "site_id"]
    el = pd.read_csv(source / "electricity_cleaned.csv", usecols=lambda c: c in ("timestamp", building),
                     parse_dates=["timestamp"])
    if building not in el.columns:
        raise SystemExit(f"ERROR: {building} has no electricity meter.")
    weather = pd.read_csv(source / "weather.csv", parse_dates=["timestamp"])
    weather = (weather[weather.site_id == site].drop_duplicates("timestamp")
               .set_index("timestamp")[["airTemperature", "dewTemperature"]])

    df = el.set_index("timestamp").rename(columns={building: "energy_kwh"})
    df["energy_kwh"] = df["energy_kwh"].round(3)
    df["demand_kw"] = df["energy_kwh"]                      # hourly kWh / 1 h = average kW
    df = df.join(weather, how="left")
    df["temperature"] = df["airTemperature"].round(1)
    df["humidity"] = relative_humidity(df["airTemperature"], df["dewTemperature"]).round(1)
    df = df[["energy_kwh", "demand_kw", "temperature", "humidity"]].reset_index()
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d %H:%M:%S")

    m = meta.loc[building]
    print(f"Wrote {out}: {len(df)} hourly rows, {df.timestamp.min()} -> {df.timestamp.max()}")
    print(f"Building {building}: {m.primaryspaceusage} / {m.sub_primaryspaceusage}, {m.sqm} m2, site {site} "
          f"({m.timezone})")
    print("Missing values:", df.drop(columns="timestamp").isna().sum().to_dict())
    print("Source: Building Data Genome Project 2 (CC BY-SA 4.0). Modified: one building selected, weather joined, "
          "humidity derived from dew point.")
    return df


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Convert a Building Data Genome Project 2 building into a PowerPulse CSV.")
    p.add_argument("--download", metavar="DIR", help="Download the BDG2 source files into DIR.")
    p.add_argument("--source", metavar="DIR", help="Folder holding metadata.csv, weather.csv, electricity_cleaned.csv.")
    p.add_argument("--list", action="store_true", help="List office buildings with complete data.")
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--building", help="building_id to convert, e.g. Hog_office_Lizzie.")
    p.add_argument("--out", help="Output CSV (default data/bdg2_<building>.csv).")
    args = p.parse_args(argv)
    if args.download:
        download(Path(args.download))
    if args.list or args.building:
        if not args.source and not args.download:
            p.error("--source is required")
        source = Path(args.source or args.download)
        if args.list:
            list_offices(source, args.limit)
        if args.building:
            out = Path(args.out or Path(__file__).resolve().parents[1] / "data" / f"bdg2_{args.building.lower()}.csv")
            convert(source, args.building, out)
    elif not args.download:
        p.error("nothing to do: use --download, --list or --building")
    return 0


if __name__ == "__main__":
    sys.exit(main())
