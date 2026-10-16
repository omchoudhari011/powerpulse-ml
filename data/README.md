# Dataset instructions

This folder ships with **`dummy_building.csv`**, a synthetic demo dataset (2 years, hourly, 17,520 rows) made by
`tools/generate_dummy_dataset.py`. It is for trying the pipeline only - it is NOT real data. For real results put
**your own building-energy CSV** here (e.g. `data/my_building.csv`) and pass it with `--csv`. The ML code in `src/`
never generates or substitutes data.

## Columns

| column | required | meaning |
|---|---|---|
| `timestamp` | yes | date-time of the reading (any format pandas can parse; set `timestamp_format` in a config JSON otherwise) |
| `demand_kw` | one of these two | electrical **demand** (power) in kW. If present it is the default forecasting target |
| `energy_kwh` | one of these two | **energy consumed** during the interval in kWh. Used as target only when `demand_kw` is absent |
| `temperature` | optional | outdoor/indoor temperature (C or F) |
| `humidity` | optional | relative humidity, 0-100 % |
| `occupancy` | optional | people count or occupancy level, >= 0 |
| `equipment_status` | optional | on/off (`1/0`, `on/off`, `true/false`, `running/stopped`, ...) |

Header example (first line of the file):

```
timestamp,energy_kwh,demand_kw,temperature,humidity,occupancy,equipment_status
```

Column names are case-insensitive. Unknown columns are ignored and reported. Missing optional columns are
simply not used. Hourly data is the reference case; other regular intervals (e.g. 15 min) are detected
and the 24 h input window / 24 h horizon / lags are converted to the right number of steps.

## How much data?

The validator reports the exact minimum for your settings. With the defaults (24 h in / 24 h out, 168 h lag,
70/15/15 split) a few weeks is the bare minimum; several months to a year or more is needed for results worth
trusting (a full year is needed for the model to see every season).

CSV files in this folder are git-ignored by default, except `dummy_building.csv`.

## Real dataset: Building Data Genome Project 2

`tools/convert_bdg2.py` turns one building from the [Building Data Genome Project 2](https://github.com/buds-lab/building-data-genome-project-2)
(hourly meter data from 1,636 real non-residential buildings, 2016-2017; Miller et al. 2020, *Scientific Data* 7:368)
into this format. Example output: `bdg2_hog_office_lizzie.csv`, a 10,352 m2 office (site "Hog", US/Central),
17,544 hourly rows, passes `src.data_validation`.

```powershell
python tools\convert_bdg2.py --download downloads\bdg2                  # ~195 MB, SHA-256 checked
python tools\convert_bdg2.py --source downloads\bdg2 --list            # office buildings with complete data
python tools\convert_bdg2.py --source downloads\bdg2 --building Hog_office_Lizzie
```

Columns: `energy_kwh` (BDG2 cleaned electricity reading), `demand_kw` (the same hourly kWh read as average kW),
`temperature` (site weather station), `humidity` (derived from air and dew-point temperature). BDG2 has no occupancy
or equipment-status columns. Meter and weather timestamps are both local site time.

License: CC BY-SA 4.0. Credit BDG2 when you use it, and share files derived from it under the same license. The
converted files differ from the original: one building selected, weather joined, humidity derived.
