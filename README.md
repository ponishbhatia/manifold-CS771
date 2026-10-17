# Next-Day PM2.5 Forecasting and Mapping — student package

Read `mini-project-1.pdf` first. This file describes the data and the starter kit.

## Contents

| Path | What it is |
|---|---|
| `pm25_forecast_assignment.pdf` | The assignment |
| `data/bihar_pm25_hourly_jan_nov_2024.csv` | Hourly PM2.5 (µg/m³) for 468 Bihar stations, 1 Jan 00:00 – 30 Nov 23:00 2024 |
| `data/bihar_stations.csv` | Location of each Bihar station |
| `data/bihar_grid_005deg.csv` | 3,396 grid points inside Bihar at 0.05° (≈ 5 km) spacing, for Part 4 maps |
| `data/bihar_boundary.csv` | Bihar state outline (longitude, latitude), for plotting |
| `data/kanpur_pm25_raw_jul_aug_2026.csv` | Part 0 only: raw Kanpur sensor data |
| `starter/forecast_template.py` | Template for the `forecast.py` you submit in Part 5 |
| `starter/check_submission.py` | Checks that your `forecast.py` works the way the grader will call it |
| `requirements.txt` | Python packages used by the starter kit |

## Bihar data format

`bihar_pm25_hourly_jan_nov_2024.csv` is a **wide** table:

- one row per hour (8,040 rows), first column `timestamp` (format `YYYY-MM-DD HH:MM`);
- one column per station (468 columns), named by `sensor_id`;
- values are PM2.5 in µg/m³, rounded to 2 decimals. There are no missing values: gaps were already filled upstream from neighbouring stations (see the assignment).

**Time zone: Bihar timestamps are local time (IST, UTC + 5:30).** The Kanpur file is in UTC.

`bihar_stations.csv` has columns `sensor_id`, `latitude`, `longitude`, in the same order as the columns of the hourly table.

```python
import pandas as pd
W = pd.read_csv("data/bihar_pm25_hourly_jan_nov_2024.csv", index_col="timestamp", parse_dates=True)
stations = pd.read_csv("data/bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")
assert list(stations.index) == list(W.columns)
noon = W[W.index.hour == 12]          # e.g. all 12:00 readings
```

Station ids are read as strings (column names). Keep them as strings throughout.

## Kanpur data format (Part 0)

`kanpur_pm25_raw_jul_aug_2026.csv` is also wide, on a 15-minute UTC grid (`timestamp_utc`), with three columns per sensor: `<id>_latitude`, `<id>_longitude`, `<id>_pm25`. It is raw: discovering its problems is the point of Part 0.

## Part 5: the `forecast` function

The grader imports your `forecast.py` and calls

```python
forecast(history, issue_time)
```

- `history` is a wide DataFrame **in exactly the layout of `bihar_pm25_hourly_jan_nov_2024.csv`** (DatetimeIndex `timestamp`, one string column per `sensor_id`), containing every hour from 1 Jan 2024 00:00 up to and including `issue_time`. During grading it also contains December hours up to `issue_time`.
- `issue_time` is a `pd.Timestamp` at 12:00 (IST).
- Return a `pd.Series` of 468 non-missing PM2.5 forecasts for `issue_time + 24 h`, indexed by `sensor_id` (strings).

Station coordinates are not passed in. If your model needs them, include `bihar_stations.csv` (or anything you derive from it) in your submission folder and load it from the folder of `forecast.py`. Start from `starter/forecast_template.py`.

Before submitting, run

```bash
python starter/check_submission.py path/to/your_submission_folder
```

It calls your `forecast` exactly as the grader will (30 issue times in November instead of December), checks the output format and the 10-second limit, and prints your November RMSE as a sanity check. November is part of your own data, so that number is not a test score.
