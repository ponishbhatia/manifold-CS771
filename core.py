"""Shared core for CS771 Mini-Project 1: loading, rows, split, scoring, persistence.

Everyone imports from here. Do not copy and modify; ask for changes instead.
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

HOURLY_FILE = "bihar_pm25_hourly_jan_nov_2024.csv"
STATIONS_FILE = "bihar_stations.csv"

TRAIN_END = pd.Timestamp("2024-08-31 23:59")   # last training target date
VAL_END = pd.Timestamp("2024-11-30 23:59")     # last validation target date
VAL_MONTHS = {9: "Sep", 10: "Oct", 11: "Nov"}


# ---------------------------------------------------------------- loading

def load(data_dir=DATA_DIR):
    """Return (df, stations).

    df: hourly PM2.5, DatetimeIndex 'timestamp' (IST), one column per sensor_id
        in the same order as bihar_stations.csv.
    stations: sensor_id, latitude, longitude, x_km, y_km (spec's conversion).
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / HOURLY_FILE, index_col=0, parse_dates=True)
    df.index.name = "timestamp"
    df.columns = df.columns.astype(str)

    stations = pd.read_csv(data_dir / STATIONS_FILE, dtype={"sensor_id": str})
    stations["x_km"] = (stations["longitude"] - 85.5) * 100.3
    stations["y_km"] = (stations["latitude"] - 25.7) * 110.6

    df = df[stations["sensor_id"].tolist()]          # enforce station order

    full = pd.date_range(df.index.min(), df.index.max(), freq="h")
    assert df.index.equals(full), "hourly grid has gaps or duplicates"
    assert df.shape[1] == 468, f"expected 468 stations, got {df.shape[1]}"
    return df, stations


# ---------------------------------------------------------------- rows

def make_rows(df, lead=24, target_hour=12, min_history_hours=24,
              train_end=TRAIN_END, last_target=VAL_END):
    """One row per (issue_time, station) for targets at `target_hour` each day.

    issue_time = target_time - lead hours. A row is kept only if the issue time
    has at least `min_history_hours` of readings (t - (min_history_hours-1) h ... t).
    Split is by TARGET date: 'train' if target <= train_end, else 'val'.

    Columns: target_time, y (target reading), y_issue (reading at issue time,
    i.e. persistence), split.
    For the Part 5 refit, call with train_end=VAL_END.
    """
    first_issue = df.index.min() + pd.Timedelta(hours=min_history_hours - 1)
    targets = df.index[(df.index.hour == target_hour) & (df.index <= last_target)]
    issues = targets - pd.Timedelta(hours=lead)
    keep = issues >= first_issue
    targets, issues = targets[keep], issues[keep]

    n_s = df.shape[1]
    idx = pd.MultiIndex.from_product([issues, df.columns],
                                     names=["issue_time", "station"])
    rows = pd.DataFrame(index=idx)
    rows["target_time"] = np.repeat(targets.values, n_s)
    rows["y"] = df.loc[targets].to_numpy().ravel()
    rows["y_issue"] = df.loc[issues].to_numpy().ravel()
    rows["split"] = np.where(rows["target_time"] <= train_end, "train", "val")
    return rows


def issue_times(rows, split):
    """Unique issue times for one split, in order."""
    return rows.index.get_level_values("issue_time")[rows["split"] == split].unique()


# ---------------------------------------------------------------- scoring

def _rmse(e):
    return float(np.sqrt(np.mean(e ** 2)))


def evaluate(pred, rows, split="val"):
    """RMSE and MAE (µg/m³) overall and per validation month.

    pred: array or Series aligned with rows[rows.split == split], already
    converted back to µg/m³. Negative values are clipped to 0 here.
    """
    sub = rows[rows["split"] == split]
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    assert pred.shape[0] == len(sub), f"pred has {pred.shape[0]} rows, expected {len(sub)}"
    assert not np.isnan(pred).any(), "pred contains NaN"

    err = pred - sub["y"].to_numpy()
    out = {"rmse": _rmse(err), "mae": float(np.mean(np.abs(err)))}
    months = sub["target_time"].dt.month.to_numpy()
    for m, name in VAL_MONTHS.items():
        mask = months == m
        if mask.any():
            out[f"rmse_{name}"] = _rmse(err[mask])
            out[f"mae_{name}"] = float(np.mean(np.abs(err[mask])))
    return out


def daily_rmse(pred, rows, split="val"):
    """Station RMSE for each target day (Part 4: picking the worst day)."""
    sub = rows[rows["split"] == split]
    err = np.clip(np.asarray(pred, dtype=float), 0, None) - sub["y"].to_numpy()
    s = pd.Series(err ** 2, index=sub["target_time"].dt.normalize().to_numpy())
    return np.sqrt(s.groupby(level=0).mean())


def results_table(results):
    """dict of {name: evaluate(...) output} -> DataFrame for the report."""
    return pd.DataFrame(results).T.round(2)


# ---------------------------------------------------------------- baselines

def persistence(rows, split="val"):
    """Forecast = reading at the issue time."""
    return rows.loc[rows["split"] == split, "y_issue"].to_numpy()


# ---------------------------------------------------------------- predictions

def to_wide(pred, rows, split="val"):
    """Flat predictions -> (target dates x 468 stations) DataFrame, µg/m³."""
    sub = rows[rows["split"] == split]
    s = pd.Series(np.clip(np.asarray(pred, dtype=float), 0, None),
                  index=pd.MultiIndex.from_arrays(
                      [sub["target_time"].to_numpy(),
                       sub.index.get_level_values("station")],
                      names=["target_time", "sensor_id"]))
    return s.unstack("sensor_id")[sub.index.get_level_values("station").unique()]


def save_preds(pred, rows, name, out_dir="preds", split="val"):
    """Save as preds/<name>.parquet (91 x 468 for validation)."""
    Path(out_dir).mkdir(exist_ok=True)
    path = Path(out_dir) / f"{name}.parquet"
    to_wide(pred, rows, split).to_parquet(path)
    return path
