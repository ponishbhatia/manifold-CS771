"""Shared Part 1 helpers for Part 2 experiments (CS771 Mini-Project 1).

Put this file next to data/ and run `python baseline.py` once (prints "baseline.py OK").

    import baseline as B

    W, stations = B.load_data()
    T_train, T_val = B.issue_times()
    y_train, y_val = B.targets(W, T_train), B.targets(W, T_val)

    A = B.neighbour_matrix(stations)
    X_train, X_val = B.part1_features(W, T_train, A), B.part1_features(W, T_val, A)
    pred_val = ...                                     # fit on train only
    B.compare({"My model": pred_val}, W, stations, T_val)

Conventions (do not change):
  * Flat arrays are ordered issue time first, then station (bihar_stations.csv order).
  * A row is train/val by its TARGET date: train targets 3 Jan - 31 Aug, val 1 Sep - 30 Nov.
  * Features for issue time t use only readings at or before t.
  * Scores are in ug/m3, forecasts clipped at 0.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA_DIR = Path(__file__).resolve().parent / "data"
N = 468  # stations

# Validation reference numbers, checked by self_check().
REFERENCE = {
    "Persistence":    {"RMSE": 32.75, "MAE": 17.47, "RMSE Sep": 23.33, "MAE Sep": 8.91,
                       "RMSE Oct": 20.22, "MAE Oct": 11.97, "RMSE Nov": 47.81, "MAE Nov": 31.70},
    "Ridge (Part 1)": {"RMSE": 29.58, "MAE": 18.16, "RMSE Sep": 20.34, "MAE Sep": 10.35,
                       "RMSE Oct": 19.36, "MAE Oct": 13.46, "RMSE Nov": 43.06, "MAE Nov": 30.81},
}


# ---------------------------------------------------------------- data and rows

def load_data(data_dir=DATA_DIR):
    """Return (W, stations). W: hourly PM2.5 (hours x 468, columns = sensor_id).
    stations: indexed by sensor_id, with latitude, longitude, x_km, y_km."""
    data_dir = Path(data_dir)
    W = pd.read_csv(data_dir / "bihar_pm25_hourly_jan_nov_2024.csv", index_col="timestamp", parse_dates=True)
    W.columns = W.columns.astype(str)
    stations = pd.read_csv(data_dir / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")
    stations["x_km"] = (stations.longitude - 85.5) * 100.3
    stations["y_km"] = (stations.latitude - 25.7) * 110.6
    assert list(stations.index) == list(W.columns) and W.shape == (8040, N) and not W.isna().any().any()
    return W, stations


def issue_times(lead=24, train_end="2024-08-31 12:00", val_end="2024-11-30 12:00"):
    """(T_train, T_val): issue times = noon target times minus `lead` hours.
    Defaults: 242 train / 91 val issue days. Part 3: lead=1,3,6,12. Part 5 refit: train_end="2024-11-30 12:00"."""
    make = lambda a, b: pd.date_range(a, b, freq="D") - pd.Timedelta(hours=lead)
    return make("2024-01-03 12:00", train_end), make("2024-09-01 12:00", val_end)


def targets(W, T, lead=24):
    """Target readings (issue time + lead), flattened: time first, then station."""
    return W.loc[pd.DatetimeIndex(T) + pd.Timedelta(hours=lead)].to_numpy().ravel()


# ---------------------------------------------------------------- scoring

def evaluate(pred, y, T, lead=24):
    """RMSE and MAE overall and per target month (dict)."""
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    assert pred.shape == y.shape and np.isfinite(pred).all(), "bad prediction array"
    err = pred - y
    out = {"RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err))}
    months = np.repeat((pd.DatetimeIndex(T) + pd.Timedelta(hours=lead)).month, N)
    for m in np.unique(months):
        e, name = err[months == m], pd.Timestamp(2024, m, 1).strftime("%b")
        out[f"RMSE {name}"], out[f"MAE {name}"] = np.sqrt(np.mean(e ** 2)), np.mean(np.abs(e))
    return out


def compare(preds, W, stations, T, lead=24):
    """Table of evaluate() for {name: pred}, with persistence and Part 1 ridge on top."""
    y = targets(W, T, lead)
    rows = {"Persistence": evaluate(persistence(W, T), y, T, lead),
            "Ridge (Part 1)": evaluate(part1_ridge_predict(W, stations, T, lead), y, T, lead)}
    rows.update({name: evaluate(p, y, T, lead) for name, p in preds.items()})
    return pd.DataFrame(rows).T.round(2)


# ---------------------------------------------------------------- features

def lag_stack(W, T, hours=24):
    """L[i, s, k] = reading of station s at T[i] - k hours (k = 0..hours-1)."""
    T = pd.DatetimeIndex(T)
    return np.stack([W.reindex(T - pd.Timedelta(hours=k)).to_numpy() for k in range(hours)], axis=2)


def neighbour_matrix(stations, k=10):
    """468 x 468 averaging matrix: 1/k on each station's k nearest OTHER stations (km).
    `readings @ A.T` gives each station's neighbour mean."""
    xy = stations[["x_km", "y_km"]].to_numpy()
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    A = np.zeros((N, N))
    A[np.arange(N)[:, None], np.argsort(D, axis=1, kind="stable")[:, :k]] = 1.0 / k
    return A


FEATURE_NAMES = [f"lag{k}" for k in range(24)] + ["nbr_now", "nbr_mean24", "net_now", "sin_doy", "cos_doy"]


def part1_features(W, T, A):
    """The 29 Part 1 features, one row per (issue time, station); columns = FEATURE_NAMES.
    Everything is relative to the issue time, so it works for any lead and for a single
    issue time inside forecast.py. d = days since 1 Jan 2024 of the issue day."""
    T = pd.DatetimeIndex(T)
    L = lag_stack(W, T, 24)
    d = np.repeat((T.normalize() - pd.Timestamp("2024-01-01")).days.to_numpy(), N)
    X = np.column_stack([L.reshape(len(T) * N, 24),
                         (L[:, :, 0] @ A.T).ravel(),                           # 10 nbrs at t
                         np.einsum("tsk,js->tjk", L, A).mean(axis=2).ravel(),  # 10 nbrs over 24 h
                         np.repeat(L[:, :, 0].mean(axis=1), N),                # network at t
                         np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365)])
    assert not np.isnan(X).any(), "need 24 h of history before every issue time"
    return X


# ---------------------------------------------------------------- baselines

def persistence(W, T):
    """Forecast = each station's reading at the issue time."""
    return W.loc[pd.DatetimeIndex(T)].to_numpy().ravel()


def part1_ridge_predict(W, stations, T, lead=24):
    """Part 1 ridge (alpha=1, raw features, pooled), refitted for this lead on the default
    train split, then predicting issue times T."""
    A = neighbour_matrix(stations)
    T_train, _ = issue_times(lead)
    model = Ridge(alpha=1.0).fit(part1_features(W, T_train, A), targets(W, T_train, lead))
    return np.clip(model.predict(part1_features(W, T, A)), 0, None)


# ---------------------------------------------------------------- self-check

def self_check():
    """Reproduce the reference numbers and run a leakage check."""
    W, stations = load_data()
    T_train, T_val = issue_times()
    assert len(targets(W, T_train)) == 113_256 and len(targets(W, T_val)) == 42_588
    table = compare({}, W, stations, T_val)
    for name, ref in REFERENCE.items():
        for k, v in ref.items():
            assert abs(table.loc[name, k] - v) < 0.01, f"{name} {k}: {table.loc[name, k]} != {v}"
    A, t = neighbour_matrix(stations), T_val[10]
    assert np.allclose(part1_features(W, [t], A), part1_features(W.loc[:t], [t], A)), "leakage"
    print("baseline.py OK")
    return table


if __name__ == "__main__":
    print(self_check())
