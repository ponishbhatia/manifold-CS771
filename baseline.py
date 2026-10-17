"""Shared Part 1 reference for Part 2 experiments (CS771 Mini-Project 1).

Put this file at the package root (next to data/ and starter/) and import it:

    import baseline as B

    W, stations = B.load_data()
    T_train, T_val = B.issue_times()                 # noon issue times, split by target date
    y_train, y_val = B.targets(W, T_train), B.targets(W, T_val)

    pred = ...                                       # your model's validation forecasts, µg/m³
    B.evaluate(pred, y_val, T_val)                   # RMSE/MAE overall + Sep/Oct/Nov
    B.compare({"My model": pred}, W, T_val)          # table next to both baselines

Conventions (do not change):
  * Every flat array is ordered issue time first, then station in bihar_stations.csv order.
  * A row is train or val by its TARGET date: train targets 3 Jan - 31 Aug, val 1 Sep - 30 Nov.
  * Features for issue time t use only readings at or before t.
  * Scores are in µg/m³ with forecasts clipped at 0.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

DATA_DIR = Path(__file__).resolve().parent / "data"
N_STATIONS = 468

# Validation reference numbers (reproduced by Part_1.ipynb; checked by self_check()).
REFERENCE = {
    "Persistence":    {"RMSE": 32.75, "MAE": 17.47, "RMSE Sep": 23.33, "MAE Sep": 8.91,
                       "RMSE Oct": 20.22, "MAE Oct": 11.97, "RMSE Nov": 47.81, "MAE Nov": 31.70},
    "Ridge (Part 1)": {"RMSE": 29.58, "MAE": 18.16, "RMSE Sep": 20.34, "MAE Sep": 10.35,
                       "RMSE Oct": 19.36, "MAE Oct": 13.46, "RMSE Nov": 43.06, "MAE Nov": 30.81},
}


# ------------------------------------------------------------------ data

def load_data(data_dir=DATA_DIR):
    """Return (W, stations).

    W: hourly PM2.5, DatetimeIndex 'timestamp' (IST), one string column per sensor_id.
    stations: indexed by sensor_id, with latitude, longitude, x_km, y_km (spec's conversion).
    """
    data_dir = Path(data_dir)
    W = pd.read_csv(data_dir / "bihar_pm25_hourly_jan_nov_2024.csv",
                    index_col="timestamp", parse_dates=True)
    W.columns = W.columns.astype(str)
    stations = pd.read_csv(data_dir / "bihar_stations.csv",
                           dtype={"sensor_id": str}).set_index("sensor_id")
    stations["x_km"] = (stations.longitude - 85.5) * 100.3
    stations["y_km"] = (stations.latitude - 25.7) * 110.6

    assert list(stations.index) == list(W.columns)
    assert W.shape == (8040, N_STATIONS) and not W.isna().any().any()
    return W, stations


# ------------------------------------------------------------------ rows

def issue_times(lead=24, train_targets=("2024-01-03 12:00", "2024-08-31 12:00"),
                val_targets=("2024-09-01 12:00", "2024-11-30 12:00")):
    """(T_train, T_val): issue times = noon target times minus `lead` hours.

    Defaults give the main task: 242 training and 91 validation issue days.
    For the Part 5 refit use train_targets=("2024-01-03 12:00", "2024-11-30 12:00").
    For Part 3 pass lead=1, 3, 6 or 12 (targets stay at 12:00).
    """
    def make(first, last):
        return pd.date_range(first, last, freq="D") - pd.Timedelta(hours=lead)
    return make(*train_targets), make(*val_targets)


def targets(W, T, lead=24):
    """Target readings (issue time + lead), flattened: time first, then station."""
    return W.loc[pd.DatetimeIndex(T) + pd.Timedelta(hours=lead)].to_numpy().ravel()


def target_months(T, lead=24):
    """Month of each flat row's target (for per-month scores)."""
    return np.repeat((pd.DatetimeIndex(T) + pd.Timedelta(hours=lead)).month, N_STATIONS)


def row_index(W, T):
    """MultiIndex (issue_time, sensor_id) matching the flat row order, if you want a DataFrame."""
    return pd.MultiIndex.from_product([pd.DatetimeIndex(T), W.columns],
                                      names=["issue_time", "sensor_id"])


# ------------------------------------------------------------------ scoring

def evaluate(pred, y, T, lead=24):
    """RMSE and MAE (µg/m³) overall and per month present among the targets."""
    pred = np.clip(np.asarray(pred, dtype=float), 0, None)
    assert pred.shape == y.shape, f"pred {pred.shape} vs y {y.shape}"
    assert np.isfinite(pred).all(), "pred has NaN or inf"
    err = pred - y
    out = {"RMSE": np.sqrt(np.mean(err ** 2)), "MAE": np.mean(np.abs(err))}
    months = target_months(T, lead)
    for m in np.unique(months):
        e = err[months == m]
        name = pd.Timestamp(2024, m, 1).strftime("%b")
        out[f"RMSE {name}"] = np.sqrt(np.mean(e ** 2))
        out[f"MAE {name}"] = np.mean(np.abs(e))
    return out


def compare(preds, W, T, lead=24, stations=None, include_baselines=True):
    """Table of evaluate() for {name: pred}, with persistence and Part 1 ridge on top."""
    y = targets(W, T, lead)
    rows = {}
    if include_baselines:
        rows["Persistence"] = evaluate(persistence(W, T), y, T, lead)
        rows["Ridge (Part 1)"] = evaluate(part1_ridge_predict(W, T, lead, stations), y, T, lead)
    for name, p in preds.items():
        rows[name] = evaluate(p, y, T, lead)
    return pd.DataFrame(rows).T.round(2)


def daily_rmse(pred, y, T):
    """Station RMSE for each issue day (Part 4: worst day)."""
    err2 = ((np.clip(pred, 0, None) - y) ** 2).reshape(len(T), N_STATIONS)
    return pd.Series(np.sqrt(err2.mean(axis=1)), index=pd.DatetimeIndex(T))


def to_wide(pred, W, T, lead=24):
    """Flat forecasts -> DataFrame (target times x sensor_id), µg/m³. Part 4 and blending."""
    return pd.DataFrame(np.clip(pred, 0, None).reshape(len(T), N_STATIONS),
                        index=pd.DatetimeIndex(T) + pd.Timedelta(hours=lead),
                        columns=W.columns)


# ------------------------------------------------------------------ baselines

def persistence(W, T):
    """Forecast = each station's reading at the issue time."""
    return W.loc[pd.DatetimeIndex(T)].to_numpy().ravel()


def neighbour_matrix(stations, k=10):
    """468 x 468 averaging matrix: 1/k on each station's k nearest other stations (km)."""
    xy = stations[["x_km", "y_km"]].to_numpy()
    D = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    idx = np.argsort(D, axis=1, kind="stable")[:, :k]
    A = np.zeros((len(xy), len(xy)))
    A[np.arange(len(xy))[:, None], idx] = 1.0 / k
    return A


def distance_matrix(stations):
    """468 x 468 distances in km (diagonal 0). Useful for radius-based neighbourhood features."""
    xy = stations[["x_km", "y_km"]].to_numpy()
    return np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(axis=2))


PART1_FEATURE_NAMES = ([f"lag{k}" for k in range(24)]
                       + ["nbr_now", "nbr_mean24", "net_now", "sin_doy", "cos_doy"])


def lag_stack(W, T, hours=24):
    """L[i, s, k] = reading of station s at T[i] - k hours, k = 0..hours-1."""
    T = pd.DatetimeIndex(T)
    return np.stack([W.reindex(T - pd.Timedelta(hours=k)).to_numpy() for k in range(hours)], axis=2)


def part1_features(W, T, A):
    """The 29 Part 1 features, one row per (issue time, station). Columns: PART1_FEATURE_NAMES.

    d = days since 1 Jan 2024 of the issue day; divisor 365; network mean includes the station.
    Works for any issue times (Part 3) and for a single issue time inside forecast.py.
    """
    T = pd.DatetimeIndex(T)
    L = lag_stack(W, T, 24)
    n = len(T)
    lags = L.reshape(n * N_STATIONS, 24)
    nbr_now = (L[:, :, 0] @ A.T).ravel()
    nbr_mean24 = np.einsum("tsk,js->tjk", L, A).mean(axis=2).ravel()
    net_now = np.repeat(L[:, :, 0].mean(axis=1), N_STATIONS)
    d = np.repeat((T.normalize() - pd.Timestamp("2024-01-01")).days.to_numpy(), N_STATIONS)
    X = np.column_stack([lags, nbr_now, nbr_mean24, net_now,
                         np.sin(2 * np.pi * d / 365), np.cos(2 * np.pi * d / 365)])
    assert not np.isnan(X).any(), "need 24 h of history before every issue time"
    return X


def part1_ridge_fit(W, T_train, lead=24, stations=None):
    """Fit the fixed Part 1 ridge (alpha=1.0, raw features, pooled). Returns (model, A)."""
    if stations is None:
        _, stations = load_data()
    A = neighbour_matrix(stations)
    model = Ridge(alpha=1.0).fit(part1_features(W, T_train, A), targets(W, T_train, lead))
    return model, A


def part1_ridge_predict(W, T, lead=24, stations=None):
    """Part 1 ridge forecasts for issue times T (refitted for this lead on the default train split)."""
    T_train, _ = issue_times(lead)
    model, A = part1_ridge_fit(W, T_train, lead, stations)
    return np.clip(model.predict(part1_features(W, T, A)), 0, None)


# ------------------------------------------------------------------ self-check

def self_check():
    """Reproduce the Part 1 reference numbers. Run once after copying this file."""
    W, stations = load_data()
    T_train, T_val = issue_times()
    assert len(targets(W, T_train)) == 113_256 and len(targets(W, T_val)) == 42_588
    table = compare({}, W, T_val, stations=stations)
    for name, ref in REFERENCE.items():
        for k, v in ref.items():
            assert abs(table.loc[name, k] - v) < 0.01, f"{name} {k}: {table.loc[name, k]} != {v}"
    A = neighbour_matrix(stations)
    t = T_val[10]
    assert np.allclose(part1_features(W, [t], A), part1_features(W.loc[:t], [t], A))
    print("baseline.py OK: row counts, reference numbers and leakage check all pass")
    return table


if __name__ == "__main__":
    print(self_check())
