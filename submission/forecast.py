"""Part 5: next-day (24 h) noon PM2.5 forecast for the 468 Bihar stations.

Linear model on log1p(PM2.5), fitted by least absolute deviations (Laplace likelihood), forecast = smeared mean
exp(mu) * smearing - 1. Fitted on all targets 3 Jan - 30 Nov 2024; nothing is refitted inside forecast().
"""
from pathlib import Path
import json

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N = 468

# Load fitted objects once, at import time.
MODEL = json.load(open(HERE / "model.json"))
MEAN, SCALE, COEF = np.array(MODEL["x_mean"]), np.array(MODEL["x_scale"]), np.array(MODEL["coef"])
SMEARING, MAX_PM25 = MODEL["smearing"], MODEL["max_pm25"]
STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")

# Neighbour-averaging matrices: row s has 1/k on the k nearest OTHER stations of s (km conversion from the spec).
xy = np.column_stack([(STATIONS.longitude - 85.5) * 100.3, (STATIONS.latitude - 25.7) * 110.6])
dist = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(axis=2))
np.fill_diagonal(dist, np.inf)
nearest = np.argsort(dist, axis=1, kind="stable")
A = {k: np.zeros((N, N)) for k in (10, 75)}
for k in A:
    A[k][np.arange(N)[:, None], nearest[:, :k]] = 1.0 / k


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    window = history.loc[issue_time - pd.Timedelta(hours=199): issue_time]   # last 200 hours (longest window: 168 h)
    V = window.to_numpy(dtype=float)                                          # (200, 468); last row = issue time
    if len(V) < 168:                                                          # too little history for the 7-day mean: persistence
        return pd.Series(np.clip(V[-1], 0, None), index=history.columns, name="pm25_forecast")
    i = len(V) - 1
    now, mean24, mean6 = V[i], V[i - 23:].mean(axis=0), V[i - 5:].mean(axis=0)

    # own: log readings at lags 0,1,2,3,6,12,18,23 h and mean, min, max, std of the last 24 h
    lags = np.log1p(V[i - 23:][::-1])                                         # row k = reading at t - k h
    own = np.column_stack([lags[[0, 1, 2, 3, 6, 12, 18, 23]].T, lags.mean(0), lags.min(0), lags.max(0), lags.std(0)])
    prevday = np.log1p(V[i - 24])[:, None]                                    # same hour yesterday

    # nbr: 10 and 75 nearest stations (now and 24 h mean), own reading minus its 10-neighbour mean
    n10, n10_24, n75, n75_24 = A[10] @ now, A[10] @ mean24, A[75] @ now, A[75] @ mean24
    nbr = np.column_stack([np.log1p(n10), np.log1p(n10_24), np.log1p(n75), np.log1p(n75_24),
                           np.log1p(now) - np.log1p(n10)])

    # network: statewide level now, its 24 h change, spread across stations
    log_net = np.log1p(now.mean())
    net = np.tile([log_net, log_net - np.log1p(V[i - 24].mean()), np.log1p(now).std()], (N, 1))

    # trend: 18 h change, last 6 h vs previous 6 h, last 24 h vs previous 24 h
    trend = np.column_stack([np.log1p(now) - np.log1p(V[i - 17]),
                             np.log1p(mean6) - np.log1p(V[i - 11:i - 5].mean(axis=0)),
                             np.log1p(mean24) - np.log1p(V[i - 47:i - 23].mean(axis=0))])

    long = np.log1p(V[i - 167:].mean(axis=0))[:, None]                        # 7-day mean

    X = np.hstack([own, prevday, nbr, net, trend, long])                      # (468, 25)
    mu = COEF[0] + ((X - MEAN) / SCALE) @ COEF[1:]                            # log1p scale
    prediction = np.exp(mu) * SMEARING - 1.0                                  # smeared-mean back-transform

    return pd.Series(np.clip(prediction, 0, MAX_PM25), index=history.columns, name="pm25_forecast")
