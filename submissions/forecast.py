"""Part 5 submission: next-day (24 h) noon PM2.5 forecast for the 468 Bihar stations.

Model: linear model on log1p(PM2.5) fitted by least absolute deviations (Laplace likelihood, IRLS), with the
smeared-mean back-transform exp(mu) * S - 1. Fitted on all targets 3 Jan - 30 Nov 2024 (part5/fit_final_model.py).
Self-contained: needs only model.json and bihar_stations.csv from this folder, no other repo code. The 25 features
reproduce cache/features_C.py (groups own, prevday, nbr, network, trend, long) for a SINGLE issue time and use
only rows of `history` at or before `issue_time`. Nothing is fitted or updated inside forecast().
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N = 468
SEL_LAGS = (0, 1, 2, 3, 6, 12, 18, 23)
TAIL = 200            # hours of history used (longest window is 168 h; the rest are shorter)

# ---- loaded once, at import time ---------------------------------------------------------------------------------
_M = json.load(open(HERE / "model.json"))
_MEAN, _SCALE = np.asarray(_M["x_mean"]), np.asarray(_M["x_scale"])
_COEF, _SMEAR, _MAX = np.asarray(_M["coef"]), float(_M["smearing"]), float(_M["max_pm25"])
STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")
assert len(STATIONS) == N and _COEF.shape == (26,)


def _neighbour_matrix(k):
    """Same as baseline.neighbour_matrix: 1/k on each station's k nearest OTHER stations (km, spec's conversion)."""
    xy = np.column_stack([(STATIONS.longitude - 85.5) * 100.3, (STATIONS.latitude - 25.7) * 110.6])
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    A = np.zeros((N, N))
    A[np.arange(N)[:, None], np.argsort(D, axis=1, kind="stable")[:, :k]] = 1.0 / k
    return A


_A10, _A75 = _neighbour_matrix(10), _neighbour_matrix(75)


def _features(V):
    """V: (hours, 468) readings, last row = issue time. Returns the (468, 25) feature matrix."""
    i = len(V) - 1
    wmean = lambda h: V[max(0, i - h + 1): i + 1].mean(axis=0)            # mean of the last h hours, clipped at data start
    now, m24, m6 = V[i], wmean(24), wmean(6)

    log_lags = np.log1p(V[i - 23: i + 1][::-1])                            # row k = reading at t - k h, k = 0..23
    own = np.column_stack([log_lags[list(SEL_LAGS)].T, log_lags.mean(0), log_lags.min(0), log_lags.max(0), log_lags.std(0)])
    prevday = np.log1p(V[i - 24])[:, None]

    nbr10_now, nbr10_24, nbr75_now, nbr75_24 = _A10 @ now, _A10 @ m24, _A75 @ now, _A75 @ m24
    nbr = np.column_stack([np.log1p(nbr10_now), np.log1p(nbr10_24), np.log1p(nbr75_now), np.log1p(nbr75_24),
                           np.log1p(now) - np.log1p(nbr10_now)])

    log_net = np.log1p(now.mean())
    net = np.tile([log_net, log_net - np.log1p(V[i - 24].mean()), np.log1p(now).std()], (N, 1))

    prev6 = V[max(0, i - 11): i - 5].mean(axis=0)                          # hours t-11 .. t-6
    prev24 = V[max(0, i - 47): i - 23].mean(axis=0)                        # hours t-47 .. t-24
    trend = np.column_stack([np.log1p(now) - np.log1p(V[i - 17]), np.log1p(m6) - np.log1p(prev6),
                             np.log1p(m24) - np.log1p(prev24)])

    long = np.log1p(wmean(168))[:, None]
    X = np.hstack([own, prevday, nbr, net, trend, long])
    assert X.shape == (N, 25) and np.isfinite(X).all()
    return X


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex, one string column per sensor_id), every hour up to issue_time.
    Returns the forecast for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    issue_time = pd.Timestamp(issue_time)
    h = history.loc[:issue_time]                                           # information cutoff: nothing after issue_time
    h = h.rename(columns=str).reindex(columns=STATIONS.index)
    assert h.index[-1] == issue_time, "history must contain the issue time"
    hours = issue_time - pd.to_timedelta(np.arange(min(TAIL, len(h)))[::-1], unit="h")   # the last TAIL hourly stamps
    h = h.reindex(hours).ffill().bfill()                                                  # guard against gaps
    V = h.to_numpy(dtype=float)
    last = V[-1]
    if len(V) < 25 or not np.isfinite(V).all():                            # < 25 h of history (features need t-24): persistence
        return pd.Series(np.clip(np.nan_to_num(last), 0, None), index=STATIONS.index, name="pm25_forecast")

    Z = (_features(V) - _MEAN) / _SCALE
    mu = _COEF[0] + Z @ _COEF[1:]                                          # log1p scale
    pred = np.clip(np.exp(mu) * _SMEAR - 1.0, 0.0, _MAX)
    pred = np.where(np.isfinite(pred), pred, last)
    return pd.Series(pred, index=STATIONS.index, name="pm25_forecast")