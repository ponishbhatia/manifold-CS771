"""Feature extraction for Part 2, model C: Laplace / heteroscedastic-Gaussian linear models.

Every feature is a function of (history W, issue times T, ctx) and uses only readings at or
before each issue time t. Inputs are log-transformed (log1p) because PM2.5 is strongly right
skewed (skew 2.4 raw, 0.33 after log1p) and because pollution changes multiplicatively.

Feature groups (name -> function), so whole groups can be switched on/off for the ablation:

    own        log1p readings at selected lags + 24 h summary of the log series   (12 cols)
    prevday    log1p reading at t-24 h (yesterday's noon)                          (1 col)
    clim       station climatology (training mean) and anomalies versus it         (4 cols)
    nbr        log1p neighbourhood means (10 and 75 nearest stations), anomaly     (5 cols)
    network    log1p network mean now, its 24 h log change, spread across stations (3 cols)
    trend      log-ratio momentum features                                         (3 cols)
    long       log1p 7-day station mean                                            (1 col)
    long30     log1p 30-day station mean (experiment, off by default)              (1 col)
    calendar   sin/cos of day of year (experiment, off by default)                 (2 cols)

FINAL_GROUPS (own, prevday, nbr, network, trend, long) is the set chosen after the ablation;
clim, long30 and calendar are kept as documented experiments that did not help.

Rows are ordered issue time first, then station, exactly as in baseline.py.
"""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

import baseline as B

N = 468
SEL_LAGS = (0, 1, 2, 3, 6, 12, 18, 23)


def _row_index(W, T):
    index = W.index.get_indexer(pd.DatetimeIndex(T))
    assert (index >= 0).all(), "issue time not in history"
    return index


def _cumulative_sum(W):
    return np.vstack([np.zeros((1, W.shape[1])), np.cumsum(W.to_numpy(), axis=0)])


def _window_mean(cumulative, index, hours):
    """Mean over hours index-hours+1 .. index, window clipped at the start of the data."""
    low = np.maximum(0, index - hours + 1)
    return (cumulative[index + 1] - cumulative[low]) / (index + 1 - low)[:, None]


def _repeat_for_stations(per_time_values):
    return np.repeat(per_time_values, N, axis=0)


def build_ctx(W, stations, train_end="2024-08-31 12:00"):
    """Quantities fitted on TRAINING hours only (never on validation)."""
    train_hours = W.loc[:train_end]
    return dict(
        A10=B.neighbour_matrix(stations, k=10),
        A75=B.neighbour_matrix(stations, k=75),
        clim_mean=train_hours.mean().to_numpy(),
        clim_log_mean=np.log1p(train_hours).mean().to_numpy(),
    )


def own(W, T, ctx):
    log_lags = np.log1p(B.lag_stack(W, T, 24))
    picked = log_lags[:, :, list(SEL_LAGS)].reshape(-1, len(SEL_LAGS))
    summary = np.column_stack([
        log_lags.mean(axis=2).ravel(),
        log_lags.min(axis=2).ravel(),
        log_lags.max(axis=2).ravel(),
        log_lags.std(axis=2).ravel(),
    ])
    names = [f"log_lag{k}" for k in SEL_LAGS] + ["log_mean24", "log_min24", "log_max24", "log_std24"]
    return np.column_stack([picked, summary]), names


def prevday(W, T, ctx):
    values, index = W.to_numpy(), _row_index(W, T)
    assert (index - 24 >= 0).all()
    return np.log1p(values[index - 24]).reshape(-1, 1), ["log_d1"]


def clim(W, T, ctx):
    values, index = W.to_numpy(), _row_index(W, T)
    cumulative = _cumulative_sum(W)
    log_clim = np.log1p(ctx["clim_mean"])[None, :]
    now_anomaly = np.log1p(values[index]) - log_clim
    mean24_anomaly = np.log1p(_window_mean(cumulative, index, 24)) - log_clim
    columns = [np.tile(log_clim, (len(index), 1)),
               np.tile(ctx["clim_log_mean"][None, :], (len(index), 1)),
               now_anomaly, mean24_anomaly]
    return np.stack(columns, axis=2).reshape(-1, 4), ["log_clim", "clim_log_mean", "anom_now", "anom_mean24"]


def nbr(W, T, ctx):
    values, index = W.to_numpy(), _row_index(W, T)
    cumulative = _cumulative_sum(W)
    now_nbr10 = values[index] @ ctx["A10"].T
    mean24_nbr10 = _window_mean(cumulative, index, 24) @ ctx["A10"].T
    now_nbr75 = values[index] @ ctx["A75"].T
    columns = [np.log1p(now_nbr10), np.log1p(mean24_nbr10), np.log1p(now_nbr75),
               np.log1p(_window_mean(cumulative, index, 24) @ ctx["A75"].T),
               np.log1p(values[index]) - np.log1p(now_nbr10)]
    return (np.stack(columns, axis=2).reshape(-1, 5),
            ["log_nbr10_now", "log_nbr10_mean24", "log_nbr75_now", "log_nbr75_mean24", "local_anom"])


def network(W, T, ctx):
    values, index = W.to_numpy(), _row_index(W, T)
    log_now = np.log1p(values[index].mean(axis=1))
    log_change24 = log_now - np.log1p(values[index - 24].mean(axis=1))
    spread = np.log1p(values[index]).std(axis=1)
    return (_repeat_for_stations(np.column_stack([log_now, log_change24, spread])),
            ["log_net_now", "log_net_chg24", "net_log_std"])


def trend(W, T, ctx):
    values, index = W.to_numpy(), _row_index(W, T)
    cumulative = _cumulative_sum(W)
    change18 = np.log1p(values[index]) - np.log1p(values[index - 17])
    last6 = _window_mean(cumulative, index, 6)
    previous6 = (cumulative[index - 5] - cumulative[np.maximum(0, index - 11)]) / (index - 5 - np.maximum(0, index - 11))[:, None]
    momentum6 = np.log1p(last6) - np.log1p(previous6)
    last24 = _window_mean(cumulative, index, 24)
    low, high = np.maximum(0, index - 47), index - 24
    previous24 = (cumulative[high + 1] - cumulative[low]) / (high + 1 - low)[:, None]
    change_day = np.log1p(last24) - np.log1p(previous24)
    columns = [change18, momentum6, change_day]
    return np.stack(columns, axis=2).reshape(-1, 3), ["log_chg18", "log_mom6", "log_chg_day"]


def long(W, T, ctx):
    index = _row_index(W, T)
    cumulative = _cumulative_sum(W)
    return np.log1p(_window_mean(cumulative, index, 168)).reshape(-1, 1), ["log_mean7d"]


def long30(W, T, ctx):
    index = _row_index(W, T)
    cumulative = _cumulative_sum(W)
    return np.log1p(_window_mean(cumulative, index, 720)).reshape(-1, 1), ["log_mean30d"]


def calendar(W, T, ctx):
    days = (pd.DatetimeIndex(T).normalize() - pd.Timestamp("2024-01-01")).days.to_numpy()
    angle = 2 * np.pi * days / 365
    return _repeat_for_stations(np.column_stack([np.sin(angle), np.cos(angle)])), ["sin_doy", "cos_doy"]


GROUPS = dict(own=own, prevday=prevday, clim=clim, nbr=nbr, network=network, trend=trend,
              long=long, long30=long30, calendar=calendar)
FINAL_GROUPS = ["own", "prevday", "nbr", "network", "trend", "long"]
DEFAULT_GROUPS = FINAL_GROUPS


def extract_features(W, T, ctx, groups=None):
    """Return (X, names, group_columns) for the requested groups."""
    groups = list(groups or DEFAULT_GROUPS)
    blocks, names, group_columns, start = [], [], {}, 0
    for group in groups:
        block, block_names = GROUPS[group](W, T, ctx)
        blocks.append(block)
        names += block_names
        group_columns[group] = list(range(start, start + block.shape[1]))
        start += block.shape[1]
    X = np.column_stack(blocks)
    assert X.shape[0] == len(T) * N and np.isfinite(X).all(), "bad feature matrix"
    return X, names, group_columns


def verify_features():
    """Shape, NaN, leakage and single-issue-time parity checks (suggestions.md, tests 1-4)."""
    W, stations = B.load_data()
    T_train, T_val = B.issue_times()
    ctx = build_ctx(W, stations)
    every_group = list(GROUPS)
    X_train, names, _ = extract_features(W, T_train, ctx, every_group)
    X_val, _, _ = extract_features(W, T_val, ctx, every_group)
    assert X_train.shape == (113_256, len(names)) and X_val.shape == (42_588, len(names))
    for time_list, X_batch, position in ((T_val, X_val, 10), (T_train, X_train, 0), (T_train, X_train, 100)):
        t = time_list[position]
        full_history, _, _ = extract_features(W, [t], ctx, every_group)
        cut_history, _, _ = extract_features(W.loc[:t], [t], ctx, every_group)
        assert np.allclose(full_history, cut_history), "LEAKAGE"
        assert np.allclose(full_history, X_batch[position * N:(position + 1) * N]), "PARITY"
    print(f"features_C OK: {len(names)} columns, no NaN, no leakage, parity holds")


if __name__ == "__main__":
    verify_features()
