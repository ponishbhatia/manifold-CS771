import numpy as np, pandas as pd
import baseline as B
N = 468

def _i(W, T):
    i = W.index.get_indexer(pd.DatetimeIndex(T)); assert (i >= 0).all(); return i
def _cs(W):
    return np.vstack([np.zeros((1, W.shape[1])), np.cumsum(W.to_numpy(), axis=0)])
def _wmean(cs, i, H):                      # mean over hours i-H+1..i, window clipped at data start
    lo = np.maximum(0, i - H + 1); return (cs[i + 1] - cs[lo]) / (i + 1 - lo)[:, None]
def _rep(a): return np.repeat(a, N, axis=0)

def build_ctx(W, stations, train_end="2024-08-31 12:00"):
    xy = stations[["x_km", "y_km"]].to_numpy()
    D = np.sqrt(((xy[:, None] - xy[None]) ** 2).sum(2)); np.fill_diagonal(D, np.inf)
    R = []
    for r in (10, 25, 50):
        M = (D <= r).astype(float); e = M.sum(1) == 0
        M[e, D[e].argmin(1)] = 1.0; R.append(M / M.sum(1, keepdims=True))
    return dict(A10=B.neighbour_matrix(stations), R=R, st_mean=W.loc[:train_end].mean().to_numpy())

def own(W, T, ctx):
    L = B.lag_stack(W, T, 24); return L.reshape(-1, 24), [f"lag{k}" for k in range(24)]

def prevday(W, T, ctx):
    Wv, i = W.to_numpy(), _i(W, T); assert (i - 168 >= 0).all()
    X = np.stack([Wv[i - 24 * k] for k in range(1, 8)], axis=2)
    return X.reshape(-1, 7), [f"d{k}" for k in range(1, 8)]

def trend(W, T, ctx):
    cs, i = _cs(W), _i(W, T)
    cur = _wmean(cs, i, 24)                              # mean of hours i-23..i
    lo, hi = np.maximum(0, i - 47), i - 24               # previous 24 h window, clipped at data start
    prev = (cs[hi + 1] - cs[lo]) / (hi + 1 - lo)[:, None]
    return (cur - prev).reshape(-1, 1), ["trend24"]

def long(W, T, ctx):
    cs, i = _cs(W), _i(W, T)
    X = np.stack([_wmean(cs, i, 168), _wmean(cs, i, 720), np.tile(ctx["st_mean"], (len(i), 1))], axis=2)
    return X.reshape(-1, 3), ["mean7d", "mean30d", "train_mean"]

def nbr(W, T, ctx):
    cs, i = _cs(W), _i(W, T); Wv = W.to_numpy()
    cols = [Wv[i] @ ctx["A10"].T, _wmean(cs, i, 24) @ ctx["A10"].T] + [Wv[i] @ M.T for M in ctx["R"]]
    return np.stack(cols, axis=2).reshape(-1, 5), ["nbr10_now", "nbr10_mean24", "nbr_r10", "nbr_r25", "nbr_r50"]

def network(W, T, ctx):
    Wv, i = W.to_numpy(), _i(W, T)
    now = Wv[i].mean(1); chg = now - Wv[i - 24].mean(1); sd = Wv[i].std(1)
    return _rep(np.column_stack([now, chg, sd])), ["net_now", "net_chg24", "net_std"]

def calendar(W, T, ctx):
    dow = (pd.DatetimeIndex(T) + pd.Timedelta(hours=24)).dayofweek.to_numpy()   # day of the TARGET
    return _rep(np.eye(7)[dow][:, 1:]), [f"dow{k}" for k in range(1, 7)]

def transform(W, T, ctx):
    Wv, i = W.to_numpy(), _i(W, T)
    cols = [np.log1p(Wv[i]), np.log1p(Wv[i] @ ctx["A10"].T), np.log1p(np.repeat(Wv[i].mean(1)[:, None], N, 1))]
    return np.stack(cols, axis=2).reshape(-1, 3), ["log_lag0", "log_nbr10", "log_net"]

GROUPS = dict(own=own, prevday=prevday, trend=trend, long=long, nbr=nbr, network=network, calendar=calendar, transform=transform)
