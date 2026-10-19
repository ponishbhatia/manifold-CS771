"""Feature extraction for Part 2: Weighted k-NN Analog Forecasting.

Features are organised into six groups (16 columns in total) so that whole
groups can be switched on and off for ablation studies. Every feature needs
at most 24 hours of history, so there are no NaNs and all 113,256 training
rows (Jan 3 - Aug 31) and 42,588 validation rows (Sep 1 - Nov 30) are usable.
"""
from __future__ import annotations

import sys
from collections.abc import Sequence
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd

import baseline as B

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
N_STATIONS = 468
N_NEIGHBOURS = 75
HISTORY_HOURS = 24          # length of the lag window requested from the data
DAYS_PER_YEAR = 365         # period of the annual calendar encoding

# Lag-window sizes (in hours) used by individual feature groups.
SHORT_WINDOW = 2            # own_lags / own_summary
MOMENTUM_WINDOW = 6         # trend: last 6h vs. previous 6h
TREND_WINDOW = 18           # trend / spatial: "yesterday 18:00 -> today 12:00"

# Index of the lag (hours before the issue time) that closes each window.
LAG_NOW = 0
LAG_TREND = TREND_WINDOW - 1        # 17 h ago
LAG_DAY_AGO = HISTORY_HOURS - 1     # 23 h ago

FeatureBlocks = list[tuple[str, np.ndarray]]


# --------------------------------------------------------------------------- #
# Feature groups (each returns an array of shape (n_days * N_STATIONS, k))
# --------------------------------------------------------------------------- #
def _flat(a: np.ndarray) -> np.ndarray:
    """Flatten a (days, stations) array into a (rows, 1) column."""
    return a.reshape(-1, 1)


def _own_lags(L: np.ndarray) -> np.ndarray:
    """The station's most recent hourly readings (t and t-1h)."""
    return L[:, :, :SHORT_WINDOW].reshape(-1, SHORT_WINDOW)


def _own_summary(L: np.ndarray) -> np.ndarray:
    """Mean / min / max / std over the last SHORT_WINDOW hours."""
    recent = L[:, :, :SHORT_WINDOW]
    stats = (recent.mean(axis=2), recent.min(axis=2),
             recent.max(axis=2), recent.std(axis=2))
    return np.column_stack([_flat(s) for s in stats])


def _trend(L: np.ndarray) -> np.ndarray:
    """Rate of change and momentum (18h change, 6h-vs-6h momentum, deviation)."""
    now = L[:, :, LAG_NOW]
    change_18h = now - L[:, :, LAG_TREND]
    momentum = (L[:, :, :MOMENTUM_WINDOW].mean(axis=2)
                - L[:, :, MOMENTUM_WINDOW:2 * MOMENTUM_WINDOW].mean(axis=2))
    dev_from_mean = now - L[:, :, :TREND_WINDOW].mean(axis=2)
    return np.column_stack([_flat(change_18h), _flat(momentum), _flat(dev_from_mean)])


def _spatial_neighbours(L: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Neighbourhood context: neighbour mean now / over 18h, local anomaly."""
    now = L[:, :, LAG_NOW]
    nbr_now = now @ A.T
    # Averaging over time first is equivalent to (and much cheaper than)
    # applying A to every lag and averaging afterwards (A is linear).
    nbr_mean_18h = L[:, :, :TREND_WINDOW].mean(axis=2) @ A.T
    local_anomaly = now - nbr_now
    return np.column_stack([_flat(nbr_now), _flat(nbr_mean_18h), _flat(local_anomaly)])


def _network(L: np.ndarray) -> np.ndarray:
    """Network-wide mean level now and its 24h change (same for all stations)."""
    net_now = L[:, :, LAG_NOW].mean(axis=1)
    net_day_ago = L[:, :, LAG_DAY_AGO].mean(axis=1)
    cols = (net_now, net_now - net_day_ago)
    return np.column_stack([np.repeat(c, N_STATIONS) for c in cols])


def _calendar(T: pd.DatetimeIndex) -> np.ndarray:
    """Annual cyclical encoding of the issue date."""
    phase = 2 * np.pi * (T.dayofyear.to_numpy() - 1) / DAYS_PER_YEAR
    return np.column_stack([np.repeat(np.sin(phase), N_STATIONS),
                            np.repeat(np.cos(phase), N_STATIONS)])


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def extract_features(
    W: pd.DataFrame,
    T: pd.DatetimeIndex | Sequence[pd.Timestamp],
    stations: pd.DataFrame | None = None,
    A: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, list[int]]]:
    """Compute the feature matrix for the given issue times.

    Args:
        W: Hourly PM2.5, shape (hours, N_STATIONS).
        T: Issue times (12:00 noon each day).
        stations: Station metadata; only needed when ``A`` is not given.
        A: (N_STATIONS, N_STATIONS) neighbour matrix; built from ``stations``
            (loaded if necessary) when omitted.

    Returns:
        X: array of shape (len(T) * N_STATIONS, 16).
        groups: mapping of group name -> list of column indices in ``X``.

    Raises:
        ValueError: if the resulting matrix contains NaNs.
    """
    T = pd.DatetimeIndex(T)

    if A is None:
        if stations is None:
            _, stations = B.load_data()
        A = B.neighbour_matrix(stations, k=N_NEIGHBOURS)

    # L[i, s, k] = reading of station s at T[i] - k hours, k = 0..HISTORY_HOURS-1
    L = B.lag_stack(W, T, hours=HISTORY_HOURS)

    blocks: FeatureBlocks = [
        ("own_lags", _own_lags(L)),
        ("own_summary", _own_summary(L)),
        ("trend", _trend(L)),
        ("spatial_nbr", _spatial_neighbours(L, A)),
        ("network", _network(L)),
        ("calendar", _calendar(T)),
    ]

    X = np.column_stack([block for _, block in blocks])

    groups: dict[str, list[int]] = {}
    start = 0
    for name, block in blocks:
        groups[name] = list(range(start, start + block.shape[1]))
        start += block.shape[1]

    if np.isnan(X).any():
        raise ValueError("NaN found in feature extraction")
    return X, groups


# --------------------------------------------------------------------------- #
# Verification
# --------------------------------------------------------------------------- #
def verify_features() -> None:
    """Check shapes, group layout, NaNs and absence of data leakage."""
    print("Running feature verification checks...")
    W, stations = B.load_data()
    T_train, T_val = B.issue_times()
    A = B.neighbour_matrix(stations, k=N_NEIGHBOURS)

    X_train, groups = extract_features(W, T_train, stations, A)
    X_val, _ = extract_features(W, T_val, stations, A)

    n_features = sum(len(cols) for cols in groups.values())
    for name, X, T in (("train", X_train, T_train), ("val", X_val, T_val)):
        expected = (len(T) * N_STATIONS, n_features)
        print(f"X_{name} shape: {X.shape} (expected {expected})")
        assert X.shape == expected, f"Unexpected {name} shape"
    assert X_train.shape == (113_256, 16)
    assert X_val.shape == (42_588, 16)

    print("Feature groups summary:")
    for name, cols in groups.items():
        print(f"  - {name}: {len(cols)} features (columns {cols[0]} to {cols[-1]})")

    # Leakage: features at t from the full series must equal those from data up to t.
    t = T_val[10]
    X_full, _ = extract_features(W, [t], stations, A)
    X_sliced, _ = extract_features(W.loc[:t], [t], stations, A)
    assert np.allclose(X_full, X_sliced), "Data leakage check failed!"
    print("Zero leakage confirmed: features at t use only readings at or before t.")
    print("All checks passed successfully!")


if __name__ == "__main__":
    verify_features()
