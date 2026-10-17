"""Template for the forecast.py you submit in Part 5.

Your submission folder should contain this file (renamed forecast.py) plus any files it loads,
e.g. model.pkl and bihar_stations.csv. The grader imports this module once and then calls
forecast(history, issue_time) for 30 issue times. Do NOT refit or update model parameters inside
forecast(); computing features from `history` is fine.

As written, this template returns the persistence forecast so that it runs out of the box.
Replace the marked part with your own model.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent

# Load fitted objects once, at import time (not inside forecast()).
MODEL_PATH = HERE / "model.pkl"
MODEL = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
# STATIONS = pd.read_csv(HERE / "bihar_stations.csv", dtype={"sensor_id": str}).set_index("sensor_id")


def forecast(history: pd.DataFrame, issue_time: pd.Timestamp) -> pd.Series:
    """history: wide hourly table (DatetimeIndex 'timestamp', one string column per sensor_id),
    every hour from 1 Jan 2024 00:00 up to and including issue_time.
    issue_time: 12:00 on some day (IST).
    Returns predicted PM2.5 for issue_time + 24 h, indexed by sensor_id (all 468 stations)."""
    window = history.loc[issue_time - pd.Timedelta(hours=23): issue_time]   # last 24 hours

    # ---- replace this with your model -------------------------------------------------
    prediction = window.iloc[-1].to_numpy(dtype=float)                      # persistence
    # e.g. X = build_features(window, ...); prediction = MODEL.predict(X)
    # -------------------------------------------------------------------------------------

    return pd.Series(np.clip(prediction, 0, None), index=history.columns, name="pm25_forecast")
