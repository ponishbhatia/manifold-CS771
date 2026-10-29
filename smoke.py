# run from e.g. your home folder:  python smoke.py
import sys, importlib.util
from pathlib import Path
import pandas as pd

REPO = Path("")                      # <- edit
SUB = REPO / "submission"
spec = importlib.util.spec_from_file_location("forecast", SUB / "forecast.py")
fc = importlib.util.module_from_spec(spec); spec.loader.exec_module(fc)   # same way the grader imports it

W = pd.read_csv(REPO / "data/bihar_pm25_hourly_jan_nov_2024.csv", index_col="timestamp", parse_dates=True)
W.columns = W.columns.astype(str)

t = pd.Timestamp("2024-11-12 12:00")
out = fc.forecast(W.loc[:t].copy(), t)

assert isinstance(out, pd.Series) and len(out) == 468
assert list(out.index) == list(W.columns), "index must be the 468 sensor_ids"
assert out.notna().all() and (out >= 0).all()
truth = W.loc[t + pd.Timedelta(hours=24)]
print("RMSE", ((out - truth) ** 2).mean() ** 0.5, "| persistence", ((W.loc[t] - truth) ** 2).mean() ** 0.5)