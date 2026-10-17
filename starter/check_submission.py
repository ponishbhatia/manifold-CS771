"""Check your Part 5 submission the way the grader will run it.

Usage (from the package root):
    python starter/check_submission.py path/to/your_submission_folder

It imports forecast.py from that folder in a separate process and calls forecast(history, issue_time)
for the 30 issue times 31 Oct 12:00 ... 29 Nov 12:00 (the grader uses 30 Nov 12:00 ... 29 Dec 12:00).
It checks the output format and the 10-second limit, then prints the November RMSE as a sanity check.
November is part of your own data, so this is NOT a test score.
"""
import argparse, importlib.util, json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "bihar_pm25_hourly_jan_nov_2024.csv"
TIME_LIMIT_S = 10.0
ISSUE_TIMES = pd.date_range("2024-10-31 12:00", "2024-11-29 12:00", freq="D")


def load_history(path):
    W = pd.read_csv(path, index_col="timestamp", parse_dates=True)
    W.columns = W.columns.astype(str)
    return W


def worker(sub_dir, data_path, out_path):
    """Runs inside a child process: import the submission and call forecast() for each issue time."""
    W = load_history(data_path)
    sub_dir = Path(sub_dir).resolve()
    sys.path.insert(0, str(sub_dir)); os.chdir(sub_dir)
    spec = importlib.util.spec_from_file_location("forecast", sub_dir / "forecast.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    preds, timings, errors = {}, {}, {}
    for t in ISSUE_TIMES:
        key = str(t)
        try:
            t0 = time.perf_counter()
            out = mod.forecast(W.loc[:t].copy(), t)
            timings[key] = time.perf_counter() - t0
            if not isinstance(out, pd.Series):
                raise TypeError(f"forecast must return a pd.Series, got {type(out).__name__}")
            out.index = out.index.astype(str)
            missing = set(W.columns) - set(out.index)
            extra = set(out.index) - set(W.columns)
            if missing or extra:
                raise ValueError(f"index must be exactly the 468 sensor_ids "
                                 f"({len(missing)} missing, {len(extra)} unexpected)")
            if out.index.duplicated().any():
                raise ValueError("duplicate sensor_ids in output")
            vals = pd.to_numeric(out.reindex(W.columns), errors="coerce").to_numpy(dtype=float)
            if not np.isfinite(vals).all():
                raise ValueError(f"{(~np.isfinite(vals)).sum()} forecasts are missing or not finite")
            preds[key] = vals.tolist()
        except Exception as e:                                   # record and continue
            errors[key] = f"{type(e).__name__}: {e}"
    json.dump({"preds": preds, "timings": timings, "errors": errors}, open(out_path, "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("submission", help="folder containing forecast.py")
    ap.add_argument("--worker-out", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.worker_out:
        worker(a.submission, DATA, a.worker_out); return

    sub = Path(a.submission).resolve()
    if not (sub / "forecast.py").exists():
        sys.exit(f"No forecast.py in {sub}")
    out_path = Path.cwd() / ".check_submission_result.json"
    print(f"Running {sub / 'forecast.py'} on {len(ISSUE_TIMES)} November issue times ...")
    try:
        r = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(sub), "--worker-out", str(out_path)],
                           timeout=len(ISSUE_TIMES) * TIME_LIMIT_S + 120, capture_output=True, text=True)
    except subprocess.TimeoutExpired:
        sys.exit("FAILED: the run took too long overall (likely an infinite loop or very slow calls).")
    if r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-4000:])
        sys.exit("FAILED: your forecast.py could not be imported or crashed outside forecast().")
    res = json.load(open(out_path)); out_path.unlink()

    ok = True
    if res["errors"]:
        ok = False
        print(f"\nERRORS in {len(res['errors'])} of {len(ISSUE_TIMES)} calls, e.g.:")
        for k, v in list(res["errors"].items())[:3]:
            print(f"  {k}: {v}")
    slow = {k: v for k, v in res["timings"].items() if v > TIME_LIMIT_S}
    if slow:
        ok = False
        print(f"\nTOO SLOW: {len(slow)} calls took more than {TIME_LIMIT_S:.0f} s (max {max(slow.values()):.1f} s)")
    if res["timings"]:
        print(f"\nCall time: median {np.median(list(res['timings'].values())):.2f} s, "
              f"max {max(res['timings'].values()):.2f} s")

    if res["preds"]:
        W = load_history(DATA)
        keys = sorted(res["preds"])
        P = np.array([res["preds"][k] for k in keys])
        T = np.array([W.loc[pd.Timestamp(k) + pd.Timedelta(hours=24)].to_numpy() for k in keys])
        Q = np.array([W.loc[pd.Timestamp(k)].to_numpy() for k in keys])
        rmse = lambda a: np.sqrt(np.mean((a - T) ** 2))
        print(f"\nNovember sanity check over {len(keys)} forecasts (NOT a test score):")
        print(f"  your forecast  RMSE {rmse(P):6.2f}  MAE {np.mean(np.abs(P - T)):6.2f}")
        print(f"  persistence    RMSE {rmse(Q):6.2f}  MAE {np.mean(np.abs(Q - T)):6.2f}")
    print("\nRESULT:", "OK - format and timing look right." if ok else "PROBLEMS FOUND - see above.")


if __name__ == "__main__":
    main()
