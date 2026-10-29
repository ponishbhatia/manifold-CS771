"""Part 5: fit the final Part 2 model (Sanath's log1p Laplace linear model, smeared-mean forecast) and
save it as plain JSON for forecast.py.

Run from the repo root:
    python part5/fit_final_model.py                      # final refit: targets 3 Jan - 30 Nov  -> submission/model.json
    python part5/fit_final_model.py --train-end "2024-08-31 12:00" --out /tmp/val_model.json   # validation protocol (tests)

`--train-end` is the last TARGET time used for fitting; issue times are target - 24 h, exactly as in
baseline.issue_times(). Features, hyper-parameters (l2 = 1e-3, smeared-mean forecast, 6 groups, 25 features) are
the ones chosen on validation in models/sanath_laplace_hetero.ipynb; nothing is re-tuned here.
"""

"""For the report

Training rows: state “155,844 rows: all 468 stations × 333 issue days, targets 3 Jan–30 Nov”.
Why all rows: keep the early-January rows even though their 7-day window is clipped. In my log-space experiments, dropping the early-window rows made validation clearly worse (28.49 → 30.82 RMSE). January holds the only high-pollution training regime, so it is the closest analogue to November and December. That result came from my own model, so say it as supporting evidence, not a result for Sanath’s model.
Honest limit: the 5 points for beating persistence on December depend on a month we can’t see. Evidence so far is consistent: the model beats persistence by about 14% RMSE in each of Sep, Oct and Nov, and by about 14% in the held-out November run."""

import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import baseline as B
import features_C as FC
import probmodels_C as PM

ap = argparse.ArgumentParser()
ap.add_argument("--train-end", default="2024-11-30 12:00", help="last target time used for fitting")
ap.add_argument("--out", default=str(ROOT / "submission" / "model.json"))
a = ap.parse_args()

W, stations = B.load_data()
T_train, _ = B.issue_times(24, train_end=a.train_end)          # issue times; targets 3 Jan ... train_end
y = B.targets(W, T_train)
assert (T_train + np.timedelta64(24, "h")).max() <= np.datetime64(a.train_end), "a training target is after train_end"
ctx = FC.build_ctx(W, stations)                                # FINAL_GROUPS only use the neighbour matrices (no fitted statistics)
X, names, _ = FC.extract_features(W, T_train, ctx, FC.FINAL_GROUPS)
assert X.shape == (len(y), 25), X.shape

m = PM.LogLaplaceForecaster(l2=1e-3, mode="mean").fit(X, y)    # scaler, IRLS Laplace fit and smearing: training rows only
lap = m.model_
model = dict(
    feature_names=names, groups=list(FC.FINAL_GROUPS), l2=1e-3,
    x_mean=lap.scaler_.mean_.tolist(), x_scale=lap.scaler_.scale_.tolist(),
    coef=lap.coef_.tolist(),                                   # [intercept, 25 weights] on standardised log1p features
    smearing=float(m.smearing_), max_pm25=PM.MAX_PM25,
    trained_through_target=a.train_end, n_train_rows=int(len(y)),
    n_issue_days=int(len(T_train)),
)
Path(a.out).parent.mkdir(parents=True, exist_ok=True)
json.dump(model, open(a.out, "w"), indent=1)
print(f"saved {a.out}: {len(y):,} rows ({len(T_train)} issue days x 468), smearing = {m.smearing_:.4f}")