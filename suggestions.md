# Suggestions

Everyone runs their own experiments. These rules keep results comparable
and let the winning model drop into one notebook that runs top to bottom.

## Shared (required)
- Load data, split and score only through `baseline.py`:
  `load_data()`, `issue_times()`, `targets()`, `evaluate()`, `persistence()`,
  `part1_ridge_predict()`. Run `B.self_check()` first; it must print `baseline.py OK`.
- Split by target date: train = targets 3 Jan–31 Aug, val = targets 1 Sep–30 Nov.
  Noon rows: 113,256 train and 42,588 val.
- Scores are in µg/m³ with predictions clipped at 0. Convert log or change
  targets back before scoring.
- No leakage: features for issue time t use only readings at or before t.
  Anything fitted (means, scalers, PCA, clusters) is fitted on training rows only.
  No random splits or k-fold.
- Seeds 0, 1, 2. Relative paths. Versions from `requirements.txt`.

## Your features (required)
- Write features as functions of `(history, issue_time)`, relative to t,
  in a `.py` file. If your model wins, forecast.py and Part 3 reuse them.
- Organise them into 3–8 named groups, ready for the ablation.

## Sharing results
- Validation predictions: `preds/<name>_<model>.parquet`,
  91 rows (target dates) × 468 columns (sensor_id), µg/m³.
- Results log (shared sheet), one row per reported run: name, model,
  feature groups, target, hyperparameters, train RMSE, val RMSE, val MAE,
  Sep/Oct/Nov RMSE, notes.

## Suggested features (try any on your model)
| Group | Ideas |
|---|---|
| own | last 24 hourly readings, or their mean/min/max/std |
| prevday | readings at target − 24k h for k = 1..7 |
| trend | mean of last 24 h minus mean of the 24 h before |
| long | station's 7- and 30-day mean; station's training-period mean |
| nbr | mean of neighbours within 10/25/50 km; IDW-weighted mean |
| network | network mean at t and its 24 h change; top PCA components of the 468-station snapshot |
| cluster | k-means cluster id or cluster mean |
| calendar | day of week; sin/cos of day of year (may hurt: Sep–Nov unseen in training) |
| transform | log1p of any of the above |

Target options: level, log1p, or change from persistence (y(t+24h) − y(t)).

## Suggested tests (run before trusting a result)
1. Shape: one row per (issue_time, station); 468 rows per issue time.
2. No NaNs.
3. Leakage: features from the full history equal features from history
   cut off at t:
   `assert f(history, t).equals(f(history.loc[:t], t))`
4. Parity: features for one issue time equal that issue time's rows in
   the batch computation, so forecast.py matches training.
5. Baselines: `B.self_check()` reproduces the reference numbers
   (persistence 32.75 / 17.47, Part 1 ridge 29.58 / 18.16).
