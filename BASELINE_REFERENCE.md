# Baseline reference for Part 2

`baseline.py` holds everything from Part 1 that Part 2 experiments need. Put it at the package root (next to `data/` and `starter/`) and run `python baseline.py` once. It should print `baseline.py OK`.

## Rules

- Every flat array is ordered issue time first, then station in `bihar_stations.csv` order.
- A row is train or val by its target date: train targets 3 Jan – 31 Aug (242 days, 113,256 rows), val targets 1 Sep – 30 Nov (91 days, 42,588 rows).
- Features for issue time t use only readings at or before t. Anything fitted (means, scalers, PCA, clusters) is fitted on training rows only.
- Score only with `B.evaluate` or `B.compare`: µg/m³, forecasts clipped at 0. Convert log or change targets back first.

## Validation baselines (to beat)

| | RMSE | MAE | RMSE Sep | MAE Sep | RMSE Oct | MAE Oct | RMSE Nov | MAE Nov |
|---|---|---|---|---|---|---|---|---|
| Persistence | 32.75 | 17.47 | 23.33 | 8.91 | 20.22 | 11.97 | 47.81 | 31.70 |
| Ridge (Part 1) | 29.58 | 18.16 | 20.34 | 10.35 | 19.36 | 13.46 | 43.06 | 30.81 |

## Minimal experiment

```python
import numpy as np
import baseline as B

W, stations = B.load_data()
T_train, T_val = B.issue_times()
y_train, y_val = B.targets(W, T_train), B.targets(W, T_val)

# features: start from Part 1's 29 and add your own (same row order)
A = B.neighbour_matrix(stations)
X_train = B.part1_features(W, T_train, A)
X_val   = B.part1_features(W, T_val, A)

model = ...                                   # fit on X_train, y_train only
pred_val = model.predict(X_val)

B.evaluate(pred_val, y_val, T_val)            # dict: RMSE, MAE, per month
B.compare({"My model": pred_val}, W, T_val, stations=stations)   # next to both baselines
```

## What is in `baseline.py`

| Function | Use |
|---|---|
| `load_data()` | `W` (hours × 468) and `stations` (with `x_km`, `y_km`) |
| `issue_times(lead=24)` | `(T_train, T_val)`; `lead=1/3/6/12` for Part 3; `train_targets=(..., "2024-11-30 12:00")` for the Part 5 refit |
| `targets(W, T)` | flat targets for issue times T |
| `row_index(W, T)` | `(issue_time, sensor_id)` MultiIndex if you want a DataFrame |
| `lag_stack(W, T, hours)` | array `[time, station, k]` = reading at t − k h; building block for history features |
| `neighbour_matrix(stations, k)` | averaging matrix for k nearest stations; `readings @ A.T` gives neighbour means |
| `distance_matrix(stations)` | 468 × 468 km, for radius-based features |
| `part1_features(W, T, A)` | the 29 Part 1 features; names in `PART1_FEATURE_NAMES` |
| `persistence(W, T)` | persistence forecasts |
| `part1_ridge_predict(W, T, lead)` | Part 1 ridge forecasts (refits for the lead) |
| `evaluate(pred, y, T)` | RMSE/MAE overall and per month |
| `compare({name: pred}, W, T)` | results table with both baselines on top |
| `daily_rmse(pred, y, T)` | station RMSE per day (Part 4) |
| `to_wide(pred, W, T)` | forecasts as (target times × sensor_id) DataFrame |

## Log every reported run in the shared sheet

Member, Model, Features, Target, Hyperparameters, Train RMSE, Val RMSE, Val MAE, Sep/Oct/Nov RMSE, Notes.
