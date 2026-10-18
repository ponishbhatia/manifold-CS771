# Baseline reference for Part 2

`baseline.py` has the shared Part 1 code. Put it next to `data/` and run `python baseline.py` once; it should print `baseline.py OK` (it checks the numbers below and a leakage test).

It replaces `core.py` from `suggestions.md`: use these functions for loading, splitting and scoring.

## Rules

- Flat arrays are ordered issue time first, then station (`bihar_stations.csv` order).
- A row is train or val by its **target** date: train targets 3 Jan – 31 Aug (242 days, 113,256 rows), val targets 1 Sep – 30 Nov (91 days, 42,588 rows). 1 Jan and 2 Jan are not targets because 24 h of history is needed.
- Features for issue time t use only readings at or before t. Anything fitted (means, scalers, PCA, clusters) is fitted on training rows only.
- Score only with `B.evaluate` / `B.compare`: µg/m³, forecasts clipped at 0. Convert log or change targets back first.

## Validation baselines (to beat)

| | RMSE | MAE | RMSE Sep | MAE Sep | RMSE Oct | MAE Oct | RMSE Nov | MAE Nov |
|---|---|---|---|---|---|---|---|---|
| Persistence | 32.75 | 17.47 | 23.33 | 8.91 | 20.22 | 11.97 | 47.81 | 31.70 |
| Ridge (Part 1) | 29.58 | 18.16 | 20.34 | 10.35 | 19.36 | 13.46 | 43.06 | 30.81 |

## Minimal experiment

```python
import baseline as B

W, stations = B.load_data()
T_train, T_val = B.issue_times()
y_train, y_val = B.targets(W, T_train), B.targets(W, T_val)

A = B.neighbour_matrix(stations)
X_train = B.part1_features(W, T_train, A)      # start from the 29 Part 1 features,
X_val   = B.part1_features(W, T_val, A)        # add your own columns (same row order)

model = ...                                    # fit on X_train, y_train only
pred_val = model.predict(X_val)

B.evaluate(pred_val, y_val, T_val)                       # dict: RMSE, MAE, per month
B.compare({"My model": pred_val}, W, stations, T_val)    # table next to both baselines
```

## Functions

| Function | Use |
|---|---|
| `load_data()` | `W` (hours × 468) and `stations` (with `x_km`, `y_km`) |
| `issue_times(lead=24)` | `(T_train, T_val)`. Part 3: `lead=1,3,6,12`. Part 5 refit: `train_end="2024-11-30 12:00"` |
| `targets(W, T, lead=24)` | flat targets for issue times `T` |
| `lag_stack(W, T, hours)` | `L[time, station, k]` = reading at t − k h; building block for history features |
| `neighbour_matrix(stations, k=10)` | averaging matrix; `readings @ A.T` gives neighbour means |
| `part1_features(W, T, A)` | the 29 Part 1 features (names in `FEATURE_NAMES`); works for any lead and a single issue time |
| `persistence(W, T)` | persistence forecasts |
| `part1_ridge_predict(W, stations, T, lead=24)` | Part 1 ridge forecasts, refitted for the lead |
| `evaluate(pred, y, T, lead=24)` | RMSE/MAE overall and per month |
| `compare({name: pred}, W, stations, T, lead=24)` | results table with both baselines on top |

For Part 3, pass the same `lead` to `issue_times`, `targets`, `evaluate` and `compare`.

## Log every reported run in the shared sheet

Member, Model, Features, Target, Hyperparameters, Train RMSE, Val RMSE, Val MAE, Sep/Oct/Nov RMSE, Notes.
