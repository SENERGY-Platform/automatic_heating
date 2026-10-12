# Automatic Heating

Learns when a window is opened, separately for weekdays and weekends, and predicts for each day when the heating should be turned down before the next opening.

## Input and output

- Input `window_open`: a boolean. With `contact_sensor` true it comes from a contact sensor and is read inverted (true = contact closed = window closed).
- Output per cluster of openings, on the first message of a new day with more than two openings of that day type: `stopping_time` (start of the cluster minus `inertia_buffer`, UTC), `overall_confidence` and `timestamp`.

## Training

The operator is an `MLOperator` of analytics-operator-lib-python. At start, when no model is registered in MLflow, and once a day it trains on the history of its input (`history_days`), merged with the openings it saw live since the last training. The model is stored in MLflow. There is no learning phase: with enough openings in the history it predicts from the first day change on.

An input without readable history, such as the output of another operator in a new pipeline, starts cold and predicts once more than two openings of a day type are seen. A pipeline of an earlier version keeps what it learned: its `window_opening_times.pickle` is merged once at the first training.

## Configuration

| Name | Default | Meaning |
|---|---|---|
| `high_confidence_boundary` | 600 | spread of a cluster in seconds below which the confidence is 1 |
| `low_confidence_boundary` | 3600 | spread in seconds above which the confidence is 0 |
| `inertia_buffer` | 10 | minutes the stopping time lies before the cluster |
| `confidence_days` | 7 | days considered for the daily appearance of a cluster |
| `contact_sensor` | true | input comes from a contact sensor; any non-empty value switches it on, only an empty value off |
| `history_days` | 60 | days of history read at each training |

`init_phase_length` and `init_phase_level` of earlier versions are ignored. The other values are read as before; a value that is no number stops the operator at start.

## Versions

Every build on `master` pushes `:vX.Y.Z` and `:prod`; a branch `release/vN` builds only `:vN.x.y`. See `.github/workflows/prod.yml`.
