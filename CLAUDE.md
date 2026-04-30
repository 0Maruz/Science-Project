# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Wildfire **date** prediction system for Thailand (BBOX `96,4,107,22`). Given satellite hotspot history from NASA FIRMS (real data only — no simulation), a tree-ensemble regressor predicts `days_until_fire` (1–7) per spatial grid cell. Predictions are rendered on a Leaflet map with urgency tiers (CRITICAL/HIGH/MEDIUM/LOW).

This was previously a binary risk classifier; the current system is regression-based and predicts *when* a fire will occur. **Do not reintroduce classification semantics.**

### Hard rule: real data only

Every feature must come from a real, measurable source. The pipeline currently consumes:

- **NASA FIRMS VIIRS NRT** (always on) — hotspot detections; powers fire/FRP/brightness/confidence and all spatial-neighbour features.
- **Open-Meteo Archive API** (optional, no key) — real ECMWF ERA5 daily reanalysis (temp_max/min, precip_sum, wind_max, et0). Activated by running `python fetch_weather.py`, which caches to `data/weather/weather_cache.csv`.
- **Calendar** — derived from each row's real `date`.

No synthetic, simulated, randomly-generated, or interpolated values anywhere. If a real source isn't available for a given column, the column is simply not added to `FEATURES`. Don't introduce fake fallbacks or fabricated defaults — when something is missing, it's missing.

## Common commands

All Python entry points use bare imports of each other (`from features import FEATURES`), so they must be run from inside `src/`:

```bash
# 1. Pull latest VIIRS NRT hotspots from NASA FIRMS into data/firms/firms_all.csv
cd src && python fetch_firms.py [--days 1-10]

# 1b. (OPTIONAL) Pull real ERA5 weather for every active FIRMS cell into
#      data/weather/weather_cache.csv. Requires no API key. Skip this and the
#      training pipeline silently runs without weather features.
cd src && python fetch_weather.py [--start YYYY-MM-DD] [--end YYYY-MM-DD] [--limit-cells N]

# 2. Train the model (runs the full data → features → tune → save pipeline)
cd src && python train.py [--n-iter 20] [--n-splits 5] [--only lightgbm,xgboost]

# `training.py` is a thin shim around `train.main()` for backwards compatibility.

# 3. Regenerate the GeoJSON map without retraining
cd src && python risk_map.py

# 4. Serve the API (FastAPI on :8000)
cd src && uvicorn api:app --reload
```

Dependencies: `pip install -r requirements.txt` into `.venv/`. Requires a `.env` file with `FIRMS_API_KEY` (see `.env.example` for the full set). All scripts use `load_dotenv()`.

There is no test suite, linter, or build step.

## Architecture

### Module layout (in `src/`)

| File | Role |
|---|---|
| `fetch_firms.py` | Fetches VIIRS NRT hotspots from NASA FIRMS with retry/backoff; writes accumulative `data/firms/firms_all.csv`. |
| `fetch_weather.py` | **Optional.** Fetches real ECMWF ERA5 daily aggregates from Open-Meteo Archive (no key) for every active FIRMS cell, caches to `data/weather/weather_cache.csv`. Idempotent — only fetches missing (cell, date) tuples. |
| `data_loader.py` | Pure I/O: loads raw + FIRMS CSVs, cleans, snaps to grid, aggregates to daily cell-day, **densifies active cells** over the date range, and (if `weather_path` is supplied) left-joins the weather cache. |
| `features.py` | Lag/rolling/calendar + **3×3 spatial-neighbour** feature engineering, label generation, and **percentile-based urgency calibration**. Owns `FEATURES_CORE` (always-on) and `FEATURES_WEATHER` (only used when ERA5 columns are present). Use `resolve_features(df)` to get the deployed-model contract. |
| `model.py` | Candidate factory (RandomForest, LightGBM, XGBoost), `RandomizedSearchCV` tuner using `TimeSeriesSplit`, evaluation (`MAE`, `RMSE`, `R²`, `acc±1`). |
| `train.py` | Orchestrator: load → features → label → chronological 60/20/20 split → tune all candidates → pick best val MAE → held-out test eval → calibrate urgency thresholds from val predictions → refit on train+val → persist → trigger `risk_map.run()`. |
| `risk_map.py` | Loads the trained model + densified feature CSV, predicts for the latest base date, attaches calibrated urgency + `historical_fire_count_30d` (real FIRMS) + raw model output, and appends to `fire_dates_all.geojson`. Top-level GeoJSON metadata carries thresholds + held-out test metrics for the frontend. |
| `api.py` | FastAPI on top of the same artifacts — `/predictions/today`, `/predictions/timeline`, `/predictions/day/{n}`, `/predict/location`, `/metrics`, `/geojson`. Reads calibrated thresholds from `dataset_info.json`. |

### Pipeline

```
data/raw/*.csv  ──┐
                  ├─► data_loader ─► features ─► train ─► outputs/models/*.pkl
data/firms/      ─┘    (densify)    (lag/roll/   (RF, LGBM, XGB)
firms_all.csv                        calendar)         │
                                                       ▼
                                            outputs/features/full_features.csv
                                            outputs/metadata/dataset_info.json
                                                       │
                                                       ▼
                                                 risk_map.py
                                                       │
                                                       ▼
                                       outputs/riskmap/fire_dates_all.geojson
                                                       │
                                       ┌───────────────┴───────────────┐
                                       ▼                               ▼
                                    api.py                      frontend/app.js
                                  (FastAPI)               (fetches geojson directly)
```

### Two parallel data sources, intentionally merged

`data_loader.load_and_prepare()` ingests **two separate hotspot sources** and concatenates them:
- `RAW_DIR` (`data/raw/*.csv`) — historical bulk archive
- `FIRMS_PATH` (`data/firms/firms_all.csv`) — NRT data accumulated by `fetch_firms.py`

Both are gridded to `GRID_SIZE` (default 0.1°) cells via `(coord / GRID).round() * GRID`, then aggregated to one row per `(lat_grid, lon_grid, date)`.

### Densification (important)

After aggregation, `data_loader.densify_active_cells()` expands the sparse fire-only frame into a **dense (active-cell × every-day-in-range)** grid, with no-fire days filled with zeros. This is what makes lag features (`fire_lag_1` = literally yesterday) and rolling windows (`fire_sum_7d` = last 7 calendar days) correct. Without densification, "yesterday" would mean "the previous day this cell happened to burn", which is a major source of bias in the original system. **Inactive cells (zero fires ever)** are excluded — they have no signal to learn from.

### Label semantics

`features.make_label_days_until_fire()` walks each cell's densified history and labels each row with the number of days until the next fire (1–7). Rows with no fire in the next 7 days get `-1` and are dropped from training. This means **the model only learns from cells that did go on to burn**; at inference time it still emits a 0–7 value for every cell, which the urgency mapping turns into CRITICAL/HIGH/MEDIUM/LOW.

### Feature contract — single source of truth

`features.py` exposes two tuples and a resolver:

- `FEATURES_CORE` (~53 columns) — always present. Lags at 1/2/3/7/14/30 days, rolls at 3/7/14/30 days, active-day counts, FRP trend, current-day signals, cyclic month/DOY, burn-season flag, lat/lon, **3×3 spatial-neighbour** fire/FRP aggregates with their own lags & rolls.
- `FEATURES_WEATHER` (~30 columns) — only emitted when `data/weather/weather_cache.csv` exists. Per-variable today + lags 1/3/7 + rolls 3/7 over real ERA5 temp_max/temp_min/precip_sum/wind_max/et0.
- `resolve_features(df)` — returns the actually-present subset given a feature dataframe. **Use this** (or `dataset_info.json["features"]` after training) instead of hardcoding a list.

`api.py` and `risk_map.py` resolve the feature list at runtime by reading `outputs/metadata/dataset_info.json["features"]` first (matches the deployed model exactly), and fall back to `resolve_features(df)`. **Don't re-introduce hardcoded feature lists in those files.**

### Spatial neighbours (real FIRMS, not synthetic)

`features.add_neighbor_features()` sums `fire_count` and `frp_sum` across the 8 cells surrounding each `(lat_grid, lon_grid)` for the same date. Implementation shifts the source frame's coords by negative offsets so each row's `(lat_grid, lon_grid)` becomes the *target* cell whose neighbour it is, then merges. Pure aggregation — no smoothing, no interpolation, zero when a neighbour cell has no detection.

Spatial features must be computed *before* `add_temporal_features()` so the lags/rolls of `neighbor_fire_today` / `neighbor_frp_today` exist downstream.

### Urgency thresholds — calibrated, not arbitrary

`features.calibrate_urgency_thresholds(val_predictions)` computes 25/50/75 percentiles of the **real** model output on the held-out validation slice (computed in `train.py` *before* the train+val refit, so the test set stays untouched). The result lands in `dataset_info.json["urgency_thresholds"]` and is the only source of truth for `api.py` / `risk_map.py` / the frontend. The legacy fixed `0/2/4/7` cutoffs only appear as a `DEFAULT_URGENCY_THRESHOLDS` fallback when no metadata exists yet.

`features.urgency_from_thresholds(days, thresholds)` is the canonical mapping. `get_urgency()` is kept as a backwards-compat alias that uses the default thresholds.

### Model selection

`train.chronological_split()` produces a 3-way **60 / 20 / 20** train / val / test split sorted by date — train tunes, val selects, test is held out and never seen during tuning or selection.

`model.select_best()` runs `RandomizedSearchCV` (default 20 iterations, 5 folds) for each candidate against a `TimeSeriesSplit` — never a random shuffle, since rows are temporally ordered. Selection criterion is best **validation** MAE. After selection, the winner is evaluated **once** on the held-out test set (`test_metrics` in `dataset_info.json`), then refit on train+val and saved to `outputs/models/lgbm_fire_date_model.pkl`. The filename is historical (kept stable for `api.py` / `risk_map.py`) — the actual model class can be RF, LightGBM, or XGBoost; `dataset_info.json["best_model"]` tells you which. The deployed artefact is **not** re-evaluated after the train+val refit, to avoid leakage.

If you change the candidate pool or feature list, **delete `outputs/models/*.pkl` before retraining** to avoid loading a stale artifact.

### "confidence" is a rounding proxy, not a probability

Both `api.py` (`_rounding_confidence`) and `risk_map.py` (`prediction_confidence`) compute `1 - |raw_pred - rounded_pred|`. This is **not** a calibrated likelihood — it only reflects how close the regressor's continuous output landed to a whole number. Don't treat it as a probability in downstream UI or aggregations, and don't add fake calibration without changing the underlying model. The frontend tooltip labels it "rounding proxy" for this reason.

### Frontend day selector

`frontend/app.js` exposes a Day 1–7 selector that filters predicted markers by the model's clipped `days_until_fire` integer — pure filtering of real outputs, no smoothing or interpolation. Clicking a row in the timeline also triggers the same filter. The validation-metrics panel reads `metadata.metrics` written into the GeoJSON by `risk_map.append_geojson` (sourced from `dataset_info.json["model"]["test_metrics"]`).

### Frontend reads GeoJSON directly, not the API

`frontend/app.js` fetches `../outputs/riskmap/fire_dates_all.geojson` via a **relative path** — it does not hit `api.py`. To view the dashboard, the frontend must be served such that `../outputs/...` resolves (e.g. serve the project root, then open `/frontend/index.html`). The FastAPI server exists for programmatic access but is not what the dashboard depends on.

### GeoJSON is appended, not overwritten

`risk_map.append_geojson` reads the existing `fire_dates_all.geojson`, strips out predictions whose `base_date` matches the current run (preserving `source: "observed"` features), and appends the new observed + predicted features. Historical predictions for prior base dates accumulate. Delete the file to reset.

### Path resolution

All entry points resolve paths via `BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` so they work regardless of cwd. `train.py` additionally reads `RAW_DIR` / `FIRMS_PATH` / `OUTPUT_DIR` from `.env`, and **resolves any relative env values against `BASE_DIR` via `_resolve()`** — so `RAW_DIR=./data/raw` in `.env.example` works whether you launch from the project root or from `src/`. If you set absolute paths in `.env`, they pass through unchanged.

### Gotchas

- **`fetch_firms.py` HTTP 400 across all datasets** = bad / rate-limited `MAP_KEY`. Check status via `https://firms.modaps.eosdis.nasa.gov/mapserver/mapkey_status/?MAP_KEY=…`; FIRMS resets the daily transaction limit roughly every 24h. Training does not require a successful fetch — it can run on whatever is already in `data/raw/` + `data/firms/firms_all.csv`.
- **`uvicorn api:app` startup `RuntimeError: Model not found`** = no trained artifact yet. Run `train.py` to completion first; the API loads `outputs/models/lgbm_fire_date_model.pkl` at startup and refuses to serve without it.
- **Stale model after feature changes**: if you add/remove anything in `FEATURES_CORE`/`FEATURES_WEATHER`, or you start/stop running `fetch_weather.py`, delete `outputs/models/*.pkl` before retraining. `api.py` and `risk_map.py` resolve the feature list from `dataset_info.json` to stay in sync, but a leftover `.pkl` from a different feature contract will silently mispredict.
- **Weather cache lag**: Open-Meteo's archive endpoint trails real-time by ~5 days (ERA5T preliminary release). `fetch_weather.py` automatically caps `end_date` at `today - 5d` — for the most recent days, weather columns will be NaN and `features.add_temporal_features` fills NaN with 0 only at model-input time. The cache itself preserves the genuine missing-data signal; do not impute.
