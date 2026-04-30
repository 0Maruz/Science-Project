"""Time-series + spatial + calendar feature engineering and label construction.

All features are derived from REAL data sources only:
    - NASA FIRMS VIIRS NRT hotspots → fire_count, frp_*, bright_*, confidence_*
      (loaded by data_loader.load_and_prepare and densified to cell-day rows)
    - Spatial neighborhood = 8 surrounding grid cells, sourced from the same FIRMS frame
    - Calendar = real datetime values (month, day_of_year, Thailand Jan-Apr burn season)
    - Optional weather (when present in the daily frame): real ECMWF ERA5 reanalysis
      via Open-Meteo Archive API, cached by fetch_weather.py. NOT generated.

No synthetic / random / interpolated features. If a real source isn't available
for a given column, the column is simply not added to FEATURES.

Densified input is required so that rolling / lag windows are anchored to real
calendar days (not the previous fire day).

`FEATURES_CORE` is the always-on input contract. Use `resolve_features(df)` (or
read `dataset_info.json["features"]`) to get the exact list the deployed model
was trained on, since weather columns are added dynamically.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import pandas as pd

log = logging.getLogger("features")

MAX_PREDICTION_DAYS = 7
LAG_DAYS: Tuple[int, ...] = (1, 2, 3, 7, 14, 30)
ROLL_WINDOWS: Tuple[int, ...] = (3, 7, 14, 30)
NEIGHBOR_LAGS: Tuple[int, ...] = (1, 3, 7)
NEIGHBOR_ROLLS: Tuple[int, ...] = (3, 7)

GROUP_KEYS = ["lat_grid", "lon_grid"]

# Optional weather columns produced by fetch_weather.py + data_loader.merge_weather().
# Each must be REAL ECMWF reanalysis or measured weather, not derived/imputed.
WEATHER_COLUMNS: Tuple[str, ...] = (
    "temp_max",
    "temp_min",
    "precip_sum",
    "wind_max",
    "et0",
)
WEATHER_LAGS: Tuple[int, ...] = (1, 3, 7)
WEATHER_ROLLS: Tuple[int, ...] = (3, 7)

UrgencyLevel = Literal["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]

# Default fixed cutoffs — used as a fallback when no calibrated thresholds
# have been persisted to dataset_info.json yet (e.g. before first training run).
DEFAULT_URGENCY_THRESHOLDS: Dict[str, float] = {
    "CRITICAL": 0.0,   # days <= 0
    "HIGH": 2.0,       # days <= 2
    "MEDIUM": 4.0,     # days <= 4
    "LOW": float(MAX_PREDICTION_DAYS),  # days <= 7
}


# ─────────────────────────────────────────────
# Urgency calibration — derived from REAL val
# predictions, not arbitrary fixed numbers.
# ─────────────────────────────────────────────
def calibrate_urgency_thresholds(
    val_predictions: np.ndarray,
    horizon: int = MAX_PREDICTION_DAYS,
    quantiles: Tuple[float, float, float] = (0.25, 0.5, 0.75),
) -> Dict[str, float]:
    """Compute urgency cutoffs from the real validation prediction distribution.

    Lower predicted ``days_until_fire`` = more urgent. The cutoffs are simply the
    25 / 50 / 75 percentiles of the actual model output on the held-out
    validation slice — so the four buckets each carry roughly 25 % of the real
    predictions, giving stable, distribution-aware tiers instead of arbitrary
    fixed numbers.

    Returns a dict of upper-bound (inclusive) thresholds:
        {"CRITICAL": p25, "HIGH": p50, "MEDIUM": p75, "LOW": horizon}
    """
    arr = np.asarray(val_predictions, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return dict(DEFAULT_URGENCY_THRESHOLDS)

    q1, q2, q3 = np.quantile(arr, quantiles)
    # Enforce monotonicity in case the distribution is degenerate (e.g. all
    # predictions identical) and clamp to the prediction horizon.
    q1 = float(min(q1, horizon))
    q2 = float(min(max(q2, q1), horizon))
    q3 = float(min(max(q3, q2), horizon))

    return {
        "CRITICAL": q1,
        "HIGH": q2,
        "MEDIUM": q3,
        "LOW": float(horizon),
    }


def urgency_from_thresholds(
    days: float,
    thresholds: Optional[Dict[str, float]] = None,
) -> UrgencyLevel:
    """Map a (real model output) days-until-fire value to an urgency tier."""
    t = thresholds or DEFAULT_URGENCY_THRESHOLDS
    d = float(days)
    if d <= t["CRITICAL"]:
        return "CRITICAL"
    if d <= t["HIGH"]:
        return "HIGH"
    if d <= t["MEDIUM"]:
        return "MEDIUM"
    if d <= t["LOW"]:
        return "LOW"
    return "NONE"


def get_urgency(days: int) -> UrgencyLevel:
    """Backwards-compatible fixed-threshold mapping. Prefer
    ``urgency_from_thresholds`` with calibrated thresholds where available."""
    return urgency_from_thresholds(days, DEFAULT_URGENCY_THRESHOLDS)


# ─────────────────────────────────────────────
# Spatial neighbours — real FIRMS data, just
# aggregated over the 8 surrounding grid cells.
# ─────────────────────────────────────────────
def add_neighbor_features(daily: pd.DataFrame, grid_size: float = 0.1) -> pd.DataFrame:
    """For each cell-day, sum fire_count and frp_sum over the 8 adjacent cells.

    Pure spatial aggregation of REAL FIRMS detections — no smoothing or
    interpolation. Cells without any neighbour activity get exact zero.
    """
    df = daily.copy()
    df["lat_grid"] = df["lat_grid"].round(6)
    df["lon_grid"] = df["lon_grid"].round(6)

    base = df[["lat_grid", "lon_grid", "date", "fire_count", "frp_sum"]].copy()

    df["neighbor_fire_today"] = 0.0
    df["neighbor_frp_today"] = 0.0

    offsets = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ]

    for dlat, dlon in offsets:
        # Shift each source row's coords by the *negative* offset so that the
        # row's (lat_grid, lon_grid) now equals the TARGET cell that has this
        # row as its (dlat, dlon) neighbour.
        shifted = base.copy()
        shifted["lat_grid"] = (shifted["lat_grid"] - dlat * grid_size).round(6)
        shifted["lon_grid"] = (shifted["lon_grid"] - dlon * grid_size).round(6)
        shifted = shifted.rename(
            columns={"fire_count": "_nfire", "frp_sum": "_nfrp"}
        )
        merged = df.merge(
            shifted[["lat_grid", "lon_grid", "date", "_nfire", "_nfrp"]],
            on=["lat_grid", "lon_grid", "date"],
            how="left",
        )
        df["neighbor_fire_today"] = (
            df["neighbor_fire_today"].to_numpy()
            + merged["_nfire"].fillna(0).to_numpy()
        )
        df["neighbor_frp_today"] = (
            df["neighbor_frp_today"].to_numpy()
            + merged["_nfrp"].fillna(0).to_numpy()
        )

    return df


# ─────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────
def _ensure_sorted(df: pd.DataFrame) -> pd.DataFrame:
    return df.sort_values(GROUP_KEYS + ["date"]).reset_index(drop=True)


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Lag, rolling-window, and trend features per cell.

    Uses ONLY observed FIRMS counts/FRP and the spatial-neighbour aggregates
    that were derived from the same real FIRMS frame.
    """
    df = _ensure_sorted(df).copy()
    grp = df.groupby(GROUP_KEYS, sort=False)

    # ── per-cell self lags / rolls ──
    for lag in LAG_DAYS:
        df[f"fire_lag_{lag}"] = grp["fire_count"].shift(lag).fillna(0)
        df[f"frp_lag_{lag}"] = grp["frp_sum"].shift(lag).fillna(0)

    for w in ROLL_WINDOWS:
        rolled_fire = grp["fire_count"].rolling(w, min_periods=1).sum()
        rolled_frp_sum = grp["frp_sum"].rolling(w, min_periods=1).sum()
        rolled_frp_max = grp["frp_max"].rolling(w, min_periods=1).max()
        rolled_active = (
            grp["fire_count"]
            .rolling(w, min_periods=1)
            .apply(lambda x: float((x > 0).sum()), raw=True)
        )

        df[f"fire_sum_{w}d"] = rolled_fire.reset_index(level=GROUP_KEYS, drop=True)
        df[f"frp_sum_{w}d"] = rolled_frp_sum.reset_index(level=GROUP_KEYS, drop=True)
        df[f"frp_max_{w}d"] = rolled_frp_max.reset_index(level=GROUP_KEYS, drop=True)
        df[f"active_days_{w}d"] = rolled_active.reset_index(level=GROUP_KEYS, drop=True)

    df["frp_trend_1d"] = grp["frp_sum"].diff().fillna(0)
    df["frp_trend_3d"] = grp["frp_sum"].diff(3).fillna(0)

    df["fire_count_today"] = df["fire_count"].fillna(0)
    df["frp_sum_today"] = df["frp_sum"].fillna(0)
    df["bright_mean_today"] = df["bright_mean"].fillna(0)
    df["confidence_mean_today"] = df["confidence_mean"].fillna(0)

    # ── neighbour lags / rolls (spatial signal from REAL adjacent-cell fires) ──
    if "neighbor_fire_today" in df.columns:
        for lag in NEIGHBOR_LAGS:
            df[f"neighbor_fire_lag_{lag}"] = (
                grp["neighbor_fire_today"].shift(lag).fillna(0)
            )
            df[f"neighbor_frp_lag_{lag}"] = (
                grp["neighbor_frp_today"].shift(lag).fillna(0)
            )
        for w in NEIGHBOR_ROLLS:
            rolled_n_fire = grp["neighbor_fire_today"].rolling(w, min_periods=1).sum()
            rolled_n_frp = grp["neighbor_frp_today"].rolling(w, min_periods=1).sum()
            df[f"neighbor_fire_sum_{w}d"] = rolled_n_fire.reset_index(
                level=GROUP_KEYS, drop=True
            )
            df[f"neighbor_frp_sum_{w}d"] = rolled_n_frp.reset_index(
                level=GROUP_KEYS, drop=True
            )

    # ── weather lags / rolls (only when REAL weather columns are present) ──
    weather_cols_present = [c for c in WEATHER_COLUMNS if c in df.columns]
    if weather_cols_present:
        for col in weather_cols_present:
            # Today's value = real measurement / reanalysis at the base date
            df[f"{col}_today"] = df[col].fillna(0)
            for lag in WEATHER_LAGS:
                df[f"{col}_lag_{lag}"] = grp[col].shift(lag).fillna(0)
            for w in WEATHER_ROLLS:
                rolled = grp[col].rolling(w, min_periods=1).mean()
                df[f"{col}_mean_{w}d"] = rolled.reset_index(level=GROUP_KEYS, drop=True)

    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cyclic month / day-of-year and a Thailand burn-season flag (Jan–Apr).

    All values come directly from the row's real ``date`` field.
    """
    df = df.copy()
    dt = pd.to_datetime(df["date"])
    df["month"] = dt.dt.month
    df["day_of_year"] = dt.dt.dayofyear
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12.0)
    df["doy_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365.0)
    df["doy_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365.0)
    df["is_burn_season"] = dt.dt.month.between(1, 4).astype(int)
    return df


def make_label_days_until_fire(
    df: pd.DataFrame, horizon: int = MAX_PREDICTION_DAYS
) -> pd.DataFrame:
    """Vectorized label: days until the next strict-future fire in this cell, ≤ horizon.

    Requires `df` to be densified (consecutive calendar days per cell). Rows with
    no fire in the next `horizon` days get label = -1 and are excluded from
    training downstream.
    """
    df = _ensure_sorted(df).copy()
    labels = np.full(len(df), -1, dtype=np.int8)
    fire_today = (df["fire_count"].fillna(0) > 0).astype(np.int8)

    for k in range(1, horizon + 1):
        shifted = (
            df.groupby(GROUP_KEYS, sort=False)["fire_count"]
            .shift(-k)
            .fillna(0)
            .to_numpy()
        )
        future_fire = shifted > 0
        unset = labels == -1
        labels = np.where(future_fire & unset, k, labels)

    df["days_until_fire"] = labels.astype(int)
    df["fire_today"] = fire_today.astype(int)
    return df


def build_features(
    daily: pd.DataFrame,
    horizon: int = MAX_PREDICTION_DAYS,
    grid_size: float = 0.1,
) -> pd.DataFrame:
    """End-to-end feature construction from a daily cell-day frame.

    Order matters: spatial neighbours must be computed BEFORE temporal lags so
    that ``neighbor_fire_lag_*`` etc. can be derived from the per-day neighbour
    aggregate.
    """
    df = add_neighbor_features(daily, grid_size=grid_size)
    df = add_temporal_features(df)
    df = add_calendar_features(df)
    df = make_label_days_until_fire(df, horizon=horizon)
    log.info(
        "Built features for %d rows, %d positive labels (fire within %d days)",
        len(df),
        int((df["days_until_fire"] >= 0).sum()),
        horizon,
    )
    return df


# ─────────────────────────────────────────────
# Feature contract
# ─────────────────────────────────────────────
def _build_core_feature_list() -> List[str]:
    cols: List[str] = []
    cols += [f"fire_lag_{l}" for l in LAG_DAYS]
    cols += [f"frp_lag_{l}" for l in LAG_DAYS]
    for w in ROLL_WINDOWS:
        cols += [
            f"fire_sum_{w}d",
            f"frp_sum_{w}d",
            f"frp_max_{w}d",
            f"active_days_{w}d",
        ]
    cols += ["frp_trend_1d", "frp_trend_3d"]
    cols += [
        "fire_count_today",
        "frp_sum_today",
        "bright_mean_today",
        "confidence_mean_today",
    ]
    # Spatial neighbour signals (always emitted by add_neighbor_features)
    cols += ["neighbor_fire_today", "neighbor_frp_today"]
    cols += [f"neighbor_fire_lag_{l}" for l in NEIGHBOR_LAGS]
    cols += [f"neighbor_frp_lag_{l}" for l in NEIGHBOR_LAGS]
    for w in NEIGHBOR_ROLLS:
        cols += [f"neighbor_fire_sum_{w}d", f"neighbor_frp_sum_{w}d"]
    # Calendar signals
    cols += ["month_sin", "month_cos", "doy_sin", "doy_cos", "is_burn_season"]
    cols += ["lat_grid", "lon_grid"]
    return cols


def _build_weather_feature_list() -> List[str]:
    cols: List[str] = []
    for c in WEATHER_COLUMNS:
        cols.append(f"{c}_today")
        for lag in WEATHER_LAGS:
            cols.append(f"{c}_lag_{lag}")
        for w in WEATHER_ROLLS:
            cols.append(f"{c}_mean_{w}d")
    return cols


FEATURES_CORE: Tuple[str, ...] = tuple(_build_core_feature_list())
FEATURES_WEATHER: Tuple[str, ...] = tuple(_build_weather_feature_list())

# Backwards-compat alias. Consumers that need the *exact* deployed-model
# contract should call resolve_features(df) or read dataset_info.json["features"].
FEATURES: Tuple[str, ...] = FEATURES_CORE


def resolve_features(df: pd.DataFrame) -> List[str]:
    """Return the feature list actually present in `df`, including weather
    columns when fetch_weather.py has supplied them. Pure column existence
    check — no heuristics."""
    feats: List[str] = [c for c in FEATURES_CORE if c in df.columns]
    feats += [c for c in FEATURES_WEATHER if c in df.columns]
    return feats
