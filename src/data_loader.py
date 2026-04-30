"""Load, clean, grid, and aggregate NASA FIRMS / raw hotspot CSVs.

End product: a daily cell-day dataframe with columns
    [lat_grid, lon_grid, date, fire_count, frp_sum, frp_mean, frp_max,
     bright_mean, bright_max, confidence_mean]

Densification: by default we expand sparse fire-only rows into a dense
(active-cell × every-day) grid where days with no detection are filled with
zeros. This gives correct "yesterday/last-week" semantics for rolling features.
"""

from __future__ import annotations

import glob
import logging
import os
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

log = logging.getLogger("data_loader")

DEFAULT_GRID = 0.1

AGG_COLUMNS = [
    "fire_count",
    "frp_sum",
    "frp_mean",
    "frp_max",
    "bright_mean",
    "bright_max",
    "confidence_mean",
]

# Names produced by fetch_weather.py — must match features.WEATHER_COLUMNS.
WEATHER_COLUMNS = ["temp_max", "temp_min", "precip_sum", "wind_max", "et0"]


def _coerce_acq_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """Build a datetime from FIRMS' split acq_date / acq_time columns."""
    df = df.copy()
    df["acq_datetime"] = pd.to_datetime(
        df["acq_date"].astype(str)
        + " "
        + df["acq_time"].astype(str).str.zfill(4),
        errors="coerce",
    )
    return df


def _parse_confidence(series: pd.Series) -> pd.Series:
    """VIIRS reports confidence as l/n/h letters; MODIS reports 0-100 integers."""
    letter_map = {"l": 0, "n": 50, "h": 100}
    mapped = series.astype(str).str.lower().map(letter_map)
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def _resolve_paths(paths_or_globs: Iterable[str]) -> List[str]:
    resolved: List[str] = []
    for p in paths_or_globs:
        if not p:
            continue
        if any(ch in p for ch in "*?["):
            resolved.extend(sorted(glob.glob(p)))
        elif os.path.isdir(p):
            resolved.extend(sorted(glob.glob(os.path.join(p, "*.csv"))))
        elif os.path.exists(p):
            resolved.append(p)
    return resolved


def load_firms_csv(paths_or_globs: Iterable[str]) -> pd.DataFrame:
    files = _resolve_paths(paths_or_globs)
    if not files:
        raise FileNotFoundError(
            f"No FIRMS CSVs found at: {list(paths_or_globs)}. Run fetch_firms.py first."
        )

    log.info("Loading %d FIRMS CSV file(s)", len(files))
    frames = [pd.read_csv(f) for f in files]
    return pd.concat(frames, ignore_index=True)


def clean_hotspots(
    df: pd.DataFrame,
    min_confidence: int = 0,
    drop_frp_outliers: bool = True,
    frp_quantile: float = 0.999,
) -> pd.DataFrame:
    """Standardize columns, coerce numerics, apply quality + outlier filters."""
    df = _coerce_acq_datetime(df)
    df = df.rename(columns={"latitude": "lat", "longitude": "lon"})

    df["bright_main"] = np.nan
    if "bright_ti4" in df.columns:
        df["bright_main"] = df["bright_ti4"]
    if "bright" in df.columns:
        df["bright_main"] = df["bright_main"].fillna(df["bright"])

    if "confidence" in df.columns:
        df["confidence"] = _parse_confidence(df["confidence"])
    else:
        df["confidence"] = np.nan

    for col in ["lat", "lon", "frp", "bright_main", "confidence"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["date"] = df["acq_datetime"].dt.date
    df = df.dropna(subset=["lat", "lon", "date", "frp"])

    df = df[(df["frp"] >= 0) & (df["frp"].notna())]
    if drop_frp_outliers and len(df) > 1000:
        cap = df["frp"].quantile(frp_quantile)
        n_before = len(df)
        df = df[df["frp"] <= cap]
        log.info(
            "Capped FRP at q%.3f=%.1f, dropped %d rows",
            frp_quantile,
            cap,
            n_before - len(df),
        )

    if min_confidence > 0:
        n_before = len(df)
        df = df[df["confidence"].fillna(0) >= min_confidence]
        log.info(
            "Confidence filter ≥%d dropped %d rows",
            min_confidence,
            n_before - len(df),
        )

    keep = ["lat", "lon", "date", "frp", "bright_main", "confidence", "acq_datetime"]
    return df[keep].reset_index(drop=True)


def grid_and_aggregate(
    df: pd.DataFrame, grid_size: float = DEFAULT_GRID
) -> pd.DataFrame:
    """Snap each detection to a lat/lon grid cell and aggregate to one row per cell-day."""
    df = df.copy()
    df["lat_grid"] = (df["lat"] / grid_size).round() * grid_size
    df["lon_grid"] = (df["lon"] / grid_size).round() * grid_size

    daily = df.groupby(["lat_grid", "lon_grid", "date"], as_index=False).agg(
        fire_count=("frp", "count"),
        frp_sum=("frp", "sum"),
        frp_mean=("frp", "mean"),
        frp_max=("frp", "max"),
        bright_mean=("bright_main", "mean"),
        bright_max=("bright_main", "max"),
        confidence_mean=("confidence", "mean"),
    )

    return daily.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)


def densify_active_cells(daily: pd.DataFrame) -> pd.DataFrame:
    """Expand to (active-cell × every-day-in-range) so rolling features see real zero-days.

    Active cell = cell with at least one detection in the dataset. Inactive cells
    are excluded entirely (they have no signal to learn from).
    """
    if daily.empty:
        return daily

    cells = daily[["lat_grid", "lon_grid"]].drop_duplicates().reset_index(drop=True)
    date_range = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D").date
    dense_idx = cells.assign(_k=1).merge(
        pd.DataFrame({"date": date_range, "_k": 1}), on="_k"
    ).drop(columns="_k")

    dense_idx["date"] = pd.to_datetime(dense_idx["date"]).dt.date
    daily = daily.copy()
    daily["date"] = pd.to_datetime(daily["date"]).dt.date

    out = dense_idx.merge(daily, on=["lat_grid", "lon_grid", "date"], how="left")
    out[AGG_COLUMNS] = out[AGG_COLUMNS].fillna(0.0)

    log.info(
        "Densified %d active cells × %d days = %d rows",
        len(cells),
        len(date_range),
        len(out),
    )
    return out.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)


def merge_weather(daily: pd.DataFrame, weather_path: Optional[str]) -> pd.DataFrame:
    """Left-join real ERA5 weather (from fetch_weather.py cache) onto the daily frame.

    No-op if weather_path is missing or the file doesn't exist. The cache must
    have columns [lat_grid, lon_grid, date, temp_max, temp_min, precip_sum,
    wind_max, et0]. Cells / dates not in the cache get NaN, which features.py
    converts to 0 only at model-input time.
    """
    if not weather_path or not os.path.exists(weather_path):
        return daily

    try:
        wx = pd.read_csv(weather_path)
    except Exception as exc:
        log.warning("Could not read weather cache (%s): %s — skipping merge.", weather_path, exc)
        return daily

    if wx.empty:
        return daily

    expected = {"lat_grid", "lon_grid", "date", *WEATHER_COLUMNS}
    missing = expected - set(wx.columns)
    if missing:
        log.warning(
            "Weather cache is missing columns %s — skipping merge.", sorted(missing)
        )
        return daily

    wx["date"] = pd.to_datetime(wx["date"]).dt.date
    wx["lat_grid"] = wx["lat_grid"].round(6)
    wx["lon_grid"] = wx["lon_grid"].round(6)

    daily = daily.copy()
    daily["lat_grid"] = daily["lat_grid"].round(6)
    daily["lon_grid"] = daily["lon_grid"].round(6)

    merged = daily.merge(
        wx[["lat_grid", "lon_grid", "date", *WEATHER_COLUMNS]],
        on=["lat_grid", "lon_grid", "date"],
        how="left",
    )
    log.info(
        "Merged weather: %d / %d rows have real ERA5 values",
        int(merged[WEATHER_COLUMNS[0]].notna().sum()),
        len(merged),
    )
    return merged


def load_and_prepare(
    raw_dir: Optional[str],
    firms_path: Optional[str],
    grid_size: float = DEFAULT_GRID,
    min_confidence: int = 0,
    densify: bool = True,
    weather_path: Optional[str] = None,
) -> pd.DataFrame:
    """One-shot: glob raw_dir + firms_path → cleaned, gridded, daily aggregate.

    If ``weather_path`` is provided and exists, real ERA5 daily aggregates are
    left-joined onto the densified frame. Run ``fetch_weather.py`` to populate.
    """
    sources: List[str] = []
    if raw_dir:
        sources.append(raw_dir)
    if firms_path:
        sources.append(firms_path)

    raw = load_firms_csv(sources)
    log.info("Loaded %d raw hotspot rows", len(raw))

    cleaned = clean_hotspots(raw, min_confidence=min_confidence)
    log.info("After cleaning: %d rows", len(cleaned))

    daily = grid_and_aggregate(cleaned, grid_size=grid_size)
    log.info(
        "Aggregated to %d cell-day rows across %d cells",
        len(daily),
        daily[["lat_grid", "lon_grid"]].drop_duplicates().shape[0],
    )

    if densify:
        daily = densify_active_cells(daily)

    daily = merge_weather(daily, weather_path)
    return daily
