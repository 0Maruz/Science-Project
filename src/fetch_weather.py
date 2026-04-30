"""Open-Meteo weather fetcher — REAL ECMWF ERA5 reanalysis only.

Fetches daily temperature / precipitation / wind / evapotranspiration values
for every active FIRMS grid cell over the dataset's full date range and caches
the result to ``data/weather/weather_cache.csv``.

Key properties (per the project rule: real data only):
    • Source = Open-Meteo's free Archive API, which serves ECMWF ERA5 / ERA5-Land
      reanalysis (https://open-meteo.com/en/docs/historical-weather-api).
      No API key, no commercial restriction for non-commercial use.
    • Each cached row corresponds to a real (lat_grid, lon_grid, date) tuple
      that appears in the densified FIRMS dataset.
    • No interpolation, no fabrication. If the API returns a NULL for a given
      (cell, date), the column is left empty and downstream features.py will
      treat it as a 0-fill *only* at model-input time — the cache itself
      preserves the genuine missing-data signal.

Usage::

    cd src && python fetch_weather.py [--limit-cells N] [--start YYYY-MM-DD] [--end YYYY-MM-DD]

After running, train.py / data_loader.py automatically detect the cache and
merge it onto the daily frame, so weather features become part of the model
input contract on the next training run.
"""

from __future__ import annotations

import argparse
import logging
import os
import time
from datetime import date, datetime, timedelta
from typing import List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from data_loader import grid_and_aggregate, clean_hotspots, load_firms_csv

load_dotenv()

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
log = logging.getLogger("fetch_weather")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _resolve(base_dir: str, value: Optional[str]) -> Optional[str]:
    if not value:
        return value
    return value if os.path.isabs(value) else os.path.normpath(os.path.join(base_dir, value))


RAW_DIR     = _resolve(BASE_DIR, os.getenv("RAW_DIR"))     or os.path.join(BASE_DIR, "data", "raw")
FIRMS_PATH  = _resolve(BASE_DIR, os.getenv("FIRMS_PATH"))  or os.path.join(BASE_DIR, "data", "firms", "firms_all.csv")
WEATHER_DIR = _resolve(BASE_DIR, os.getenv("WEATHER_DIR")) or os.path.join(BASE_DIR, "data", "weather")
WEATHER_CACHE_PATH = os.path.join(WEATHER_DIR, "weather_cache.csv")

# ECMWF ERA5 archive — no key required, ~5-day lag for the most recent dates.
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Real, measurable daily aggregates from ERA5 reanalysis.
DAILY_VARS = [
    "temperature_2m_max",
    "temperature_2m_min",
    "precipitation_sum",
    "wind_speed_10m_max",
    "et0_fao_evapotranspiration",
]
RENAME = {
    "temperature_2m_max": "temp_max",
    "temperature_2m_min": "temp_min",
    "precipitation_sum": "precip_sum",
    "wind_speed_10m_max": "wind_max",
    "et0_fao_evapotranspiration": "et0",
}

TIMEZONE = os.getenv("TIMEZONE", "Asia/Bangkok")
GRID_SIZE = float(os.getenv("GRID_SIZE", "0.1"))
ARCHIVE_LAG_DAYS = 5  # ERA5T preliminary data lag


# ─────────────────────────────────────────────────────────────────────────────
# HTTP session
# ─────────────────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    s = requests.Session()
    retry = Retry(
        total=5,
        backoff_factor=2.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        respect_retry_after_header=True,
    )
    adapter = HTTPAdapter(max_retries=retry)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


# ─────────────────────────────────────────────────────────────────────────────
# FIRMS → active cells / date range
# ─────────────────────────────────────────────────────────────────────────────

def discover_active_cells() -> tuple[pd.DataFrame, date, date]:
    """Return (cells_df, min_date, max_date) from the densified FIRMS frame."""
    sources = []
    if RAW_DIR:
        sources.append(RAW_DIR)
    if FIRMS_PATH:
        sources.append(FIRMS_PATH)

    raw = load_firms_csv(sources)
    cleaned = clean_hotspots(raw, min_confidence=0)
    daily = grid_and_aggregate(cleaned, grid_size=GRID_SIZE)

    cells = (
        daily[["lat_grid", "lon_grid"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    return cells, daily["date"].min(), daily["date"].max()


# ─────────────────────────────────────────────────────────────────────────────
# Open-Meteo fetch (real ERA5 reanalysis)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_one_cell(
    session: requests.Session,
    lat: float,
    lon: float,
    start: date,
    end: date,
) -> pd.DataFrame:
    """Fetch a single grid cell's ERA5 daily aggregates over [start, end]."""
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join(DAILY_VARS),
        "timezone": TIMEZONE,
    }
    res = session.get(ARCHIVE_URL, params=params, timeout=60)
    res.raise_for_status()
    payload = res.json()

    daily = payload.get("daily")
    if not daily or "time" not in daily:
        return pd.DataFrame()

    df = pd.DataFrame(daily)
    df = df.rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df = df.rename(columns=RENAME)
    df["lat_grid"] = round(lat, 6)
    df["lon_grid"] = round(lon, 6)
    keep = ["lat_grid", "lon_grid", "date", *RENAME.values()]
    return df[keep]


# ─────────────────────────────────────────────────────────────────────────────
# Idempotent cache update
# ─────────────────────────────────────────────────────────────────────────────

def update_cache(
    cells: pd.DataFrame,
    start: date,
    end: date,
    cache_path: str = WEATHER_CACHE_PATH,
    sleep_between_calls: float = 0.15,
    limit_cells: Optional[int] = None,
) -> int:
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    session = _make_session()

    # Cap end at archive availability (ERA5T has a ~5-day lag).
    today = datetime.utcnow().date()
    end = min(end, today - timedelta(days=ARCHIVE_LAG_DAYS))
    if start > end:
        log.warning("Start (%s) is after archive-available end (%s); nothing to fetch.", start, end)
        return 0

    # Load existing cache so we only fetch missing (cell, date) tuples.
    if os.path.exists(cache_path) and os.path.getsize(cache_path) > 0:
        cache = pd.read_csv(cache_path)
        cache["date"] = pd.to_datetime(cache["date"]).dt.date
        existing = set(zip(cache["lat_grid"].round(6), cache["lon_grid"].round(6), cache["date"]))
    else:
        cache = pd.DataFrame()
        existing = set()

    cells = cells.copy()
    cells["lat_grid"] = cells["lat_grid"].round(6)
    cells["lon_grid"] = cells["lon_grid"].round(6)
    if limit_cells:
        cells = cells.head(limit_cells)

    log.info(
        "Fetching weather for %d cells over %s → %s (timezone=%s)",
        len(cells), start, end, TIMEZONE,
    )

    new_frames: List[pd.DataFrame] = []
    fetched_cells = 0

    for idx, row in cells.iterrows():
        lat, lon = float(row["lat_grid"]), float(row["lon_grid"])

        try:
            df = fetch_one_cell(session, lat, lon, start, end)
        except requests.RequestException as exc:
            log.error("Fetch failed for (%.3f, %.3f): %s", lat, lon, exc)
            continue

        if df.empty:
            continue

        # Drop tuples we already have cached.
        keys = list(zip(df["lat_grid"].round(6), df["lon_grid"].round(6), df["date"]))
        mask = [k not in existing for k in keys]
        df = df[mask]

        if not df.empty:
            new_frames.append(df)
            fetched_cells += 1

        if (idx + 1) % 25 == 0:
            log.info("  progress: %d / %d cells", idx + 1, len(cells))

        time.sleep(sleep_between_calls)

    if not new_frames:
        log.info("No new weather rows fetched (cache already up to date).")
        return 0

    fresh = pd.concat(new_frames, ignore_index=True)
    if not cache.empty:
        combined = pd.concat([cache, fresh], ignore_index=True)
    else:
        combined = fresh

    combined = combined.drop_duplicates(subset=["lat_grid", "lon_grid", "date"])
    combined = combined.sort_values(["lat_grid", "lon_grid", "date"]).reset_index(drop=True)
    combined.to_csv(cache_path, index=False)

    log.info(
        "Saved %d total weather rows (%d new) → %s",
        len(combined), len(fresh), cache_path,
    )
    return len(fresh)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def _cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch real ERA5 daily weather (Open-Meteo Archive) for "
                    "every active FIRMS grid cell."
    )
    p.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    p.add_argument("--end",   help="Override end date (YYYY-MM-DD)")
    p.add_argument("--limit-cells", type=int, default=None,
                   help="Fetch only the first N active cells (debugging)")
    p.add_argument("--sleep", type=float, default=0.15,
                   help="Seconds to sleep between API calls")
    return p.parse_args()


def main() -> None:
    args = _cli()
    cells, dmin, dmax = discover_active_cells()

    start = datetime.strptime(args.start, "%Y-%m-%d").date() if args.start else dmin
    end   = datetime.strptime(args.end,   "%Y-%m-%d").date() if args.end   else dmax

    update_cache(
        cells=cells,
        start=start,
        end=end,
        sleep_between_calls=args.sleep,
        limit_cells=args.limit_cells,
    )


if __name__ == "__main__":
    main()
