# =========================================================
# FIRE DATE MAP GENERATION (OBSERVED + PREDICTED DATES)
# =========================================================
#
# All values written to fire_dates_all.geojson are derived from REAL data:
#   • observed:   NASA FIRMS VIIRS NRT detections (latest densified day)
#   • predicted:  model output on real features for the latest base date
#   • historical_fire_count_30d: literal sum of FIRMS detections in this
#                                grid cell over the 30 days ending at base_date
#   • urgency_level: derived via thresholds calibrated on real validation
#                    predictions (dataset_info.json["urgency_thresholds"])
# =========================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
from datetime import timedelta
from dotenv import load_dotenv

from features import (
    FEATURES_CORE,
    FEATURES_WEATHER,
    MAX_PREDICTION_DAYS,
    DEFAULT_URGENCY_THRESHOLDS,
    urgency_from_thresholds,
)

# =========================================================
# CONFIG
# =========================================================

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "lgbm_fire_date_model.pkl")
DATA_PATH  = os.path.join(BASE_DIR, "outputs", "features", "full_features.csv")
META_PATH  = os.path.join(BASE_DIR, "outputs", "metadata", "dataset_info.json")

RISKMAP_DIR  = os.path.join(BASE_DIR, "outputs", "riskmap")
GEOJSON_PATH = os.path.join(RISKMAP_DIR, "fire_dates_all.geojson")
LATEST_PATH  = os.path.join(RISKMAP_DIR, "latest.json")

os.makedirs(RISKMAP_DIR, exist_ok=True)

HISTORY_WINDOW_DAYS = 30


# =========================================================
# METADATA HELPERS
# =========================================================

def _load_metadata() -> dict:
    if not os.path.exists(META_PATH):
        return {}
    try:
        with open(META_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _resolve_features(meta: dict, df: pd.DataFrame) -> list:
    """Prefer the persisted feature list (matches the deployed model exactly)."""
    feats = meta.get("features")
    if feats:
        return list(feats)
    # Fall back: any core/weather column that's actually present.
    return [c for c in (*FEATURES_CORE, *FEATURES_WEATHER) if c in df.columns]


def _resolve_thresholds(meta: dict) -> dict:
    t = meta.get("urgency_thresholds")
    if isinstance(t, dict) and {"CRITICAL", "HIGH", "MEDIUM", "LOW"} <= set(t):
        return {k: float(v) for k, v in t.items()}
    return dict(DEFAULT_URGENCY_THRESHOLDS)


# =========================================================
# LOAD MODEL & DATA
# =========================================================

def load_assets():
    model = joblib.load(MODEL_PATH)

    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.date

    return model, df


# =========================================================
# OBSERVED LAYER (latest real day from FIRMS)
# =========================================================

def build_observed(df: pd.DataFrame):
    observed_date = df["date"].max()
    obs = df[df["date"] == observed_date].copy()
    return obs, observed_date


# =========================================================
# HISTORICAL FIRE COUNT (real FIRMS detections in last N days)
# =========================================================

def historical_fire_counts(df: pd.DataFrame, base_date, window_days: int = HISTORY_WINDOW_DAYS) -> pd.DataFrame:
    """Real per-cell sum of fire_count over the [base_date-window, base_date] window.

    No imputation, no extrapolation — just a groupby on the densified FIRMS
    frame. Cells with no detections in the window get exactly 0.
    """
    start = base_date - timedelta(days=window_days)
    window = df[(df["date"] >= start) & (df["date"] <= base_date)]
    counts = (
        window.groupby(["lat_grid", "lon_grid"], as_index=False)["fire_count"]
        .sum()
        .rename(columns={"fire_count": "historical_fire_count_30d"})
    )
    counts["historical_fire_count_30d"] = counts["historical_fire_count_30d"].astype(int)
    return counts


# =========================================================
# PREDICTED LAYER
# =========================================================

def build_predicted(df: pd.DataFrame, model, base_date, meta: dict):
    base = df[df["date"] == base_date].copy()

    feature_cols = _resolve_features(meta, df)
    missing = [c for c in feature_cols if c not in base.columns]
    if missing:
        raise RuntimeError(
            f"Feature CSV is missing {len(missing)} columns expected by the model: "
            f"{missing[:5]}{'…' if len(missing) > 5 else ''}. Re-run train.py."
        )
    X = base[feature_cols].fillna(0)

    raw_pred = model.predict(X)
    days_clipped = np.clip(np.round(raw_pred), 0, MAX_PREDICTION_DAYS).astype(int)

    base["raw_prediction"] = raw_pred
    base["days_until_fire"] = days_clipped
    base["predicted_fire_date"] = [
        (base_date + timedelta(days=int(d))).strftime("%Y-%m-%d")
        for d in days_clipped
    ]

    # Rounding-proximity proxy. Documented as NOT a calibrated probability.
    base["prediction_confidence"] = 1.0 - np.abs(raw_pred - days_clipped)

    thresholds = _resolve_thresholds(meta)
    base["urgency_level"] = [
        urgency_from_thresholds(int(d), thresholds) for d in days_clipped
    ]

    # Attach real historical fire count per cell (last 30 days from FIRMS)
    counts = historical_fire_counts(df, base_date)
    base = base.merge(counts, on=["lat_grid", "lon_grid"], how="left")
    base["historical_fire_count_30d"] = (
        base["historical_fire_count_30d"].fillna(0).astype(int)
    )

    return base, base_date, thresholds


# =========================================================
# WRITE GEOJSON (append-then-overwrite-current-base-date)
# =========================================================

def append_geojson(observed: pd.DataFrame, predicted: pd.DataFrame, base_date, thresholds: dict, metrics: dict):
    base_date_str = base_date.strftime("%Y-%m-%d")

    geojson = {"type": "FeatureCollection", "features": []}
    if os.path.exists(GEOJSON_PATH):
        try:
            with open(GEOJSON_PATH, "r", encoding="utf-8") as f:
                geojson = json.load(f)
        except json.JSONDecodeError:
            print("⚠️ Corrupted GeoJSON → recreate")

    # Drop only the predictions for the current base_date (preserve observed
    # rows and predictions for prior base dates).
    geojson["features"] = [
        f for f in geojson["features"]
        if f["properties"].get("base_date") != base_date_str
        or f["properties"].get("source") == "observed"
    ]

    # Top-level metadata so the frontend can read calibrated thresholds and
    # validation metrics without a separate fetch.
    geojson["metadata"] = {
        "base_date": base_date_str,
        "horizon_days": MAX_PREDICTION_DAYS,
        "urgency_thresholds": thresholds,
        "metrics": metrics,
        "history_window_days": HISTORY_WINDOW_DAYS,
    }

    # ---------- OBSERVED (latest densified FIRMS day) ----------
    observed_date_str = observed["date"].iloc[0].strftime("%Y-%m-%d") if len(observed) else base_date_str
    for _, r in observed.iterrows():
        if int(r.get("fire_count", 0)) <= 0:
            # Densified rows include cells with zero fires; only emit real detections.
            continue
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])],
            },
            "properties": {
                "date": observed_date_str,
                "source": "observed",
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"]),
                "fire_count": int(r["fire_count"]),
            },
        })

    # ---------- PREDICTED ----------
    for _, r in predicted.iterrows():
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lon_grid"]), float(r["lat_grid"])],
            },
            "properties": {
                "base_date": base_date_str,
                "source": "predicted",
                "days_until_fire": int(r["days_until_fire"]),
                "predicted_fire_date": str(r["predicted_fire_date"]),
                "urgency_level": str(r["urgency_level"]),
                "confidence": float(r["prediction_confidence"]),
                "raw_prediction": float(r["raw_prediction"]),
                "historical_fire_count_30d": int(r["historical_fire_count_30d"]),
                "lat": float(r["lat_grid"]),
                "lon": float(r["lon_grid"]),
            },
        })

    with open(GEOJSON_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

    with open(LATEST_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_date": base_date_str,
                "observed_date": observed_date_str,
                "prediction_horizon_days": MAX_PREDICTION_DAYS,
                "urgency_thresholds": thresholds,
                "metrics": metrics,
            },
            f,
            indent=2,
        )


# =========================================================
# PIPELINE
# =========================================================

def run():
    print("🔄 Loading assets...")
    model, df = load_assets()
    meta = _load_metadata()

    print("📍 Building observed layer (real FIRMS detections)...")
    observed, obs_date = build_observed(df)

    print(f"🔮 Predicting fire dates for next {MAX_PREDICTION_DAYS} days...")
    predicted, base_date, thresholds = build_predicted(df, model, obs_date, meta)

    metrics = (meta.get("model") or {}).get("test_metrics", {}) or {}
    append_geojson(observed, predicted, base_date, thresholds, metrics)

    print("\n✅ FIRE DATE MAP UPDATED")
    print("Observed date :", obs_date)
    print("Base date     :", base_date)
    print("Thresholds    :", thresholds)
    print("GeoJSON       :", GEOJSON_PATH)

    urgency_counts = predicted["urgency_level"].value_counts()
    print("\n📊 URGENCY SUMMARY (calibrated thresholds):")
    for level in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "NONE"]:
        print(f"  {level}: {int(urgency_counts.get(level, 0))} locations")


# =========================================================
# ENTRY
# =========================================================

if __name__ == "__main__":
    run()
