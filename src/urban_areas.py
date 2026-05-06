"""Curated Thai urban areas, used to filter wildfire predictions out of cities.

WHY THIS EXISTS
---------------
FIRMS satellite hotspot detections fire in cities for reasons that aren't
wildfires: open garbage burning (very common in Thailand), industrial heat
sources, agricultural waste burning, vehicle/structure fires, heated rooftops.
The model trained on the densified FIRMS frame learns to predict fire activity
at urban locations because it really did happen there — but those are not
wildfires, and predicting them on the dashboard is misleading.

This module provides two things:
  1. A curated list of major Thai urban areas with hand-tuned urban radii.
  2. A vectorized "is this point inside any urban area?" function used by
     risk_map.py to filter predictions before display.

THE LIST
--------
Top 30+ Thai cities by population (Department of Local Administration figures
plus metropolitan-area estimates for the Bangkok cluster). Coordinates are
city centers from OpenStreetMap. Radii are hand-tuned to roughly match the
visible urban footprint on satellite imagery — a cell whose center falls
inside this radius is considered "in the city" and is dropped from the
prediction map.

To extend or adjust:
  • Edit the THAI_URBAN_AREAS list below.
  • Bump a city's `radius_km` if the urban sprawl extends further than the
    hand-tuned default.
  • Add tourist resorts, industrial estates, or military bases if you find
    they show as fire hotspots in your dashboard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import numpy as np


@dataclass(frozen=True)
class UrbanArea:
    """A circular urban exclusion zone."""
    name: str
    lat: float
    lon: float
    radius_km: float


# =============================================================================
# Curated list of Thai urban areas
#
# Sources: Wikipedia (List of municipalities in Thailand, Bangkok Metropolitan
# Region), Department of Local Administration 2024, OpenStreetMap. Radii are
# hand-tuned to roughly match the urban footprint on satellite imagery.
# =============================================================================
THAI_URBAN_AREAS: Tuple[UrbanArea, ...] = (
    # === Bangkok Metropolitan Region (BMR) — ~15M people in a tight cluster ===
    # Bangkok proper has 10M+ people; the radius covers most of the BMR
    # contiguous urban sprawl west to Nonthaburi and south to Samut Prakan.
    UrbanArea("Bangkok",                  13.7563, 100.5018, 28.0),
    UrbanArea("Nonthaburi",               13.8622, 100.5145,  6.0),
    UrbanArea("Pak Kret",                 13.9132, 100.4986,  4.5),
    UrbanArea("Samut Prakan",             13.5990, 100.5998,  6.5),
    UrbanArea("Pathum Thani",             14.0207, 100.5247,  4.5),
    UrbanArea("Rangsit",                  13.9788, 100.6149,  4.5),
    UrbanArea("Phra Pradaeng",            13.6592, 100.5331,  4.0),
    UrbanArea("Nakhon Pathom",            13.8196, 100.0626,  4.0),
    UrbanArea("Samut Sakhon",             13.5475, 100.2746,  4.5),

    # === Eastern seaboard (industrial + tourism) ===
    # Heavy garbage and industrial burning — important to exclude.
    UrbanArea("Chonburi",                 13.3611, 100.9847,  5.5),
    UrbanArea("Si Racha",                 13.1746, 100.9305,  4.5),
    UrbanArea("Pattaya",                  12.9236, 100.8825,  7.0),
    UrbanArea("Rayong",                   12.6814, 101.2789,  4.5),

    # === Northern Thailand (in burning-season hotspot region) ===
    # These cities sit in the middle of heavily-burned agricultural valleys —
    # excluding them is especially important to avoid masking real fires.
    UrbanArea("Chiang Mai",               18.7883,  98.9853, 10.0),
    UrbanArea("Lamphun",                  18.5744,  99.0086,  3.5),
    UrbanArea("Chiang Rai",               19.9105,  99.8406,  4.5),
    UrbanArea("Lampang",                  18.2855,  99.5117,  4.5),
    UrbanArea("Phitsanulok",              16.8211, 100.2659,  4.5),
    UrbanArea("Nakhon Sawan",             15.7045, 100.1374,  4.0),
    UrbanArea("Mae Sot",                  16.7167,  98.5667,  3.0),
    UrbanArea("Tak",                      16.8847,  99.1258,  3.0),

    # === Northeast Thailand (Isan) ===
    UrbanArea("Nakhon Ratchasima",        14.9799, 102.0978,  6.0),
    UrbanArea("Khon Kaen",                16.4419, 102.8359,  5.5),
    UrbanArea("Udon Thani",               17.4151, 102.7878,  5.5),
    UrbanArea("Ubon Ratchathani",         15.2287, 104.8590,  5.0),
    UrbanArea("Roi Et",                   16.0540, 103.6510,  3.5),
    UrbanArea("Nong Khai",                17.8782, 102.7414,  3.5),

    # === Central / West ===
    UrbanArea("Ayutthaya",                14.3692, 100.5876,  4.0),
    UrbanArea("Lopburi",                  14.7995, 100.6533,  3.5),
    UrbanArea("Saraburi",                 14.5289, 100.9106,  3.5),
    UrbanArea("Kanchanaburi",             14.0228,  99.5328,  3.5),
    UrbanArea("Ratchaburi",               13.5283,  99.8134,  3.5),

    # === South ===
    UrbanArea("Hat Yai",                   7.0086, 100.4747,  6.0),
    UrbanArea("Songkhla",                  7.1996, 100.5950,  4.5),
    UrbanArea("Surat Thani",               9.1383,  99.3215,  4.5),
    UrbanArea("Phuket",                    7.8804,  98.3923,  6.5),
    UrbanArea("Nakhon Si Thammarat",       8.4304,  99.9633,  4.0),
    UrbanArea("Trang",                     7.5645,  99.6238,  3.5),
    UrbanArea("Yala",                      6.5413, 101.2803,  4.0),
    UrbanArea("Krabi",                     8.0863,  98.9063,  3.5),
)


# =============================================================================
# Vectorized distance computation
# =============================================================================

EARTH_RADIUS_KM = 6371.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two (lat, lon) points."""
    rad = math.pi / 180.0
    dlat = (lat2 - lat1) * rad
    dlon = (lon2 - lon1) * rad
    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1 * rad) * math.cos(lat2 * rad) * math.sin(dlon / 2) ** 2)
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def classify_urban(
    lats: Iterable[float],
    lons: Iterable[float],
    urban_areas: Iterable[UrbanArea] = THAI_URBAN_AREAS,
    buffer_km: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorized urban-area lookup for a batch of grid cells.

    For each input (lat, lon) pair we compute:
      • is_urban: True if the cell falls inside ANY urban area's
        (radius_km + buffer_km) zone.
      • nearest_dist_km: great-circle distance to the closest urban
        center. Use ``is_urban`` to determine whether the point is inside
        the urban radius.
      • nearest_name: name of the closest urban area.

    The implementation projects all points and all urban centers into radians,
    then loops over urban areas (typically ~40 entries) doing a fully-vectorized
    haversine pass per area. For 100k cells × 40 cities this runs in well
    under a second on modest hardware.

    Args:
        lats, lons: Equal-length sequences of grid-cell coordinates.
        urban_areas: Iterable of UrbanArea — defaults to the curated Thai list.
        buffer_km: Additional km added to every urban radius. Use this to
            exclude cells *near* (but not strictly inside) cities. Set 0 to
            use only the hand-tuned radii.

    Returns:
        Tuple of three numpy arrays: (is_urban, nearest_dist_km, nearest_name).
    """
    lats_arr = np.asarray(lats, dtype=float)
    lons_arr = np.asarray(lons, dtype=float)
    if lats_arr.shape != lons_arr.shape:
        raise ValueError("lats and lons must have the same length")
    n_pts = lats_arr.size

    rad = math.pi / 180.0
    lats_r = lats_arr * rad
    lons_r = lons_arr * rad
    cos_lats = np.cos(lats_r)

    is_urban = np.zeros(n_pts, dtype=bool)
    min_dist = np.full(n_pts, np.inf)
    nearest_name = np.array([""] * n_pts, dtype=object)

    for u in urban_areas:
        u_lat_r = u.lat * rad
        u_lon_r = u.lon * rad
        dlat = u_lat_r - lats_r
        dlon = u_lon_r - lons_r
        a = (np.sin(dlat / 2) ** 2
             + cos_lats * math.cos(u_lat_r) * np.sin(dlon / 2) ** 2)
        d = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))

        # Mark cells inside this city's (radius + buffer) zone
        inside_this = d <= (u.radius_km + buffer_km)
        is_urban |= inside_this

        # Track the nearest city per cell
        is_closer = d < min_dist
        min_dist = np.where(is_closer, d, min_dist)
        if is_closer.any():
            # Object-dtype arrays don't broadcast nicely with np.where for
            # strings, so update by mask.
            nearest_name[is_closer] = u.name

    return is_urban, min_dist, nearest_name