import L from "leaflet";
import {
  APPROX_METERS_PER_DEGREE,
  DOT_CLAMP_FRAC_REF,
  DOT_CLAMP_MAX_PX_REF,
  DOT_CLAMP_MIN_PX_REF,
} from "../constants";
import type { FireFeature } from "../types";

// Detect grid resolution from the actual prediction features so dot
// sizing self-adjusts when GRID_SIZE changes upstream.
export function detectGridSizeDegrees(features: FireFeature[]): number {
  if (!features || features.length < 2) return 0.1;
  const lats = [...new Set(features.map((f) => f.geometry.coordinates[1]))]
    .sort((a, b) => a - b);
  let minDiff = Infinity;
  for (let i = 1; i < lats.length; i++) {
    const d = lats[i] - lats[i - 1];
    if (d > 1e-6 && d < minDiff) minDiff = d;
  }
  if (!isFinite(minDiff) || minDiff < 0.005 || minDiff > 0.5) return 0.1;
  return minDiff;
}

export function gridSizeMeters(predictedFeatures: FireFeature[]): number {
  return detectGridSizeDegrees(predictedFeatures) * APPROX_METERS_PER_DEGREE;
}

export function dotRadiusMeters(fraction: number, gridMeters: number): number {
  return (gridMeters / 2) * fraction;
}

export function metersPerPixel(map: L.Map, lat: number): number {
  return (
    (156543.03392 * Math.cos((lat * Math.PI) / 180)) /
    Math.pow(2, map.getZoom())
  );
}

export function clampPxForFrac(frac: number): { min: number; max: number } {
  const scale = frac / DOT_CLAMP_FRAC_REF;
  return {
    min: DOT_CLAMP_MIN_PX_REF * scale,
    max: DOT_CLAMP_MAX_PX_REF * scale,
  };
}

// Convert a meters-radius into one that, at the current zoom and latitude,
// falls inside [minPx, maxPx]. Mid-zoom levels keep the natural meters-based
// behaviour; only the extremes are clamped.
export function clampedRadiusMeters(
  map: L.Map,
  lat: number,
  baseM: number,
  minPx: number,
  maxPx: number
): number {
  const mPerPx = metersPerPixel(map, lat);
  let px = baseM / mPerPx;
  if (px > maxPx) px = maxPx;
  if (px < minPx) px = minPx;
  return px * mPerPx;
}

// We monkey-patch the L.Circle instance with our anchor metadata so the
// global zoomend handler can re-clamp every dot in one pass.
export type AnchoredCircle = L.Circle & {
  _baseRadiusM: number;
  _minPx: number;
  _maxPx: number;
  _anchorLat: number;
};
