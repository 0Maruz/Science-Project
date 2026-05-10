import type { UrgencyLevel } from "./types";

export const URGENCY_COLORS: Record<UrgencyLevel, string> = {
  CRITICAL: "#dc2626",
  HIGH: "#ea580c",
  MEDIUM: "#f59e0b",
  LOW: "#10b981",
  NONE: "#6b7280",
};

// Per-tier dot fractions of cell-width — chosen so even adjacent
// CRITICAL+CRITICAL dots leave ~20% of the cell width as edge gap.
export const URGENCY_DOT_FRAC: Record<UrgencyLevel, number> = {
  CRITICAL: 0.4,
  HIGH: 0.32,
  MEDIUM: 0.25,
  LOW: 0.18,
  NONE: 0.14,
};

export const OBSERVED_DOT_FRAC = 0.3;
export const APPROX_METERS_PER_DEGREE = 111320;

// Pixel-size clamps applied across zoom levels — without these a 2 km
// CRITICAL dot becomes ~900 px wide at zoom 14 and ~1 px at zoom 6.
export const DOT_CLAMP_MIN_PX_REF = 3;
export const DOT_CLAMP_MAX_PX_REF = 28;
export const DOT_CLAMP_FRAC_REF = 0.4;
