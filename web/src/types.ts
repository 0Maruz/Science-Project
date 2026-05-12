// Mirrors what risk_map.append_geojson writes. The backend is the authority;
// these types only document what the frontend reads.

export type UrgencyLevel = "CRITICAL" | "HIGH" | "MEDIUM" | "LOW" | "NONE";

export interface UrgencyThresholds {
  CRITICAL: number;
  HIGH: number;
  MEDIUM: number;
  LOW: number;
}

export interface ValidationMetrics {
  mae_days?: number;
  rmse_days?: number;
  r2?: number;
  accuracy_within_1day?: number;
}

export interface SnapshotHitRate {
  hits?: number;
  misses?: number;
  future?: number;
}

export interface GeoJsonMetadata {
  urgency_thresholds?: UrgencyThresholds;
  metrics?: ValidationMetrics;
  validation_summary?: {
    per_snapshot?: Record<string, SnapshotHitRate>;
  };
}

export interface PredictionProperties {
  source: "predicted" | "observed";
  base_date?: string;
  predicted_fire_date?: string;
  days_until_fire?: number;
  raw_prediction?: number;
  urgency_level?: UrgencyLevel;
  confidence?: number;
  province?: string;
  historical_fire_count_30d?: number;
  fire_days_per_year?: number;
  tree_cover_pct_2000?: number;
  tree_loss_pct_recent?: number;
  nearest_urban_area?: string;
  nearest_urban_distance_km?: number;
  // observed
  date?: string;
  fire_count?: number;
}

export interface FireFeature {
  type: "Feature";
  geometry: { type: "Point"; coordinates: [number, number] };
  properties: PredictionProperties;
}

export interface FireGeoJson {
  type: "FeatureCollection";
  features: FireFeature[];
  metadata?: GeoJsonMetadata;
}

export type DaySelection = "all" | "0" | "1" | "2" | "3" | "4" | "5" | "6" | "7";

export interface DisplayOptions {
  showObserved: boolean;
  showLiveFires: boolean;
  // Internal "always-on" flags, kept in the type so MapView's prop contract
  // doesn't have to branch. The Sidebar no longer exposes user toggles for
  // these — predictions + cell pins are part of the core dashboard now.
  showPredicted: boolean;
  showCellPins: boolean;
  heatRadius: number;
}

export type LiveFireStatus = "idle" | "loading" | "ok" | "error";

export interface LiveFireMeta {
  status: LiveFireStatus;
  count: number;
  lastFetch: Date | null;
  error: string | null;
}

// GISTDA ArcGIS feature attributes — typed loosely because the upstream
// schema can vary between the NPP and MODIS endpoints (some fields like
// `satellite` only show up on one).
export interface GistdaFeature {
  attributes: {
    latitude?: number | string;
    longitude?: number | string;
    confident?: string | number;
    lu_name?: string;
    pv_tn?: string;
    ap_tn?: string;
    date?: number;
    time?: string;
    satellite?: string;
  };
}
