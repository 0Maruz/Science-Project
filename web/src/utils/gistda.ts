import type { GistdaFeature } from "../types";

// GISTDA NRT VIIRS hotspot endpoints — public, no auth required.
// Updated ~12-hourly by GISTDA from the same Suomi-NPP satellite as FIRMS.
const GISTDA_VIIRS_URL =
  "https://gistdaportal.gistda.or.th/data/rest/services/FR_Fire/hotspot_npp_daily/MapServer/0/query";
const GISTDA_MODIS_URL =
  "https://gistdaportal.gistda.or.th/data/rest/services/FR_Fire/hotspot_daily/MapServer/0/query";

const THAILAND_BBOX = "96,4,107,22"; // matches FIRMS_BBOX in env.example

async function queryLayer(url: string, signal?: AbortSignal): Promise<GistdaFeature[]> {
  const params = new URLSearchParams({
    where: "1=1",
    geometry: THAILAND_BBOX,
    geometryType: "esriGeometryEnvelope",
    spatialRel: "esriSpatialRelIntersects",
    outFields: "latitude,longitude,confident,lu_name,pv_tn,ap_tn,date,time,satellite",
    f: "json",
  });
  const res = await fetch(`${url}?${params}`, {
    signal: signal ?? AbortSignal.timeout(20000),
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const payload = await res.json();
  if (payload.error) throw new Error(payload.error.message || "ArcGIS error");
  return (payload.features ?? []) as GistdaFeature[];
}

export async function fetchLiveFires(signal?: AbortSignal): Promise<GistdaFeature[]> {
  // VIIRS NPP is primary; MODIS is a secondary independent source. Settle
  // both so one endpoint failing doesn't black out the map.
  const [viirs, modis] = await Promise.allSettled([
    queryLayer(GISTDA_VIIRS_URL, signal),
    queryLayer(GISTDA_MODIS_URL, signal),
  ]);
  const viirsFeats = viirs.status === "fulfilled" ? viirs.value : [];
  const modisFeats = modis.status === "fulfilled" ? modis.value : [];
  if (viirs.status === "rejected" && modis.status === "rejected") {
    throw viirs.reason instanceof Error ? viirs.reason : new Error(String(viirs.reason));
  }
  return [...viirsFeats, ...modisFeats];
}

export const LIVE_FIRE_COLOR = "#06b6d4"; // cyan — distinct from urgency + observed
export const LIVE_REFRESH_MS = 30 * 60 * 1000; // 30 min auto-refresh while toggle is on
