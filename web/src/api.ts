import type { FireGeoJson } from "./types";

// In dev the Vite proxy forwards /geojson → http://localhost:8000.
// In prod FastAPI serves the SPA from the same origin, so /geojson is
// already on the right host. VITE_API_BASE lets a deployer override
// (e.g. point the static SPA at a separate API host) without rebuilding.
const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/$/, "");

export async function fetchGeoJson(): Promise<FireGeoJson> {
  const res = await fetch(`${API_BASE}/geojson`);
  if (!res.ok) throw new Error(`GeoJSON fetch failed: HTTP ${res.status}`);
  return res.json();
}
