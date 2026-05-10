import type { FireFeature, PredictionProperties } from "../types";

const COLS: (keyof PredictionProperties)[] = [
  "base_date",
  "predicted_fire_date",
  "days_until_fire",
  "raw_prediction",
  "urgency_level",
  "province",
  "historical_fire_count_30d",
  "fire_days_per_year",
  "tree_cover_pct_2000",
  "tree_loss_pct_recent",
  "nearest_urban_area",
  "nearest_urban_distance_km",
];

const escape = (v: unknown): string => {
  if (v == null) return "";
  const s = String(v);
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function exportCellsCsv(
  cells: FireFeature[],
  baseDate: string,
  province: string,
  day: string
): boolean {
  if (!cells.length) return false;

  // Final 14-col schema mirrors the original frontend: lat/lon first, then
  // the documented prediction fields. Kept stable so downstream consumers
  // don't break when we add new GeoJSON properties.
  const headers = ["lat", "lon", ...COLS];
  const rows = [headers.join(",")];
  for (const f of cells) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const row = [escape(lat), escape(lon), ...COLS.map((c) => escape(p[c]))];
    rows.push(row.join(","));
  }

  const provinceSlug = province === "all" ? "all" : province.replace(/\s+/g, "_");
  const daySlug = day === "all" ? "all" : `day${day}`;
  const filename = `fire_predictions_${baseDate}_${provinceSlug}_${daySlug}.csv`;

  const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
  return true;
}
