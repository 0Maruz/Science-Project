import L from "leaflet";
import "leaflet.markercluster";
import { useEffect, useMemo, useRef } from "react";
import {
  OBSERVED_DOT_FRAC,
  URGENCY_COLORS,
  URGENCY_DOT_FRAC,
} from "../constants";
import type {
  DisplayOptions,
  FireFeature,
  UrgencyLevel,
  UrgencyThresholds,
} from "../types";
import { createHeatmapLayer } from "../utils/heatmap";
import {
  AnchoredCircle,
  clampPxForFrac,
  clampedRadiusMeters,
  dotRadiusMeters,
  gridSizeMeters,
} from "../utils/markers";

interface MapViewProps {
  observed: FireFeature[];
  predictedAll: FireFeature[]; // all predicted features for grid-size detection
  predictedVisible: FireFeature[]; // current day-filter applied
  thresholds: UrgencyThresholds | null;
  options: DisplayOptions;
}

// Render order: heatmap below, then observed and predicted dots on top so
// popups stay clickable.
const LAYER_ORDER = ["heatmap", "observed", "predicted"] as const;
type LayerKey = (typeof LAYER_ORDER)[number];

export default function MapView(props: MapViewProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<L.Map | null>(null);
  const layersRef = useRef<Record<LayerKey, L.Layer | null>>({
    heatmap: null,
    observed: null,
    predicted: null,
  });

  // One-time map init.
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, { center: [15.0, 101.0], zoom: 6 });
    L.tileLayer("https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png", {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    map.on("zoomend", () => reclampAllMarkers(layersRef.current, map));
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  // Stable grid-size for sizing dots (detected from the full set, not just visible).
  const gridM = useMemo(() => gridSizeMeters(props.predictedAll), [props.predictedAll]);

  // Rebuild layers whenever inputs change.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    clearLayers(layersRef.current, map);

    if (props.options.showObserved) {
      layersRef.current.observed = buildObservedLayer(map, props.observed, props.options.clusterMarkers);
    }

    if (props.options.showPredicted) {
      layersRef.current.heatmap = createHeatmapLayer(
        props.predictedVisible,
        props.thresholds,
        props.options.heatRadius
      );
      if (props.options.showCellPins) {
        layersRef.current.predicted = buildPredictedLayer(map, props.predictedVisible, gridM);
      }
    }

    addLayers(layersRef.current, map);
  }, [
    props.observed,
    props.predictedVisible,
    props.thresholds,
    props.options.showObserved,
    props.options.showPredicted,
    props.options.showCellPins,
    props.options.clusterMarkers,
    props.options.heatRadius,
    gridM,
  ]);

  return <div id="map" ref={containerRef} />;
}

// ───────────────────── helpers ─────────────────────

function clearLayers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of LAYER_ORDER) {
    if (layers[k]) {
      map.removeLayer(layers[k]!);
      layers[k] = null;
    }
  }
}

function addLayers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of LAYER_ORDER) if (layers[k]) map.addLayer(layers[k]!);
}

function reclampAllMarkers(layers: Record<LayerKey, L.Layer | null>, map: L.Map) {
  for (const k of ["observed", "predicted"] as const) {
    const layer = layers[k];
    if (!layer) continue;
    const eachLayer = (layer as unknown as { eachLayer?: (cb: (m: L.Layer) => void) => void })
      .eachLayer;
    if (!eachLayer) continue;
    eachLayer.call(layer, (m) => {
      const marker = m as AnchoredCircle;
      if (marker._baseRadiusM == null || typeof marker.setRadius !== "function") return;
      marker.setRadius(
        clampedRadiusMeters(map, marker._anchorLat, marker._baseRadiusM, marker._minPx, marker._maxPx)
      );
    });
  }
}

function buildObservedLayer(map: L.Map, features: FireFeature[], cluster: boolean): L.Layer {
  const layer = cluster
    ? (L as unknown as { markerClusterGroup: () => L.LayerGroup }).markerClusterGroup()
    : L.layerGroup();

  const renderer = L.canvas({ padding: 0.5 });
  const gridM = gridSizeMeters(features);
  const baseM = dotRadiusMeters(OBSERVED_DOT_FRAC, gridM);
  const px = clampPxForFrac(OBSERVED_DOT_FRAC);

  for (const f of features) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);
    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: "#ff5722",
      color: "#fff",
      weight: 1,
      fillOpacity: 0.8,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;
    marker.bindPopup(
      `<div class="popup">
         <b style="color:#ff5722;">🔥 Observed Fire (FIRMS)</b><br>
         <small>Date: ${p.date ?? "—"}</small><br>
         <small>FIRMS detections: ${p.fire_count ?? "—"}</small><br>
         <small>Location: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>
       </div>`
    );
    (layer as unknown as { addLayer: (m: L.Layer) => void }).addLayer(marker);
  }
  return layer as unknown as L.Layer;
}

function buildPredictedLayer(map: L.Map, features: FireFeature[], gridM: number): L.Layer {
  const layer = L.layerGroup();
  const renderer = L.canvas({ padding: 0.5 });

  for (const f of features) {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;
    const urgency = (p.urgency_level ?? "NONE") as UrgencyLevel;
    const color = URGENCY_COLORS[urgency] ?? URGENCY_COLORS.NONE;
    const frac = URGENCY_DOT_FRAC[urgency] ?? URGENCY_DOT_FRAC.NONE;
    const baseM = dotRadiusMeters(frac, gridM);
    const px = clampPxForFrac(frac);
    const radiusM = clampedRadiusMeters(map, lat, baseM, px.min, px.max);

    const marker = L.circle([lat, lon], {
      radius: radiusM,
      renderer,
      fillColor: color,
      color: "#fff",
      weight: 1,
      fillOpacity: 0.85,
    }) as AnchoredCircle;
    marker._baseRadiusM = baseM;
    marker._minPx = px.min;
    marker._maxPx = px.max;
    marker._anchorLat = lat;

    const fireDate = p.predicted_fire_date ?? "—";
    const daysUntil = p.days_until_fire ?? 0;
    const confidence = p.confidence != null ? (p.confidence * 100).toFixed(0) : "—";
    const rawPred = p.raw_prediction != null ? Number(p.raw_prediction).toFixed(2) : null;
    const histCount = p.historical_fire_count_30d;

    let html =
      `<div class="popup" style="min-width:220px;">
         <b style="color:${color};">🔮 Fire Prediction</b><br>
         <div class="popup-block">
           <b>Predicted: ${fireDate}</b><br>
           <small>In ${daysUntil} day${daysUntil !== 1 ? "s" : ""}` +
      (rawPred ? ` (raw=${rawPred})` : "") +
      `</small>
         </div>
         <small>Urgency: <b>${p.urgency_level ?? "—"}</b> (calibrated)</small><br>
         <small>Confidence (rounding proxy): ${confidence}%</small><br>`;
    if (histCount != null)
      html += `<small>Historical fires (30d, FIRMS): <b>${histCount}</b></small><br>`;
    if (p.fire_days_per_year != null)
      html += `<small>Historical rate: <b>${Number(p.fire_days_per_year).toFixed(1)}</b> fire-days/year</small><br>`;
    if (p.nearest_urban_area && p.nearest_urban_distance_km != null) {
      const km = Number(p.nearest_urban_distance_km).toFixed(1);
      html += `<small>Nearest city: ${p.nearest_urban_area} (${km} km)</small><br>`;
    }
    html += `<small>Cell: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small></div>`;
    marker.bindPopup(html);
    layer.addLayer(marker);
  }
  return layer;
}
