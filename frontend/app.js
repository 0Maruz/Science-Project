// ===================================
// FIRE DATE PREDICTION DASHBOARD
//
// All values shown to the user come from REAL sources:
//   • observed:  NASA FIRMS detections at the latest base date
//   • predicted: model output (real features) for the next 7 days
//   • thresholds: calibrated from real validation predictions
//   • metrics:   real held-out test metrics from train.py
//   • historical fire count: literal sum of FIRMS detections per cell
// No synthetic / faked / interpolated data anywhere in the UI.
// ===================================

const map = L.map("map", { center: [13.5, 101], zoom: 6 });

L.tileLayer(
  "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png",
  { attribution: "&copy; OpenStreetMap contributors" }
).addTo(map);

// ===================================
// State
// ===================================
const state = {
  geojson: null,
  selectedDay: "all", // "all" | 0..7
  thresholds: null,   // { CRITICAL, HIGH, MEDIUM, LOW }
  metrics: null,      // held-out test metrics
  layers: { observed: null, predicted: null },
};

const URGENCY_COLORS = {
  CRITICAL: "#dc2626",
  HIGH: "#ea580c",
  MEDIUM: "#f59e0b",
  LOW: "#10b981",
  NONE: "#6b7280",
};
const URGENCY_SIZES = { CRITICAL: 9, HIGH: 8, MEDIUM: 7, LOW: 6, NONE: 5 };

// ===================================
// Init
// ===================================
async function init() {
  try {
    const response = await fetch("../outputs/riskmap/fire_dates_all.geojson");
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    state.geojson = await response.json();
  } catch (err) {
    console.error("Failed to load GeoJSON:", err);
    alert("Failed to load fire prediction data. Run train.py + risk_map.py first.");
    return;
  }

  // GeoJSON top-level metadata is written by risk_map.append_geojson.
  const meta = state.geojson.metadata || {};
  state.thresholds = meta.urgency_thresholds || null;
  state.metrics = meta.metrics || null;

  renderThresholds();
  renderMetrics();
  bindEvents();
  displayData();
}

// ===================================
// Render calibrated threshold ranges in urgency cards
// ===================================
function renderThresholds() {
  const t = state.thresholds;
  if (!t) {
    document.getElementById("thresholdNote").textContent =
      "No calibrated thresholds in metadata — falling back to legacy 0/2/4/7 cutoffs.";
    document.getElementById("criticalRange").textContent = "≤ 0 d";
    document.getElementById("highRange").textContent = "≤ 2 d";
    document.getElementById("mediumRange").textContent = "≤ 4 d";
    document.getElementById("lowRange").textContent = "≤ 7 d";
    return;
  }
  const fmt = (v) => `≤ ${Number(v).toFixed(1)} d`;
  document.getElementById("criticalRange").textContent = fmt(t.CRITICAL);
  document.getElementById("highRange").textContent = fmt(t.HIGH);
  document.getElementById("mediumRange").textContent = fmt(t.MEDIUM);
  document.getElementById("lowRange").textContent = fmt(t.LOW);
}

// ===================================
// Render real held-out test metrics
// ===================================
function renderMetrics() {
  const m = state.metrics || {};
  const fmt = (v, digits = 3) =>
    typeof v === "number" && isFinite(v) ? v.toFixed(digits) : "—";
  document.getElementById("metricMAE").textContent = fmt(m.mae_days);
  document.getElementById("metricRMSE").textContent = fmt(m.rmse_days);
  document.getElementById("metricR2").textContent = fmt(m.r2);
  const acc = m.accuracy_within_1day;
  document.getElementById("metricAcc").textContent =
    typeof acc === "number" ? `${(acc * 100).toFixed(1)}%` : "—";
  if (!Object.keys(m).length) {
    document.getElementById("metricsNote").textContent =
      "No metrics in GeoJSON metadata — re-run train.py to populate.";
  }
}

// ===================================
// Display
// ===================================
function displayData() {
  if (!state.geojson || !state.geojson.features) return;
  clearLayers();

  const observed  = state.geojson.features.filter(f => f.properties.source === "observed");
  let   predicted = state.geojson.features.filter(f => f.properties.source === "predicted");

  // Latest base_date wins (file may contain history of prior base dates).
  const latestBaseDate = predicted
    .map(f => f.properties.base_date)
    .filter(Boolean)
    .sort()
    .pop() || "N/A";
  document.getElementById("baseDate").textContent = latestBaseDate;
  predicted = predicted.filter(f => f.properties.base_date === latestBaseDate);

  // Apply day filter (real model output, no smoothing)
  const daySelected = state.selectedDay;
  const dayFiltered = daySelected === "all"
    ? predicted
    : predicted.filter(f => f.properties.days_until_fire === Number(daySelected));

  document.getElementById("daySelectorInfo").textContent =
    daySelected === "all"
      ? `Showing all ${predicted.length} predicted cells.`
      : `Showing ${dayFiltered.length} cells predicted to fire on ` +
        `${dateAdd(latestBaseDate, Number(daySelected))} (Day +${daySelected}).`;

  if (document.getElementById("showObserved").checked) {
    state.layers.observed = createObservedLayer(observed);
  }
  if (document.getElementById("showPredicted").checked) {
    state.layers.predicted = createPredictedLayer(dayFiltered);
  }

  updateStatistics(predicted); // urgency summary always reflects all 7 days
  updateTimeline(predicted, latestBaseDate);
  addLayersToMap();
}

function dateAdd(isoDate, days) {
  const d = new Date(isoDate);
  if (isNaN(d.getTime())) return "—";
  d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}

// ===================================
// Layers
// ===================================
function createObservedLayer(features) {
  const cluster = document.getElementById("clusterMarkers").checked;
  const layer = cluster ? L.markerClusterGroup() : L.layerGroup();

  features.forEach(f => {
    const [lon, lat] = f.geometry.coordinates;
    const props = f.properties;
    const marker = L.circleMarker([lat, lon], {
      radius: 6, fillColor: "#ff5722", color: "#fff", weight: 1, fillOpacity: 0.8,
    });
    marker.bindPopup(`
      <div class="popup">
        <b style="color:#ff5722;">🔥 Observed Fire (FIRMS)</b><br>
        <small>Date: ${props.date}</small><br>
        <small>FIRMS detections: ${props.fire_count ?? "—"}</small><br>
        <small>Location: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small>
      </div>
    `);
    layer.addLayer(marker);
  });

  return layer;
}

function createPredictedLayer(features) {
  const cluster = document.getElementById("clusterMarkers").checked;
  const layer = cluster ? L.markerClusterGroup() : L.layerGroup();

  features.forEach(f => {
    const [lon, lat] = f.geometry.coordinates;
    const p = f.properties;

    const color = URGENCY_COLORS[p.urgency_level] || URGENCY_COLORS.NONE;
    const size  = URGENCY_SIZES[p.urgency_level]  || URGENCY_SIZES.NONE;

    const marker = L.circleMarker([lat, lon], {
      radius: size, fillColor: color, color: "#fff", weight: 1, fillOpacity: 0.85,
    });

    const fireDate   = p.predicted_fire_date;
    const daysUntil  = p.days_until_fire;
    const confidence = (p.confidence != null) ? (p.confidence * 100).toFixed(0) : "—";
    const rawPred    = (p.raw_prediction != null) ? Number(p.raw_prediction).toFixed(2) : null;
    const histCount  = p.historical_fire_count_30d;

    let html = `
      <div class="popup" style="min-width:220px;">
        <b style="color:${color};">🔮 Fire Prediction</b><br>
        <div class="popup-block">
          <b>Predicted: ${fireDate}</b><br>
          <small>In ${daysUntil} day${daysUntil !== 1 ? "s" : ""}` +
          (rawPred ? ` (raw=${rawPred})` : "") + `</small>
        </div>
        <small>Urgency: <b>${p.urgency_level}</b> (calibrated)</small><br>
        <small>Confidence (rounding proxy): ${confidence}%</small><br>`;
    if (histCount != null) {
      html += `<small>Historical fires (30d, FIRMS): <b>${histCount}</b></small><br>`;
    }
    html += `<small>Cell: ${lat.toFixed(3)}°, ${lon.toFixed(3)}°</small></div>`;
    marker.bindPopup(html);

    layer.addLayer(marker);
  });

  return layer;
}

// ===================================
// Stats / timeline
// ===================================
function updateStatistics(predicted) {
  const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
  predicted.forEach(f => {
    const u = f.properties.urgency_level;
    if (u in counts) counts[u]++;
  });
  document.getElementById("criticalCount").textContent = counts.CRITICAL;
  document.getElementById("highCount").textContent = counts.HIGH;
  document.getElementById("mediumCount").textContent = counts.MEDIUM;
  document.getElementById("lowCount").textContent = counts.LOW;
}

function updateTimeline(predicted, baseDate) {
  const timeline = document.getElementById("timeline");
  timeline.innerHTML = "";

  const dayCounts = {};
  for (let i = 0; i <= 7; i++) dayCounts[i] = 0;
  predicted.forEach(f => {
    const d = f.properties.days_until_fire;
    if (d >= 0 && d <= 7) dayCounts[d]++;
  });

  for (let i = 0; i <= 7; i++) {
    const dateStr = dateAdd(baseDate, i);
    const count = dayCounts[i];
    const label = i === 0 ? "Today" : i === 1 ? "Tomorrow" : `+${i} days`;

    const item = document.createElement("div");
    item.className = "timeline-item" + (count > 0 ? " has-fires" : "");
    item.innerHTML = `
      <div class="timeline-day">${label}</div>
      <div class="timeline-date">${dateStr}</div>
      <div class="timeline-count">${count} fire${count !== 1 ? "s" : ""}</div>
    `;
    item.style.cursor = "pointer";
    item.addEventListener("click", () => selectDay(String(i)));
    timeline.appendChild(item);
  }
}

// ===================================
// Layer mgmt
// ===================================
function clearLayers() {
  for (const k of Object.keys(state.layers)) {
    if (state.layers[k]) {
      map.removeLayer(state.layers[k]);
      state.layers[k] = null;
    }
  }
}
function addLayersToMap() {
  for (const k of Object.keys(state.layers)) {
    if (state.layers[k]) map.addLayer(state.layers[k]);
  }
}

// ===================================
// Day selector
// ===================================
function selectDay(day) {
  state.selectedDay = day;
  document.querySelectorAll(".day-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.day === String(day));
  });
  displayData();
}

function bindEvents() {
  document.querySelectorAll(".day-btn").forEach(btn => {
    btn.addEventListener("click", () => selectDay(btn.dataset.day));
  });
  document.getElementById("showObserved").addEventListener("change", displayData);
  document.getElementById("showPredicted").addEventListener("change", displayData);
  document.getElementById("clusterMarkers").addEventListener("change", displayData);
}

// ===================================
// Start
// ===================================
init();
