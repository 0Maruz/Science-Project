import { useEffect, useMemo, useState } from "react";
import { fetchGeoJson } from "./api";
import MapView from "./components/MapView";
import Sidebar from "./components/Sidebar";
import type {
  DaySelection,
  DisplayOptions,
  FireGeoJson,
} from "./types";
import { exportCellsCsv } from "./utils/csv";
import { dateAdd } from "./utils/dates";

const DEFAULT_OPTIONS: DisplayOptions = {
  showObserved: false,
  showPredicted: true,
  clusterMarkers: false,
  showCellPins: false,
  heatRadius: 50,
};

export default function App() {
  const [geojson, setGeojson] = useState<FireGeoJson | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [selectedBaseDate, setSelectedBaseDate] = useState<string>("latest");
  const [selectedProvince, setSelectedProvince] = useState<string>("all");
  const [selectedDay, setSelectedDay] = useState<DaySelection>("all");
  const [options, setOptions] = useState<DisplayOptions>(DEFAULT_OPTIONS);

  // Load GeoJSON once on mount.
  useEffect(() => {
    fetchGeoJson()
      .then(setGeojson)
      .catch((e: Error) => {
        console.error(e);
        setError(
          "Failed to load fire prediction data. Run train.py + risk_map.py first."
        );
      });
  }, []);

  const derived = useMemo(() => {
    if (!geojson) return null;
    const features = geojson.features ?? [];

    const observed = features.filter((f) => f.properties.source === "observed");
    const predictedAll = features.filter(
      (f) => f.properties.source === "predicted"
    );

    const allBaseDates = [
      ...new Set(predictedAll.map((f) => f.properties.base_date).filter(Boolean) as string[]),
    ].sort();
    const latestBaseDate = allBaseDates[allBaseDates.length - 1] ?? "N/A";

    let activeBaseDate = selectedBaseDate;
    if (
      activeBaseDate === "latest" ||
      !allBaseDates.includes(activeBaseDate)
    ) {
      activeBaseDate = latestBaseDate;
    }

    const snapshotPredicted = predictedAll.filter(
      (f) => f.properties.base_date === activeBaseDate
    );

    const provinceSet = new Set(
      snapshotPredicted
        .map((f) => (f.properties.province ?? "").trim())
        .filter(Boolean)
    );
    const provinces = [...provinceSet].sort();

    let provinceFiltered = snapshotPredicted;
    let resolvedProvince = selectedProvince;
    if (selectedProvince !== "all") {
      if (provinceSet.has(selectedProvince)) {
        provinceFiltered = snapshotPredicted.filter(
          (f) => f.properties.province === selectedProvince
        );
      } else {
        // Snapshot doesn't contain the picked province — report this so the
        // App effect can reset state (next render).
        resolvedProvince = "all";
      }
    }

    const dayFiltered =
      selectedDay === "all"
        ? provinceFiltered
        : provinceFiltered.filter(
            (f) => f.properties.days_until_fire === Number(selectedDay)
          );

    // Day-selector status message — same wording as the original frontend.
    const daySelectorMessage =
      selectedDay === "all"
        ? `Showing all ${provinceFiltered.length} predicted cells.`
        : `Showing ${dayFiltered.length} cells predicted to fire on ${dateAdd(
            activeBaseDate,
            Number(selectedDay)
          )} (Day +${selectedDay}).`;

    return {
      observed,
      predictedAll,                  // for grid-size detection
      snapshotPredicted: provinceFiltered, // for sidebar urgency / timeline / landcover
      visiblePredicted: dayFiltered,       // map + CSV export
      allBaseDates,
      latestBaseDate,
      activeBaseDate,
      provinces,
      resolvedProvince,
      daySelectorMessage,
    };
  }, [geojson, selectedBaseDate, selectedProvince, selectedDay]);

  // If the active province got reset because the snapshot dropped it, sync
  // the controlled state so the dropdown reflects "all".
  useEffect(() => {
    if (derived && derived.resolvedProvince !== selectedProvince) {
      setSelectedProvince(derived.resolvedProvince);
    }
  }, [derived, selectedProvince]);

  // If the active day went empty in this snapshot, fall back to "all".
  useEffect(() => {
    if (!derived || selectedDay === "all") return;
    const count = derived.snapshotPredicted.filter(
      (f) => f.properties.days_until_fire === Number(selectedDay)
    ).length;
    if (count === 0) setSelectedDay("all");
  }, [derived, selectedDay]);

  const onExportCsv = () => {
    if (!derived || !derived.visiblePredicted.length) {
      alert("Nothing to export — current filter has no cells.");
      return;
    }
    exportCellsCsv(
      derived.visiblePredicted,
      derived.activeBaseDate,
      selectedProvince,
      selectedDay
    );
  };

  const onBaseDateChange = (v: string) => {
    setSelectedBaseDate(v);
    // Reset day filter so the new snapshot's full prediction set is visible.
    setSelectedDay("all");
  };

  if (error) {
    return (
      <div style={{ padding: "40px", color: "#f87171" }}>
        <h2>Failed to load</h2>
        <p>{error}</p>
      </div>
    );
  }

  if (!geojson || !derived) {
    return (
      <div style={{ padding: "40px", color: "#a0a3aa" }}>
        Loading fire prediction data…
      </div>
    );
  }

  const meta = geojson.metadata ?? {};

  return (
    <>
      <Sidebar
        activeBaseDate={derived.activeBaseDate}
        allBaseDates={derived.allBaseDates}
        selectedBaseDate={selectedBaseDate}
        onBaseDateChange={onBaseDateChange}
        provinces={derived.provinces}
        selectedProvince={selectedProvince}
        onProvinceChange={setSelectedProvince}
        selectedDay={selectedDay}
        onDayChange={setSelectedDay}
        predicted={derived.snapshotPredicted}
        visibleCount={derived.visiblePredicted.length}
        daySelectorMessage={derived.daySelectorMessage}
        thresholds={meta.urgency_thresholds ?? null}
        metrics={meta.metrics ?? null}
        metadata={meta}
        options={options}
        onOptionsChange={(o) => setOptions((prev) => ({ ...prev, ...o }))}
        onExportCsv={onExportCsv}
      />

      <MapView
        observed={derived.observed}
        predictedAll={derived.predictedAll}
        predictedVisible={derived.visiblePredicted}
        thresholds={meta.urgency_thresholds ?? null}
        options={options}
      />

      <Legend />
    </>
  );
}

function Legend() {
  return (
    <div className="legend">
      <div className="legend-title">Fire Urgency</div>
      <div className="legend-items">
        <div className="legend-item">
          <div className="legend-color critical" />
          <span>Critical</span>
        </div>
        <div className="legend-item">
          <div className="legend-color high" />
          <span>High</span>
        </div>
        <div className="legend-item">
          <div className="legend-color medium" />
          <span>Medium</span>
        </div>
        <div className="legend-item">
          <div className="legend-color low" />
          <span>Low</span>
        </div>
        <div className="legend-item">
          <div className="legend-color observed" />
          <span>Observed Fire</span>
        </div>
      </div>
    </div>
  );
}
