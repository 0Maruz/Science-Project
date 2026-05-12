import { useEffect } from "react";
import type {
  FireFeature,
  GeoJsonMetadata,
  GistdaFeature,
  UrgencyThresholds,
  ValidationMetrics,
} from "../types";

interface Props {
  open: boolean;
  onClose: () => void;
  activeBaseDate: string;
  predicted: FireFeature[]; // current snapshot (post-province filter)
  observed: FireFeature[]; // all observed features
  liveFires: GistdaFeature[];
  metrics: ValidationMetrics | null;
  thresholds: UrgencyThresholds | null;
  metadata: GeoJsonMetadata | null;
  selectedProvince: string;
  selectedDay: string;
}

// Pop-up with the detail an operator might want before trusting a number on
// the dashboard: training metrics, snapshot composition, urgency thresholds,
// data freshness, etc. Reads everything from already-loaded state — no
// network call.
export default function InfoModal(props: Props) {
  useEffect(() => {
    if (!props.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") props.onClose();
    };
    document.addEventListener("keydown", onKey);
    return () => document.removeEventListener("keydown", onKey);
  }, [props.open, props.onClose]);

  if (!props.open) return null;

  const m = props.metrics ?? {};
  const acc = m.accuracy_within_1day;
  const mae = m.mae_days;
  const rmse = m.rmse_days;
  const r2 = m.r2;

  // Per-tier counts on the current snapshot
  const tiers = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 } as Record<string, number>;
  for (const f of props.predicted) {
    const u = f.properties.urgency_level;
    if (u && u in tiers) tiers[u]++;
  }

  // Per-day counts
  const dayCounts: Record<number, number> = {};
  for (let i = 0; i <= 7; i++) dayCounts[i] = 0;
  for (const f of props.predicted) {
    const d = f.properties.days_until_fire;
    if (d != null && d >= 0 && d <= 7) dayCounts[d]++;
  }

  // Province distribution
  const byProvince: Record<string, number> = {};
  for (const f of props.predicted) {
    const p = (f.properties.province ?? "").trim() || "(unassigned)";
    byProvince[p] = (byProvince[p] ?? 0) + 1;
  }
  const topProvinces = Object.entries(byProvince)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 5);

  const fmt = (v: number | undefined, d = 3) =>
    typeof v === "number" && isFinite(v) ? v.toFixed(d) : "—";
  const fmtPct = (v: number | undefined) =>
    typeof v === "number" ? `${(v * 100).toFixed(1)}%` : "—";

  return (
    <div className="info-modal-backdrop" role="dialog" aria-modal="true" onClick={props.onClose}>
      <div className="info-modal" onClick={(e) => e.stopPropagation()}>
        <div className="info-modal-header">
          <h2>Model & Snapshot Details</h2>
          <button className="info-modal-close" onClick={props.onClose} aria-label="Close">
            ×
          </button>
        </div>

        <div className="info-modal-body">
          <section>
            <h3>Held-out test accuracy</h3>
            <div className="info-grid">
              <div><span>±1 day acc</span><b>{fmtPct(acc)}</b></div>
              <div><span>MAE</span><b>{fmt(mae, 2)} d</b></div>
              <div><span>RMSE</span><b>{fmt(rmse, 2)} d</b></div>
              <div><span>R²</span><b>{fmt(r2, 3)}</b></div>
            </div>
            <p className="info-note">
              From the held-out final 20% of the chronologically-split training
              data. The model never saw these rows during tuning or selection.
            </p>
          </section>

          <section>
            <h3>Current snapshot · {props.activeBaseDate}</h3>
            <div className="info-grid">
              <div><span>Predicted cells</span><b>{props.predicted.length}</b></div>
              <div><span>Observed (FIRMS)</span><b>{props.observed.length}</b></div>
              <div><span>Live (GISTDA)</span><b>{props.liveFires.length || "—"}</b></div>
              <div><span>Province filter</span><b>{props.selectedProvince === "all" ? "all" : props.selectedProvince}</b></div>
            </div>
          </section>

          <section>
            <h3>Urgency distribution</h3>
            <div className="info-grid">
              <div><span style={{ color: "#dc2626" }}>CRITICAL</span><b>{tiers.CRITICAL}</b></div>
              <div><span style={{ color: "#ea580c" }}>HIGH</span><b>{tiers.HIGH}</b></div>
              <div><span style={{ color: "#f59e0b" }}>MEDIUM</span><b>{tiers.MEDIUM}</b></div>
              <div><span style={{ color: "#10b981" }}>LOW</span><b>{tiers.LOW}</b></div>
            </div>
            {props.thresholds && (
              <p className="info-note">
                Cutoffs: ≤{props.thresholds.CRITICAL.toFixed(1)}d CRITICAL,
                ≤{props.thresholds.HIGH.toFixed(1)}d HIGH,
                ≤{props.thresholds.MEDIUM.toFixed(1)}d MEDIUM,
                ≤{props.thresholds.LOW.toFixed(1)}d LOW.
              </p>
            )}
          </section>

          <section>
            <h3>Predicted day distribution</h3>
            <div className="info-bars">
              {Array.from({ length: 8 }, (_, i) => {
                const count = dayCounts[i];
                const max = Math.max(...Object.values(dayCounts), 1);
                const pct = (count / max) * 100;
                const label = i === 0 ? "Today" : `+${i}d`;
                return (
                  <div key={i} className="info-bar-row">
                    <span className="info-bar-label">{label}</span>
                    <div className="info-bar-track"><div className="info-bar-fill" style={{ width: `${pct}%` }} /></div>
                    <span className="info-bar-count">{count}</span>
                  </div>
                );
              })}
            </div>
          </section>

          {topProvinces.length > 0 && (
            <section>
              <h3>Top 5 provinces by predicted cells</h3>
              <table className="info-table">
                <tbody>
                  {topProvinces.map(([name, count]) => (
                    <tr key={name}>
                      <td>{name}</td>
                      <td><b>{count}</b></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </section>
          )}

          <section>
            <h3>Data sources</h3>
            <ul className="info-list">
              <li><b>NASA FIRMS VIIRS NRT</b> — hotspot detections + FRP / brightness, used for training labels and historical aggregates.</li>
              <li><b>GISTDA NRT VIIRS + MODIS</b> — independent hotspot feed from Thailand's national space agency, optional live overlay.</li>
              <li><b>Open-Meteo ERA5</b> — daily reanalysis (temp / precip / wind), used as features only when the weather cache is present.</li>
              <li><b>Hansen GFC v1.11</b> — per-cell tree cover baseline (2000) + recent loss %, distinguishes wildfire risk from agricultural-burn signal.</li>
            </ul>
            <p className="info-note">
              All values shown anywhere in the dashboard come from real sources — no synthetic, interpolated, or simulated data.
            </p>
          </section>

          <section>
            <h3>Selected filter</h3>
            <p className="info-note">
              Day filter: <b>{props.selectedDay === "all" ? "all" : `+${props.selectedDay}d`}</b> · Province: <b>{props.selectedProvince === "all" ? "all" : props.selectedProvince}</b>
            </p>
          </section>
        </div>
      </div>
    </div>
  );
}
