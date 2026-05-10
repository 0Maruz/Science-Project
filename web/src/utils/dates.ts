export function dateAdd(isoDate: string, days: number): string {
  const d = new Date(isoDate);
  if (isNaN(d.getTime())) return "—";
  d.setDate(d.getDate() + days);
  return d.toISOString().split("T")[0];
}

export interface FreshnessInfo {
  cls: "fresh" | "stale" | "expired";
  label: string;
  msg: string;
}

export function computeFreshness(baseDateIso: string | null): FreshnessInfo | null {
  if (!baseDateIso || baseDateIso === "N/A") {
    return { cls: "expired", label: "no data", msg: "" };
  }
  const base = new Date(baseDateIso);
  const today = new Date();
  base.setHours(0, 0, 0, 0);
  today.setHours(0, 0, 0, 0);
  const lagDays = Math.round((today.getTime() - base.getTime()) / 86400000);

  if (lagDays <= 0)
    return { cls: "fresh", label: "live", msg: "Data is current — Day +1 = real tomorrow." };
  if (lagDays === 1)
    return { cls: "fresh", label: "1 day old", msg: "Yesterday's data — Day +1 = real today." };
  if (lagDays <= 3)
    return {
      cls: "stale",
      label: `${lagDays} d behind`,
      msg: `Last FIRMS pull was ${lagDays} days ago. Run fetch_firms.py + risk_map.py to refresh.`,
    };
  return {
    cls: "expired",
    label: `${lagDays} d behind`,
    msg: `Data is ${lagDays} days stale — predictions may not reflect current conditions. Refresh with fetch_firms.py.`,
  };
}
