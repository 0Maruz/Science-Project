#!/usr/bin/env bash
# =====================================================================
# 🔥 Fire Date Predictor — one-shot runner
#
# Real data only. No simulated values, no fake fallbacks.
#
# Default behaviour:
#   1. Train on whatever real FIRMS data is already in data/raw + data/firms/
#   2. Refresh outputs/riskmap/fire_dates_all.geojson (real model output)
#   3. Serve the static dashboard on  http://localhost:8080/frontend/
#   4. Serve the FastAPI on            http://localhost:8000/
#
# Optional flags fetch the latest real hotspots and/or real ERA5 weather
# before training. Skip training entirely with --no-train.
# =====================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

FETCH_FIRMS=0
FETCH_WEATHER=0
DO_TRAIN=1
OPEN_BROWSER=0
WEATHER_ARGS=()
TRAIN_ARGS=()
HTTP_PORT=8080
API_PORT=8000

usage() {
  cat <<EOF
Usage: $0 [flags] [-- <extra args forwarded to train.py>]

Real-data pipeline orchestrator. Runs train.py (which triggers risk_map.run()
inside it) and then serves both the dashboard and the FastAPI.

Flags:
  --fresh                Fetch latest NASA FIRMS VIIRS NRT hotspots first
                         (requires FIRMS_API_KEY in .env). Falls back to the
                         cached data if the fetch fails.
  --weather              Run fetch_weather.py to pull real ERA5 daily
                         aggregates from Open-Meteo (no API key) before
                         training so weather features are included.
  --weather-arg <ARG>    Pass an extra arg to fetch_weather.py (repeatable).
                         e.g. --weather-arg --limit-cells --weather-arg 50
  --no-train             Skip training; just start the servers using whatever
                         is already in outputs/.
  --open                 Open the dashboard in the default browser when ready.
  --http-port N          Port for the static dashboard server (default 8080).
  --api-port  N          Port for FastAPI (default 8000).
  -h, --help             Show this message.

Anything after a literal "--" is forwarded verbatim to train.py, e.g.:
  $0 --fresh -- --n-iter 30 --only lightgbm,xgboost

Real data sources:
  • NASA FIRMS VIIRS NRT  (always)
  • Open-Meteo ERA5        (only if --weather is used or cache exists)
EOF
}

# ── Parse flags ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --fresh)        FETCH_FIRMS=1; shift ;;
    --weather)      FETCH_WEATHER=1; shift ;;
    --weather-arg)  WEATHER_ARGS+=("$2"); shift 2 ;;
    --no-train)     DO_TRAIN=0; shift ;;
    --open)         OPEN_BROWSER=1; shift ;;
    --http-port)    HTTP_PORT="$2"; shift 2 ;;
    --api-port)     API_PORT="$2"; shift 2 ;;
    -h|--help)      usage; exit 0 ;;
    --)             shift; TRAIN_ARGS+=("$@"); break ;;
    *)              echo "Unknown flag: $1" >&2; usage >&2; exit 1 ;;
  esac
done

# ── Activate virtualenv if present ──────────────────────────────────────
if [[ -d ".venv" ]]; then
  echo "→ Activating .venv"
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Sanity-check Python deps without installing anything (no surprise side effects)
if ! python -c "import fastapi, uvicorn, pandas, sklearn, joblib" >/dev/null 2>&1; then
  cat <<'EOF' >&2
❌ Missing Python deps. Run once:
     python -m venv .venv && source .venv/bin/activate
     pip install -r requirements.txt
EOF
  exit 1
fi

# ── Stage 1: optional FIRMS fetch (real NASA data) ──────────────────────
if [[ $FETCH_FIRMS -eq 1 ]]; then
  echo "→ Fetching latest FIRMS VIIRS NRT hotspots …"
  if ! (cd src && python fetch_firms.py); then
    echo "  ⚠️  FIRMS fetch failed (likely missing/exhausted FIRMS_API_KEY)."
    echo "      Continuing with whatever real data is already cached."
  fi
fi

# ── Stage 2: optional ERA5 weather fetch (real Open-Meteo data) ─────────
if [[ $FETCH_WEATHER -eq 1 ]]; then
  echo "→ Fetching real ERA5 daily weather (Open-Meteo Archive) …"
  (cd src && python fetch_weather.py "${WEATHER_ARGS[@]}")
fi

# ── Stage 3: train (loads real features → tunes → persists artefacts) ───
if [[ $DO_TRAIN -eq 1 ]]; then
  echo "→ Training on real FIRMS data (+ ERA5 weather if cached) …"
  (cd src && python train.py "${TRAIN_ARGS[@]}")
fi

# ── Sanity check artefacts before serving ───────────────────────────────
if [[ ! -f "outputs/models/lgbm_fire_date_model.pkl" ]]; then
  cat <<'EOF' >&2
❌ No trained model at outputs/models/lgbm_fire_date_model.pkl
   Run without --no-train, or place a trained artefact there first.
EOF
  exit 1
fi
if [[ ! -f "outputs/riskmap/fire_dates_all.geojson" ]]; then
  echo "→ GeoJSON missing; running risk_map.py to generate it from real data …"
  (cd src && python risk_map.py)
fi

# ── Stage 4: serve the dashboard + API ──────────────────────────────────
DASHBOARD_URL="http://localhost:${HTTP_PORT}/frontend/"
API_URL="http://localhost:${API_PORT}/"

# Background HTTP server for the static dashboard (so ../outputs/... resolves).
echo "→ Static dashboard  →  ${DASHBOARD_URL}"
python -m http.server "${HTTP_PORT}" >/dev/null 2>&1 &
HTTP_PID=$!

cleanup() {
  echo
  echo "→ Shutting down servers …"
  if [[ -n "${HTTP_PID:-}" ]] && kill -0 "${HTTP_PID}" 2>/dev/null; then
    kill "${HTTP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ $OPEN_BROWSER -eq 1 ]]; then
  if command -v xdg-open >/dev/null 2>&1; then
    (sleep 1 && xdg-open "${DASHBOARD_URL}") &
  elif command -v open >/dev/null 2>&1; then
    (sleep 1 && open "${DASHBOARD_URL}") &
  fi
fi

cat <<EOF

✅ Ready.
   Dashboard:  ${DASHBOARD_URL}
   API:        ${API_URL}
   API docs:   ${API_URL}docs

   All values shown are derived from real data sources only:
     • NASA FIRMS VIIRS NRT hotspots
     • Open-Meteo ERA5 reanalysis (if cached)
     • Calendar derived from real dates
   No synthetic, simulated, or interpolated values.

   Press Ctrl+C to stop both servers.

EOF

# Foreground FastAPI — Ctrl+C tears down the HTTP server via the trap.
(cd src && exec uvicorn api:app --host 0.0.0.0 --port "${API_PORT}")
