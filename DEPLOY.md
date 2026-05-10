# Production deploy — Railway

Single Docker image, two services in one Railway project, one shared volume.

```
                         ┌──────────────────────────────────┐
                         │   Railway Project: fire-date     │
                         │                                  │
   GitHub ── push ──►   ─┤  ┌──────────┐    ┌──────────┐    │
                         │  │   web    │    │   cron   │    │
                         │  │ uvicorn  │    │ pipeline │    │
                         │  │ :PORT    │    │ daily 6Z │    │
                         │  └────┬─────┘    └────┬─────┘    │
                         │       │ /srv          │ /srv     │
                         │       └─────┬─────────┘          │
                         │             ▼                    │
                         │     ┌───────────────┐            │
                         │     │ Volume "srv"  │            │
                         │     │ /srv/data     │            │
                         │     │ /srv/outputs  │            │
                         │     └───────────────┘            │
                         └──────────────────────────────────┘
```

The `web` service serves `/api/...` + `/geojson` + the built React SPA at `/`.
The `cron` service runs `fetch_firms.py && train.py` daily, writing to the
same `/srv` volume so the next API request reads the freshly-trained model.

## What's in the box

| File | Purpose |
|---|---|
| `Dockerfile` | Multi-stage: Node 20 builds the SPA → Python 3.11 runs FastAPI + serves the SPA. |
| `railway.json` | Web-service config: builder=DOCKERFILE, healthcheck=`/health`. |
| `railway.cron.json` | Cron-service config: same image, runs the pipeline at 06:00 UTC. |
| `web/` | Vite + React + TypeScript dashboard (replaces `frontend/`). |
| `src/api.py` | FastAPI app — now honors `OUTPUT_DIR`, mounts the SPA at `/`. |
| `src/fetch_firms.py`, `src/train.py`, `src/risk_map.py` | Existing pipeline; runs in the cron service. |

The legacy `frontend/` directory is left untouched for local dev fallback —
delete it once Railway is the source of truth.

## First-time setup

1. **Push the repo to GitHub** (Railway deploys from a GitHub-connected repo).

2. **Create a Railway project + the web service:**
   - New Project → Deploy from GitHub Repo → pick the repo.
   - Railway will read `railway.json` automatically (DOCKERFILE builder, /health check).
   - Set environment variables on this service:
     ```
     FIRMS_API_KEY = <your NASA FIRMS map key>
     ```
     All other path env vars (`OUTPUT_DIR`, `DATA_DIR`, `WEB_DIST_DIR`, …)
     are baked into the Dockerfile defaults.

3. **Create a Volume:**
   - Project → Volumes → New Volume → name it `srv`, size 1–5 GB.
   - Attach to the web service at mount path `/srv`.
   - The Dockerfile pre-creates `/srv/data/{raw,firms,weather}` and `/srv/outputs`
     so the volume comes up with the right structure on first mount.

4. **Create the cron service** (same repo, same image, different command):
   - Project → New Service → Deploy from GitHub → pick the **same repo**.
   - In the service settings, set the config file path to `railway.cron.json`
     (Settings → Service → Config Path).
   - Attach the **same volume** `srv` at `/srv`.
   - Copy `FIRMS_API_KEY` env var across.
   - The cron service runs once at 06:00 UTC daily and exits.

5. **Bootstrap the model** (the API needs an existing model to start):

   Either trigger the cron service manually from the Railway UI for its first
   run, OR train locally + commit `outputs/` and let the first deploy ship
   with a model. Locally:
   ```bash
   cd src && python fetch_firms.py && python train.py
   git add outputs/ && git commit -m "bootstrap: initial trained model"
   git push
   ```
   The first deploy then has a usable model and the cron starts refreshing it
   on the daily schedule.

## Local dev parity

Two ways to run locally with the same architecture:

**A. Native (fast iteration on the SPA):**
```bash
# terminal 1 — backend
cd src && uvicorn api:app --reload

# terminal 2 — Vite dev server with proxy → :8000
cd web && npm install && npm run dev
# open http://localhost:5173
```

The Vite proxy forwards `/geojson`, `/api`, `/health`, `/predict*`,
`/predictions*`, `/metrics`, `/metadata` to FastAPI on `:8000` — same paths
the production single-origin setup uses, so no code branches.

**B. Docker (closer to prod):**
```bash
docker build -t fire-date .
docker run --rm -p 8000:8000 \
  -e FIRMS_API_KEY=$FIRMS_API_KEY \
  -v $(pwd)/outputs:/srv/outputs \
  -v $(pwd)/data:/srv/data \
  fire-date
# open http://localhost:8000
```

## Env vars

| Var | Default | Set where |
|---|---|---|
| `FIRMS_API_KEY` | — (required) | Both services in Railway |
| `PORT` | 8000 | Railway sets automatically |
| `OUTPUT_DIR` | `/srv/outputs` | Dockerfile (override per service if needed) |
| `DATA_DIR`, `RAW_DIR`, `FIRMS_DIR`, `WEATHER_DIR`, `FIRMS_PATH` | `/srv/data/...` | Dockerfile |
| `WEB_DIST_DIR` | `/app/web/dist` | Dockerfile (where the build stage drops the SPA) |
| `VITE_API_BASE` | empty (same-origin) | Set on the web service if hosting the SPA on a different origin |
| `MIN_HISTORICAL_FIRES_FOR_DISPLAY` etc. | see CLAUDE.md | Optional risk-map filters |

## Operational notes

- **Cost**: one Hobby plan ($5/mo) usually covers both services for moderate
  traffic. Volume billed separately by GB.
- **Cold starts**: the API loads the LightGBM model and feature parquet at
  startup. With ~70 features and a few MB of features, startup is ~3–5 s.
  Healthcheck timeout is 60 s in `railway.json` to absorb that.
- **Cron failure**: `restartPolicyType: NEVER` on the cron service means a
  failed run does NOT auto-retry — you'll see the failure in Railway logs at
  the next firing. Investigate before relying on the next day's prediction.
- **GeoJSON staleness**: `/health` exposes `geojson_age_days`. The SPA shows
  a freshness badge derived from `base_date` separately. If `geojson_age_days
  > 5`, the cron likely failed silently — check Railway logs.
- **Volume contents survive redeploys** of the web image, so model retraining
  doesn't have to happen on every deploy.

## Updating the dashboard

The SPA is built into the image at `docker build` time. Any change in `web/`
needs a redeploy:

```bash
git push origin main   # Railway redeploys automatically
```

To preview before pushing:

```bash
cd web && npm run build && npm run preview   # serves dist/ on :4173
```

## Rollback

Railway keeps prior deploys — pick a previous deployment in the dashboard and
"Redeploy" to roll back the code. Volume contents (model artifacts) are NOT
rolled back; if a bad model is in the volume, manually trigger the cron
service's `train.py` step from a working commit.
