# syntax=docker/dockerfile:1.7

# =============================================================================
# Stage 1 — build the Vite + React + TypeScript SPA
# =============================================================================
FROM node:20-alpine AS web-build

WORKDIR /web

# package.json + lockfile first → cached layer for dep installs across rebuilds.
COPY web/package.json web/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY web/ ./
RUN npm run build


# =============================================================================
# Stage 2 — Python runtime serving FastAPI + the built SPA
# =============================================================================
FROM python:3.11-slim AS runtime

# rasterio depends on system GDAL/PROJ; LightGBM needs libgomp.
# wget is in base; curl is for the health check.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        libgomp1 libexpat1 libgdal32 libproj25 \
        ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    # All persistent state lives under /srv so a single Railway volume mounted
    # there is shared between the web service and the cron pipeline service.
    # api.py + train.py + fetch_firms.py read these via os.environ.
    OUTPUT_DIR=/srv/outputs \
    DATA_DIR=/srv/data \
    RAW_DIR=/srv/data/raw \
    FIRMS_DIR=/srv/data/firms \
    FIRMS_PATH=/srv/data/firms/firms_all.parquet \
    WEATHER_DIR=/srv/data/weather \
    WEB_DIST_DIR=/app/web/dist

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy Python source + supporting artifacts (data/ is volume-mounted in prod).
COPY src/ /app/src/
COPY env.example /app/env.example

# Pull the built SPA out of the web-build stage. WEB_DIST_DIR points here so
# api.py's catch-all SPA fallback finds index.html and the assets/ subtree.
COPY --from=web-build /web/dist /app/web/dist

# Default to non-root for the API service. The cron service overrides USER
# to root in railway.cron.json since it writes into the mounted volume.
RUN useradd --create-home --shell /usr/sbin/nologin appuser \
 && mkdir -p /srv/outputs /srv/data/raw /srv/data/firms /srv/data/weather \
 && chown -R appuser:appuser /app /srv
USER appuser

# Railway sets $PORT; default to 8000 for `docker run` locally.
ENV PORT=8000
EXPOSE 8000

# Healthcheck hits the structured /health route written into api.py.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

WORKDIR /app/src
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
