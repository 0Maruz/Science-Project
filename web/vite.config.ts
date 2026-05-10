import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, the FastAPI backend runs on :8000 and we proxy /geojson + /api/*
// to it so the SPA can talk to a real backend without CORS gymnastics. In
// production, FastAPI serves the built SPA from the same origin (see api.py
// StaticFiles mount), so no proxy is needed.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/geojson": "http://localhost:8000",
      "/api": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/metrics": "http://localhost:8000",
      "/metadata": "http://localhost:8000",
      "/predictions": "http://localhost:8000",
      "/predict": "http://localhost:8000",
    },
  },
  build: {
    outDir: "dist",
    sourcemap: false,
    target: "es2020",
  },
});
