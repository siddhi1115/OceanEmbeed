"""
main_vercel.py
--------------
Vercel-compatible FastAPI entry point for OceanEmbed.

This is a LIGHTWEIGHT gateway that:
  - Serves static metadata endpoints directly
  - Proxies heavy ML inference endpoints to ML_BACKEND_URL
    (your Docker/Render/Cloud Run deployment)

Set the environment variable in Vercel dashboard:
  ML_BACKEND_URL=https://your-ml-backend.onrender.com

Run locally (dev):
  ML_BACKEND_URL=http://localhost:8000 uvicorn backend.main_vercel:app --reload --port 3001
"""

import os
from typing import Optional

import httpx
import orjson
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

ML_BACKEND_URL = os.environ.get("ML_BACKEND_URL", "").rstrip("/")

app = FastAPI(
    title="OceanEmbed API (Vercel Gateway)",
    description=(
        "Subsurface ocean temperature reconstruction from satellite surface inputs. "
        "This gateway proxies ML inference to a dedicated backend service. "
        "SIH26066 / MoES / INCOIS."
    ),
    version="1.0.0-prototype",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _require_backend():
    if not ML_BACKEND_URL:
        raise HTTPException(
            status_code=503,
            detail=(
                "ML_BACKEND_URL environment variable is not set. "
                "Deploy the Docker backend (see Dockerfile.backend) and set "
                "ML_BACKEND_URL in your Vercel project environment variables."
            )
        )


async def _proxy_get(path: str, params: dict = None):
    _require_backend()
    url = f"{ML_BACKEND_URL}{path}"
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            resp = await client.get(url, params=params)
            return JSONResponse(
                content=orjson.loads(resp.content),
                status_code=resp.status_code,
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"ML backend unreachable: {e}")


async def _proxy_post(path: str, body: dict):
    _require_backend()
    url = f"{ML_BACKEND_URL}{path}"
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.post(url, json=body)
            return JSONResponse(
                content=orjson.loads(resp.content),
                status_code=resp.status_code,
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"ML backend unreachable: {e}")


# ─── Lightweight endpoints (no ML needed) ─────────────────────────────────────

@app.get("/api/health")
async def health():
    backend_ok = False
    backend_note = "ML_BACKEND_URL not configured"
    if ML_BACKEND_URL:
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                r = await client.get(f"{ML_BACKEND_URL}/api/health")
                backend_ok = r.status_code == 200
                backend_note = r.json().get("status", "unknown") if backend_ok else f"HTTP {r.status_code}"
        except Exception as e:
            backend_note = f"unreachable: {e}"
    return {
        "status": "ok",
        "gateway": "vercel",
        "ml_backend": backend_note,
        "ml_backend_url": ML_BACKEND_URL or "(not set)",
        "version": "1.0.0-prototype",
        "data_period": "2020-01-01 to 2020-01-05",
        "disclaimer": (
            "Prototype on sample data. Metrics demonstrate the pipeline, "
            "not operational skill."
        ),
    }


@app.get("/api/model/depths")
async def depths():
    """Return the target depth levels (static, no ML needed)."""
    return {"depths_m": TARGET_DEPTHS}


# ─── Proxied endpoints (requires ML backend) ──────────────────────────────────

@app.get("/api/dates")
async def dates():
    return await _proxy_get("/api/dates")


@app.get("/api/surface")
async def surface(
    date: str = Query(...),
    variable: str = Query("sst"),
):
    return await _proxy_get("/api/surface", {"date": date, "variable": variable})


class ReconstructRequest(BaseModel):
    date: str
    depth_m: Optional[float] = None


@app.post("/api/reconstruct")
async def reconstruct(req: ReconstructRequest):
    return await _proxy_post("/api/reconstruct", req.model_dump())


@app.get("/api/profile")
async def profile(
    lat: float = Query(...),
    lon: float = Query(...),
    date: str = Query(...),
):
    return await _proxy_get("/api/profile", {"lat": lat, "lon": lon, "date": date})


@app.get("/api/section")
async def section(
    date: str = Query(...),
    lat: Optional[float] = Query(None),
    lon: Optional[float] = Query(None),
):
    params = {"date": date}
    if lat is not None:
        params["lat"] = lat
    if lon is not None:
        params["lon"] = lon
    return await _proxy_get("/api/section", params)


@app.get("/api/metrics")
async def metrics():
    return await _proxy_get("/api/metrics")


@app.get("/api/argo")
async def argo(date: Optional[str] = Query(None)):
    params = {}
    if date:
        params["date"] = date
    return await _proxy_get("/api/argo", params)


@app.get("/api/model/info")
async def model_info():
    return await _proxy_get("/api/model/info")


@app.get("/api/d26")
async def d26(date: str = Query(...)):
    return await _proxy_get("/api/d26", {"date": date})


@app.get("/api/ohc")
async def ohc(
    date: str = Query(...),
    depth_min_m: float = Query(0.0),
    depth_max_m: float = Query(300.0),
):
    return await _proxy_get("/api/ohc", {
        "date": date,
        "depth_min_m": depth_min_m,
        "depth_max_m": depth_max_m,
    })


@app.get("/api/export")
async def export(
    date: str = Query(...),
    depth_m: Optional[float] = Query(None),
    format: str = Query("csv"),
):
    _require_backend()
    params = {"date": date, "format": format}
    if depth_m is not None:
        params["depth_m"] = depth_m
    url = f"{ML_BACKEND_URL}/api/export"
    async with httpx.AsyncClient(timeout=120.0) as client:
        try:
            resp = await client.get(url, params=params)
            return StreamingResponse(
                content=resp.aiter_bytes(),
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=resp.headers.get("content-type", "application/octet-stream"),
            )
        except httpx.RequestError as e:
            raise HTTPException(502, f"ML backend unreachable: {e}")
