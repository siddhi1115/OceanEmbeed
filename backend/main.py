"""
main.py
-------
FastAPI application entry point for OceanEmbed.

Run (locally):
  cd backend
  uvicorn main:app --reload --port 8000

Run (Docker):
  docker-compose up backend
"""

import json
import io
import csv
from datetime import datetime
from pathlib import Path
from typing import Optional, List

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

# Add parent to path for ml/ and pipeline/ imports
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from backend.inference import (
    get_available_dates,
    build_grid_coords,
    reconstruct_full,
    reconstruct_baseline_full,
    get_glorys_full,
    get_surface_variable,
    get_model,
    get_land_mask,
    get_config,
    TARGET_DEPTHS,
)

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="OceanEmbed API",
    description=(
        "Subsurface ocean temperature reconstruction from satellite surface inputs. "
        "Prototype on 5-day sample data (2020-01-01 to 2020-01-05). "
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

# ─── Helpers ──────────────────────────────────────────────────────────────────

METRICS_PATH = ROOT / "backend" / "metrics.json"


def _load_metrics():
    if not METRICS_PATH.exists():
        raise HTTPException(
            status_code=503,
            detail="metrics.json not found. Run ml/evaluate.py first."
        )
    with open(METRICS_PATH) as f:
        return json.load(f)


def _grid_to_json(arr: np.ndarray, lats: list, lons: list,
                  variable: str, units: str) -> dict:
    """Convert (H, W) numpy array to JSON-serialisable dict."""
    data = arr.tolist()
    # Replace NaN with None for JSON
    data = [[None if (v is not None and np.isnan(v)) else v for v in row] for row in data]
    valid = arr[~np.isnan(arr)] if np.any(~np.isnan(arr)) else np.array([0.0])
    return {
        "variable": variable,
        "units": units,
        "latitude": lats,
        "longitude": lons,
        "data": data,
        "colormap_range": [round(float(np.nanpercentile(arr, 2)), 3),
                           round(float(np.nanpercentile(arr, 98)), 3)],
    }


def _nearest_depth_idx(depth_m: float) -> int:
    """Find nearest index in TARGET_DEPTHS."""
    return int(np.argmin(np.abs(np.array(TARGET_DEPTHS) - depth_m)))


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    dates = get_available_dates()
    model_loaded = get_model() is not None
    return {
        "status": "ok",
        "version": "1.0.0-prototype",
        "data_period": "2020-01-01 to 2020-01-05",
        "available_dates": len(dates),
        "model_loaded": model_loaded,
        "disclaimer": (
            "Prototype on sample data. Metrics demonstrate the pipeline, "
            "not operational skill."
        ),
    }


@app.get("/api/dates")
async def dates():
    return {"dates": get_available_dates()}


@app.get("/api/surface")
async def surface(
    date: str = Query(..., description="YYYY-MM-DD"),
    variable: str = Query("sst", description="sst | sss | ssh | u_cur | v_cur | u_wind | v_wind"),
):
    """Return a surface variable grid for the given date."""
    lats, lons = build_grid_coords()
    arr = get_surface_variable(date, variable)
    if arr is None:
        raise HTTPException(
            status_code=404,
            detail=f"Data not found for date={date} variable={variable}. "
                   "Ensure pipeline has been run."
        )
    units_map = {
        "sst": "°C", "sss": "PSU", "ssh": "m", "sla": "m",
        "u_cur": "m/s", "v_cur": "m/s", "u_wind": "m/s", "v_wind": "m/s",
    }
    units = units_map.get(variable, "")
    return _grid_to_json(arr, lats, lons, variable, units)


class ReconstructRequest(BaseModel):
    date: str
    depth_m: Optional[float] = None  # None → all 15 depths


@app.post("/api/reconstruct")
async def reconstruct(req: ReconstructRequest):
    """
    Reconstruct subsurface temperature for a date.
    If depth_m is provided, returns a single (H, W) grid.
    Otherwise returns all 15 depth levels.
    """
    lats, lons = build_grid_coords()
    pred = reconstruct_full(req.date)
    if pred is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded or processed data not found. "
                   "Run ml/train.py and pipeline/harmonize.py first."
        )

    if req.depth_m is not None:
        d_idx = _nearest_depth_idx(req.depth_m)
        actual_depth = TARGET_DEPTHS[d_idx]
        arr = pred[d_idx]
        return {
            "requested_depth_m": req.depth_m,
            "actual_depth_m": actual_depth,
            **_grid_to_json(arr, lats, lons, f"temperature_{actual_depth}m", "°C"),
            "split": "in-sample",
        }
    else:
        # Return all depths
        depths_data = []
        for d_idx, depth in enumerate(TARGET_DEPTHS):
            arr = pred[d_idx]
            depths_data.append({
                "depth_m": depth,
                **_grid_to_json(arr, lats, lons, f"temperature_{depth}m", "°C"),
            })
        return {"depths": depths_data, "split": "in-sample"}


@app.get("/api/profile")
async def profile(
    lat: float = Query(...),
    lon: float = Query(...),
    date: str = Query(...),
):
    """
    Return vertical temperature profiles at a grid cell for:
      - reconstructed (model)
      - GLORYS (reference target)
      - ARGO (if available for this date/location)
    """
    lats, lons = build_grid_coords()
    lat_arr = np.array(lats)
    lon_arr = np.array(lons)
    gi = int(np.argmin(np.abs(lat_arr - lat)))
    gj = int(np.argmin(np.abs(lon_arr - lon)))

    pred = reconstruct_full(date)
    glorys = get_glorys_full(date)

    result = {
        "lat": float(lat_arr[gi]),
        "lon": float(lon_arr[gj]),
        "date": date,
        "depths_m": TARGET_DEPTHS,
        "reconstructed": None,
        "glorys": None,
        "argo": None,
    }

    if pred is not None:
        result["reconstructed"] = [
            round(float(pred[d, gi, gj]), 3) if not np.isnan(pred[d, gi, gj]) else None
            for d in range(len(TARGET_DEPTHS))
        ]

    if glorys is not None:
        result["glorys"] = [
            round(float(glorys[d, gi, gj]), 3) if not np.isnan(glorys[d, gi, gj]) else None
            for d in range(len(TARGET_DEPTHS))
        ]

    # Look for ARGO matchups near this location
    argo_path = ROOT / "data" / "processed" / "argo_matchups.json"
    if argo_path.exists():
        with open(argo_path) as f:
            argo_data = json.load(f)
        for p in argo_data.get("profiles", []):
            if (abs(p["grid_lat"] - float(lat_arr[gi])) < 0.26 and
                    abs(p["grid_lon"] - float(lon_arr[gj])) < 0.26 and
                    p.get("date", "")[:10] == date[:10]):
                depths_map = p.get("temp_at_15depths", {})
                result["argo"] = [
                    depths_map.get(f"{int(d)}m") for d in TARGET_DEPTHS
                ]
                break

    return result


@app.get("/api/section")
async def section(
    date: str = Query(...),
    lat: Optional[float] = Query(None, description="Fix latitude, vary longitude"),
    lon: Optional[float] = Query(None, description="Fix longitude, vary latitude"),
):
    """
    Return a depth-vs-distance cross section.
    Provide either lat (zonal section) or lon (meridional section), not both.
    Returns: {axis_values, depths_m, temps_2d (n_depths × n_axis_points)}
    """
    if lat is None and lon is None:
        raise HTTPException(400, "Provide either lat or lon parameter.")

    lats, lons = build_grid_coords()
    pred = reconstruct_full(date)
    if pred is None:
        raise HTTPException(503, "Model prediction not available.")

    lat_arr = np.array(lats)
    lon_arr = np.array(lons)

    if lat is not None:
        gi = int(np.argmin(np.abs(lat_arr - lat)))
        section_data = pred[:, gi, :]  # (15, n_lons)
        axis = lon_arr.tolist()
        axis_label = "longitude"
        fixed_label = f"lat={float(lat_arr[gi]):.2f}°N"
    else:
        gj = int(np.argmin(np.abs(lon_arr - lon)))
        section_data = pred[:, :, gj]  # (15, n_lats)
        axis = lat_arr.tolist()
        axis_label = "latitude"
        fixed_label = f"lon={float(lon_arr[gj]):.2f}°E"

    # Replace NaN with None for JSON
    temps_2d = []
    for d in range(len(TARGET_DEPTHS)):
        row = section_data[d].tolist()
        row = [None if np.isnan(v) else round(v, 3) for v in row]
        temps_2d.append(row)

    return {
        "date": date,
        "fixed": fixed_label,
        "axis_label": axis_label,
        "axis_values": axis,
        "depths_m": TARGET_DEPTHS,
        "temps_2d": temps_2d,
        "units": "°C",
        "split": "in-sample",
    }


@app.get("/api/metrics")
async def metrics():
    """Return evaluation metrics from metrics.json (written by evaluate.py)."""
    return _load_metrics()


@app.get("/api/argo")
async def argo(date: Optional[str] = Query(None)):
    """Return ARGO profile matchup points (for the given date or all)."""
    argo_path = ROOT / "data" / "processed" / "argo_matchups.json"
    if not argo_path.exists():
        return {"profiles": [], "note": "No ARGO matchups computed yet."}
    with open(argo_path) as f:
        data = json.load(f)
    profiles = data.get("profiles", [])
    if date:
        profiles = [p for p in profiles if p.get("date", "")[:10] == date[:10]]
    return {"profiles": profiles, "n": len(profiles)}


@app.get("/api/model/info")
async def model_info_endpoint():
    """Return model architecture summary."""
    cfg = get_config()
    model = get_model()
    if model is None:
        return {
            "status": "not_loaded",
            "note": "Run ml/train.py to train the model."
        }
    from ml.oceanembed import model_info
    info = model_info(model)
    return {
        **info,
        "model_config": cfg.get("model", {}),
        "input_channels_description": {
            0: "SST (°C)",
            1: "SSS (PSU)",
            2: "SLA (m)",
            3: "U current (m/s)",
            4: "V current (m/s)",
            5: "U wind (m/s)",
            6: "V wind (m/s)",
            7: "Latitude (normalized)",
            8: "Longitude (normalized)",
        },
        "output": "Temperature at 15 depth levels (°C)",
        "depth_levels": TARGET_DEPTHS,
    }


@app.get("/api/export")
async def export(
    date: str = Query(...),
    depth_m: Optional[float] = Query(None),
    format: str = Query("csv", description="csv | nc"),
):
    """
    Export reconstructed temperature data.
    format=csv: returns a CSV with lat, lon, depth columns
    format=nc:  returns a NetCDF4 file (all depths)
    """
    pred = reconstruct_full(date)
    if pred is None:
        raise HTTPException(503, "Model prediction not available.")

    lats, lons = build_grid_coords()
    lat_arr = np.array(lats)
    lon_arr = np.array(lons)

    if format == "csv":
        output = io.StringIO()
        writer = csv.writer(output)

        depths_to_export = [depth_m] if depth_m else TARGET_DEPTHS
        header = ["latitude", "longitude"] + [f"temp_{int(d)}m_C" for d in depths_to_export]
        writer.writerow(header)

        land_mask = get_land_mask()
        for i, lat in enumerate(lat_arr):
            for j, lon in enumerate(lon_arr):
                if land_mask is not None and not land_mask[i, j]:
                    continue  # skip land
                row = [round(float(lat), 2), round(float(lon), 2)]
                for d in depths_to_export:
                    d_idx = _nearest_depth_idx(d)
                    val = pred[d_idx, i, j]
                    row.append(round(float(val), 3) if not np.isnan(val) else "")
                writer.writerow(row)

        output.seek(0)
        filename = f"OceanEmbed_{date}_{'all_depths' if depth_m is None else f'{int(depth_m)}m'}.csv"
        return StreamingResponse(
            io.BytesIO(output.getvalue().encode()),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )

    elif format == "nc":
        try:
            import xarray as xr
            import tempfile
            import os

            ds = xr.Dataset(
                {"temperature": xr.DataArray(
                    pred,
                    dims=["depth", "latitude", "longitude"],
                    coords={
                        "depth": np.array(TARGET_DEPTHS),
                        "latitude": lat_arr,
                        "longitude": lon_arr,
                    },
                    attrs={"units": "degrees_C", "long_name": "Reconstructed sea water temperature"}
                )},
                attrs={
                    "date": date,
                    "source": "OceanEmbed prototype reconstruction",
                    "disclaimer": "In-sample prototype. Not operational.",
                    "split": "in-sample",
                }
            )

            with tempfile.NamedTemporaryFile(suffix=".nc", delete=False) as tmp:
                tmp_path = tmp.name
            ds.to_netcdf(tmp_path)
            with open(tmp_path, "rb") as f:
                content = f.read()
            os.unlink(tmp_path)

            filename = f"OceanEmbed_{date}.nc"
            return StreamingResponse(
                io.BytesIO(content),
                media_type="application/x-netcdf",
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except Exception as e:
            raise HTTPException(500, f"NetCDF export failed: {e}")

    else:
        raise HTTPException(400, f"Unknown format '{format}'. Use 'csv' or 'nc'.")


# ─── Derived products ─────────────────────────────────────────────────────────

@app.get("/api/d26")
async def d26(date: str = Query(...)):
    """
    Compute D26 (depth of 26°C isotherm) from reconstructed profile.
    Returns (H, W) grid in metres.
    """
    pred = reconstruct_full(date)
    if pred is None:
        raise HTTPException(503, "Model prediction not available.")

    lats, lons = build_grid_coords()
    H, W = len(lats), len(lons)
    depths = np.array(TARGET_DEPTHS, dtype=float)
    d26 = np.full((H, W), np.nan)

    for i in range(H):
        for j in range(W):
            profile = pred[:, i, j]
            if np.any(np.isnan(profile)):
                continue
            # Find where temp crosses 26°C (descending from surface)
            if profile[0] < 26.0:
                d26[i, j] = 0.0  # surface already below 26°C
                continue
            crossed = np.where(profile < 26.0)[0]
            if len(crossed) == 0:
                d26[i, j] = depths[-1]  # never crosses, set to max depth
                continue
            idx = crossed[0]
            # Linear interpolation between idx-1 and idx
            d_above = depths[idx - 1]
            d_below = depths[idx]
            t_above = profile[idx - 1]
            t_below = profile[idx]
            frac = (26.0 - t_above) / (t_below - t_above + 1e-9)
            d26[i, j] = d_above + frac * (d_below - d_above)

    return _grid_to_json(d26, lats, lons, "D26", "m")


@app.get("/api/ohc")
async def ohc(
    date: str = Query(...),
    depth_min_m: float = Query(0.0),
    depth_max_m: float = Query(300.0),
):
    """
    Compute Upper Ocean Heat Content (OHC, 0-300 m by default).
    Units: 10⁷ J/m² (standard for OHC)
    """
    pred = reconstruct_full(date)
    if pred is None:
        raise HTTPException(503, "Model prediction not available.")

    lats, lons = build_grid_coords()
    H, W = len(lats), len(lons)
    depths = np.array(TARGET_DEPTHS, dtype=float)
    rho_cp = 4.09e6  # J/(m³·°C) for seawater

    # Select depth indices within range
    d_mask = (depths >= depth_min_m) & (depths <= depth_max_m)
    d_idx = np.where(d_mask)[0]
    if len(d_idx) == 0:
        raise HTTPException(400, f"No depth levels in range {depth_min_m}–{depth_max_m} m")

    ohc = np.full((H, W), np.nan)
    sel_depths = depths[d_idx]
    T_REF = 26.0  # reference temperature (°C)

    for i in range(H):
        for j in range(W):
            profile = pred[d_idx, i, j]
            if np.any(np.isnan(profile)):
                continue
            # Trapezoidal integration of (T - T_ref) with depth
            dT = profile - T_REF
            # Use trapezoidal rule
            ohc_val = rho_cp * np.trapz(dT, sel_depths)
            ohc[i, j] = ohc_val / 1e7  # convert to 10⁷ J/m²

    return _grid_to_json(ohc, lats, lons, "OHC", "10⁷ J/m²")
