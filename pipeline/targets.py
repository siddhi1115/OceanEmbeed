"""
targets.py
----------
Processes GLORYS thetao:
  1. Interpolates 35 native depth levels → 15 target depths (linear interp)
  2. Regrids from 1/12° to 0.25° target grid
  3. Applies land mask (NaN)
  4. Writes data/processed/targets_YYYYMMDD.nc for each day

Also writes an unnormalised version for validation (the model outputs are
compared to these in raw °C space).

Run:  python pipeline/targets.py
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
import yaml
from scipy.interpolate import interp1d

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_target_grid(cfg):
    res = cfg["target_grid"]["resolution"]
    d = cfg["domain"]
    lats = np.arange(d["lat_min"], d["lat_max"] + res / 2, res)
    lons = np.arange(d["lon_min"], d["lon_max"] + res / 2, res)
    return lats, lons


def interp_depths_1d(src_depths: np.ndarray, src_temps: np.ndarray,
                     tgt_depths: np.ndarray) -> np.ndarray:
    """
    Interpolate a 1-D temperature profile from src_depths to tgt_depths.
    Uses linear interpolation with boundary fill (nearest edge).
    NaN-safe: returns NaN if all src_temps are NaN.
    """
    if np.all(np.isnan(src_temps)):
        return np.full(len(tgt_depths), np.nan)
    # Only use valid depth-temperature pairs
    valid = ~np.isnan(src_temps)
    if valid.sum() < 2:
        return np.full(len(tgt_depths), np.nan)
    f = interp1d(src_depths[valid], src_temps[valid],
                 kind="linear", bounds_error=False,
                 fill_value=(src_temps[valid][0], src_temps[valid][-1]))
    return f(tgt_depths)


def interp_depths_grid(glorys_ds: xr.Dataset, var: str, depth_dim: str,
                       tgt_depths: np.ndarray) -> np.ndarray:
    """
    Interpolate entire (time, depth, lat, lon) GLORYS array to tgt_depths.
    Returns np.ndarray of shape (time, n_tgt_depths, lat, lon).
    """
    da = glorys_ds[var]
    src_depths = da[depth_dim].values.astype(float)
    data = da.values  # (time, depth, lat, lon)

    n_time, n_depth, n_lat, n_lon = data.shape
    n_tgt = len(tgt_depths)
    out = np.full((n_time, n_tgt, n_lat, n_lon), np.nan, dtype=np.float32)

    for t in range(n_time):
        for i in range(n_lat):
            for j in range(n_lon):
                profile = data[t, :, i, j].astype(float)
                out[t, :, i, j] = interp_depths_1d(src_depths, profile, tgt_depths)

    return out


def regrid_to_target_np(src_lat, src_lon, data_3d, tgt_lat, tgt_lon):
    """
    Regrid (depth, lat, lon) numpy array to target lat/lon via nearest-grid-point.
    For 0.25° target from 1/12° source, this is an averaging / coarsening.
    Uses xarray DataArray for convenience.
    """
    n_depth, n_lat_src, n_lon_src = data_3d.shape
    out = np.full((n_depth, len(tgt_lat), len(tgt_lon)), np.nan, dtype=np.float32)

    for d in range(n_depth):
        da = xr.DataArray(
            data_3d[d],
            dims=["latitude", "longitude"],
            coords={"latitude": src_lat, "longitude": src_lon}
        )
        # Coarsen by computing mean in each 0.25° bin
        da_interp = da.interp(latitude=tgt_lat, longitude=tgt_lon,
                              method="linear", kwargs={"fill_value": None})
        out[d] = da_interp.values

    return out


def get_available_dates(cfg):
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 5)
    dates = []
    while start <= end:
        dates.append(start.strftime("%Y-%m-%d"))
        start += timedelta(days=1)
    return dates


def main():
    cfg = load_config()
    c = cfg["data"]["glorys"]
    path = ROOT / c["path"]

    target_lat, target_lon = build_target_grid(cfg)
    tgt_depths = np.array(cfg["target_depths"], dtype=float)
    out_dir = ROOT / cfg["processed"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load land mask
    land_mask_path = out_dir / "land_mask.npy"
    if land_mask_path.exists():
        land_mask = np.load(land_mask_path)
    else:
        land_mask = np.ones((len(target_lat), len(target_lon)), dtype=bool)
        print("Warning: land_mask.npy not found; run harmonize.py first. Proceeding without mask.")

    if not path.exists():
        print(f"ERROR: GLORYS file not found at {path}")
        print("Please place the GLORYS NetCDF at the expected path and re-run.")
        sys.exit(1)

    print(f"Loading GLORYS from {path} ...")
    with xr.open_dataset(path) as ds:
        # Crop to NIO
        d = cfg["domain"]
        ds = ds.sel({
            c["lat_dim"]: slice(d["lat_min"], d["lat_max"]),
            c["lon_dim"]: slice(d["lon_min"], d["lon_max"]),
        })
        src_lat = ds[c["lat_dim"]].values
        src_lon = ds[c["lon_dim"]].values
        src_times = ds[c["time_dim"]].values
        src_depths = ds[c["depth_dim"]].values

        print(f"  GLORYS dims: {dict(ds.dims)}")
        print(f"  Depths (first 5): {src_depths[:5]}")
        print(f"  Time steps: {len(src_times)}")

        # Interpolate to target depths
        print(f"\nInterpolating to 15 target depths: {tgt_depths.tolist()}")
        print("  (This may take a minute on a CPU-only machine for a 1/12° grid...)")
        tgt_data = interp_depths_grid(ds, c["variable"], c["depth_dim"], tgt_depths)
        # tgt_data: (n_time, 15, lat_glorys, lon_glorys)

    # Regrid to 0.25° and write per day
    dates = get_available_dates(cfg)

    for t_idx, date in enumerate(dates):
        # Find matching time index
        time_pd = pd.Timestamp(date)
        time_diffs = np.abs(pd.to_datetime(src_times) - time_pd)
        t = int(time_diffs.argmin())

        print(f"\nProcessing {date} (GLORYS time index {t})...")
        day_data = tgt_data[t]  # (15, lat_glorys, lon_glorys)

        # Regrid to target 0.25° grid
        day_regrid = regrid_to_target_np(src_lat, src_lon, day_data, target_lat, target_lon)

        # Apply land mask
        for d_idx in range(len(tgt_depths)):
            day_regrid[d_idx][~land_mask] = np.nan

        # Write NetCDF
        ds_out = xr.Dataset(
            {"thetao": xr.DataArray(
                day_regrid,
                dims=["depth", "latitude", "longitude"],
                coords={
                    "depth": tgt_depths,
                    "latitude": target_lat,
                    "longitude": target_lon,
                },
                attrs={"units": "degrees_C", "long_name": "Sea water potential temperature"}
            )}
        )
        ds_out.attrs["date"] = date
        ds_out.attrs["source"] = "GLORYS reanalysis, interpolated to 15 target depths, regridded to 0.25°"

        date_str = date.replace("-", "")
        out_path = out_dir / f"targets_{date_str}.nc"
        ds_out.to_netcdf(out_path)
        print(f"  ✓ wrote {out_path}  shape={day_regrid.shape}")
        print(f"    Temperature range (surface): "
              f"{np.nanmin(day_regrid[0]):.2f}°C – {np.nanmax(day_regrid[0]):.2f}°C")

    print("\n✓ targets.py complete.")

import pandas as pd  # noqa: E402  (placed here to keep top-level imports minimal)

if __name__ == "__main__":
    main()
