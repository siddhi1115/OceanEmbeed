"""
harmonize.py
------------
Reads every raw dataset, crops to NIO, regrids to 0.25° target grid,
aligns to daily timestamps, fixes units, applies land mask, normalises,
and writes data/processed/inputs_YYYYMMDD.nc for each available day.

Normalization statistics (mean / std per channel) are saved to
data/processed/norm_stats.json so the backend can apply them at inference
time without touching raw files.

Run:  python pipeline/harmonize.py
      python pipeline/harmonize.py --dates 2020-01-01 2020-01-03
"""

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"


# ─── helpers ──────────────────────────────────────────────────────────────────

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_target_grid(cfg):
    """Return (target_lat, target_lon) 1-D arrays at 0.25° spacing."""
    res = cfg["target_grid"]["resolution"]
    d = cfg["domain"]
    lats = np.arange(d["lat_min"], d["lat_max"] + res / 2, res)
    lons = np.arange(d["lon_min"], d["lon_max"] + res / 2, res)
    return lats, lons


def regrid_to_target(da: xr.DataArray, target_lat: np.ndarray, target_lon: np.ndarray,
                      lat_name: str = "latitude", lon_name: str = "longitude") -> xr.DataArray:
    """
    Bilinear interpolation of a DataArray onto the target lat/lon grid.
    Falls back to nearest-neighbour if interpolation fails.
    """
    rename = {}
    if lat_name in da.dims and lat_name != "latitude":
        rename[lat_name] = "latitude"
    if lon_name in da.dims and lon_name != "longitude":
        rename[lon_name] = "longitude"
    if rename:
        da = da.rename(rename)

    # Ensure coordinate is sorted ascending
    if da["latitude"][0] > da["latitude"][-1]:
        da = da.isel(latitude=slice(None, None, -1))
    if da["longitude"][0] > da["longitude"][-1]:
        da = da.isel(longitude=slice(None, None, -1))

    try:
        da_out = da.interp(latitude=target_lat, longitude=target_lon, method="linear",
                           kwargs={"fill_value": None})
    except Exception:
        da_out = da.interp(latitude=target_lat, longitude=target_lon, method="nearest")
    return da_out


def shift_lon_0to360_to_180(da: xr.DataArray, lon_name: str = "lon") -> xr.DataArray:
    """Convert 0–360 longitudes to -180–180 and sort."""
    lon = da[lon_name].values.copy()
    lon = np.where(lon > 180, lon - 360, lon)
    da = da.assign_coords({lon_name: lon})
    da = da.sortby(lon_name)
    return da


# ─── per-dataset loaders ──────────────────────────────────────────────────────

def load_sst(cfg, target_lat, target_lon, date: str) -> Optional[xr.DataArray]:
    c = cfg["data"]["sst"]
    path = ROOT / c["path"]
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        da = ds[c["variable"]].sel({c["time_dim"]: date}, method="nearest")
        da = da.squeeze()
        if c.get("kelvin_check") and float(da.mean()) > 100:
            da = da - 273.15
        da = regrid_to_target(da, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
    return da.rename("sst")


def load_sss(cfg, target_lat, target_lon, date: str) -> Optional[xr.DataArray]:
    c = cfg["data"]["sss"]
    path = ROOT / c["path"]
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        da = ds[c["variable"]].sel({c["time_dim"]: date}, method="nearest")
        da = da.squeeze()
        da = regrid_to_target(da, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
    return da.rename("sss")


def load_ssh(cfg, target_lat, target_lon, date: str) -> Optional[xr.DataArray]:
    c = cfg["data"]["ssh"]
    path = ROOT / c["path"]
    if not path.exists():
        return None
    with xr.open_dataset(path) as ds:
        da = ds[c["variable"]].sel({c["time_dim"]: date}, method="nearest")
        da = da.squeeze()
        da = regrid_to_target(da, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
    return da.rename("sla")


def load_currents(cfg, target_lat, target_lon, date: str):
    c = cfg["data"]["currents"]
    path = ROOT / c["path"]
    if not path.exists():
        return None, None
    with xr.open_dataset(path) as ds:
        u = ds[c["var_u"]].sel({c["time_dim"]: date}, method="nearest").squeeze()
        v = ds[c["var_v"]].sel({c["time_dim"]: date}, method="nearest").squeeze()
        u = regrid_to_target(u, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
        v = regrid_to_target(v, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
    return u.rename("u_cur"), v.rename("v_cur")


def load_winds(cfg, target_lat, target_lon, date: str):
    """Load CCMP 6-hourly winds, shift lon 0–360→-180–180, crop NIO, average to daily."""
    c = cfg["data"]["winds"]
    path = ROOT / c["path"]
    if not path.exists():
        return None, None
    d = cfg["domain"]
    with xr.open_dataset(path) as ds:
        # Shift longitude
        ds = ds.assign_coords({c["lon_dim"]: (
            (ds[c["lon_dim"]] + 180) % 360 - 180
        )}).sortby(c["lon_dim"])
        # Crop to NIO
        ds = ds.sel({
            c["lat_dim"]: slice(d["lat_min"], d["lat_max"]),
            c["lon_dim"]: slice(d["lon_min"], d["lon_max"]),
        })
        # Select this date's 6-hourly records and average to daily
        try:
            u_day = ds[c["var_u"]].sel({c["time_dim"]: date}, method="nearest")
        except Exception:
            # time coordinate might be cftime; fall back to groupby date
            u_day = ds[c["var_u"]].groupby(f"{c['time_dim']}.date").mean()[date]
        try:
            v_day = ds[c["var_v"]].sel({c["time_dim"]: date}, method="nearest")
        except Exception:
            v_day = ds[c["var_v"]].groupby(f"{c['time_dim']}.date").mean()[date]

        # If there are multiple timesteps for the day (6-hourly), average them
        if u_day.ndim > 2:
            u_day = u_day.mean(dim=c["time_dim"], keep_attrs=True)
            v_day = v_day.mean(dim=c["time_dim"], keep_attrs=True)

        u_day = u_day.squeeze()
        v_day = v_day.squeeze()

        u_day = regrid_to_target(u_day, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
        v_day = regrid_to_target(v_day, target_lat, target_lon, c["lat_dim"], c["lon_dim"])

    return u_day.rename("u_wind"), v_day.rename("v_wind")


def build_latlon_channels(target_lat, target_lon) -> tuple:
    """Return (lat_channel, lon_channel) normalized to -1..1."""
    lat_norm = (target_lat - target_lat.mean()) / (target_lat.max() - target_lat.min() + 1e-6) * 2
    lon_norm = (target_lon - target_lon.mean()) / (target_lon.max() - target_lon.min() + 1e-6) * 2
    lat_2d = np.tile(lat_norm[:, None], (1, len(target_lon)))
    lon_2d = np.tile(lon_norm[None, :], (len(target_lat), 1))
    return lat_2d, lon_2d


def build_land_mask(cfg, target_lat, target_lon) -> np.ndarray:
    """
    Land mask: True = ocean (valid), False = land (NaN in GLORYS surface).
    Uses GLORYS thetao at depth=0 at first available time.
    """
    c = cfg["data"]["glorys"]
    path = ROOT / c["path"]
    if not path.exists():
        # Return all-ocean mask if GLORYS not available
        return np.ones((len(target_lat), len(target_lon)), dtype=bool)
    with xr.open_dataset(path) as ds:
        da = ds[c["variable"]].isel(time=0)
        # Pick shallowest depth
        if c["depth_dim"] in da.dims:
            da = da.isel(**{c["depth_dim"]: 0})
        da = da.squeeze()
        da = regrid_to_target(da, target_lat, target_lon, c["lat_dim"], c["lon_dim"])
        mask = ~np.isnan(da.values)
    return mask


# ─── normalisation ────────────────────────────────────────────────────────────

CHANNEL_NAMES = ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind", "lat", "lon"]


def compute_norm_stats(all_days_data: list, land_mask: np.ndarray) -> dict:
    """
    Compute per-channel mean and std over all available days.
    all_days_data: list of dicts {channel_name: 2D np.array}
    Returns: {channel: {mean, std}}
    """
    stats = {}
    for ch in CHANNEL_NAMES:
        values = []
        for day in all_days_data:
            arr = day.get(ch)
            if arr is not None:
                vals = arr[land_mask].ravel()
                vals = vals[~np.isnan(vals)]
                values.append(vals)
        if values:
            all_vals = np.concatenate(values)
            stats[ch] = {"mean": float(np.mean(all_vals)), "std": float(np.std(all_vals)) or 1.0}
        else:
            stats[ch] = {"mean": 0.0, "std": 1.0}
    return stats


def normalize_array(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    return (arr - mean) / (std + 1e-6)


# ─── main ─────────────────────────────────────────────────────────────────────

def get_available_dates(cfg) -> list:
    """Get dates that are plausibly available (5-day prototype window)."""
    start = datetime(2020, 1, 1)
    end = datetime(2020, 1, 5)
    dates = []
    while start <= end:
        dates.append(start.strftime("%Y-%m-%d"))
        start += timedelta(days=1)
    return dates


def process_one_day(cfg, target_lat, target_lon, land_mask, norm_stats, date: str, out_dir: Path):
    """
    Load all channels for one day, normalise, write inputs_YYYYMMDD.nc.
    Returns raw (unnormalised) data dict for norm_stats computation.
    """
    raw = {}

    sst = load_sst(cfg, target_lat, target_lon, date)
    sss = load_sss(cfg, target_lat, target_lon, date)
    ssh = load_ssh(cfg, target_lat, target_lon, date)
    u_cur, v_cur = load_currents(cfg, target_lat, target_lon, date)
    u_wind, v_wind = load_winds(cfg, target_lat, target_lon, date)

    lat_ch, lon_ch = build_latlon_channels(target_lat, target_lon)

    raw["sst"] = sst.values if sst is not None else np.full((len(target_lat), len(target_lon)), np.nan)
    raw["sss"] = sss.values if sss is not None else np.full_like(raw["sst"], np.nan)
    raw["sla"] = ssh.values if ssh is not None else np.full_like(raw["sst"], np.nan)
    raw["u_cur"] = u_cur.values if u_cur is not None else np.full_like(raw["sst"], np.nan)
    raw["v_cur"] = v_cur.values if v_cur is not None else np.full_like(raw["sst"], np.nan)
    raw["u_wind"] = u_wind.values if u_wind is not None else np.full_like(raw["sst"], np.nan)
    raw["v_wind"] = v_wind.values if v_wind is not None else np.full_like(raw["sst"], np.nan)
    raw["lat"] = lat_ch
    raw["lon"] = lon_ch

    # Apply land mask (set land to NaN; lat/lon channels keep values)
    for ch in ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind"]:
        raw[ch][~land_mask] = np.nan

    if norm_stats is None:
        return raw  # first pass: just return raw for stats computation

    # Normalise and write NetCDF
    norm_arrs = {}
    for ch in CHANNEL_NAMES:
        arr = raw[ch].copy()
        m = norm_stats[ch]["mean"]
        s = norm_stats[ch]["std"]
        arr_norm = normalize_array(arr, m, s)
        if ch not in ("lat", "lon"):
            arr_norm[~land_mask] = np.nan
        norm_arrs[ch] = arr_norm

    ds_out = xr.Dataset(
        {ch: xr.DataArray(norm_arrs[ch], dims=["latitude", "longitude"],
                          coords={"latitude": target_lat, "longitude": target_lon})
         for ch in CHANNEL_NAMES}
    )
    ds_out.attrs["date"] = date
    ds_out.attrs["normalized"] = "True"
    ds_out.attrs["norm_stats_path"] = str(ROOT / cfg["processed"]["norm_stats"])
    ds_out.attrs["land_mask_note"] = "NaN = land"

    date_str = date.replace("-", "")
    out_path = out_dir / f"inputs_{date_str}.nc"
    ds_out.to_netcdf(out_path)
    print(f"  ✓ wrote {out_path}")
    return raw


def main(dates=None):
    cfg = load_config()
    target_lat, target_lon = build_target_grid(cfg)
    out_dir = ROOT / cfg["processed"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    all_dates = dates or get_available_dates(cfg)
    print(f"Processing dates: {all_dates}")

    # Build land mask once
    print("\nBuilding land mask from GLORYS...")
    land_mask = build_land_mask(cfg, target_lat, target_lon)
    np.save(out_dir / "land_mask.npy", land_mask)
    print(f"  Ocean cells: {land_mask.sum()} / {land_mask.size}")

    # First pass: collect raw data for norm stats
    print("\nFirst pass: loading raw data for normalisation stats...")
    all_raw = []
    for date in all_dates:
        print(f"  Loading {date}...")
        raw = process_one_day(cfg, target_lat, target_lon, land_mask, norm_stats=None, date=date, out_dir=out_dir)
        all_raw.append(raw)

    # Compute normalisation statistics
    print("\nComputing normalisation statistics...")
    norm_stats = compute_norm_stats(all_raw, land_mask)
    norm_path = ROOT / cfg["processed"]["norm_stats"]
    with open(norm_path, "w") as f:
        json.dump(norm_stats, f, indent=2)
    print(f"  ✓ wrote {norm_path}")
    for ch, s in norm_stats.items():
        print(f"    {ch:10s}: mean={s['mean']:.4f}  std={s['std']:.4f}")

    # Second pass: write normalised NetCDF files
    print("\nSecond pass: writing normalised inputs...")
    for date in all_dates:
        print(f"  Processing {date}...")
        process_one_day(cfg, target_lat, target_lon, land_mask, norm_stats, date, out_dir)

    print("\n✓ harmonize.py complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dates", nargs="+", help="Specific dates to process (YYYY-MM-DD)")
    args = parser.parse_args()
    main(dates=args.dates)
