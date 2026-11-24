"""
argo_profiles.py
----------------
Processes ARGO_Profiles_20200101.nc:
  - Groups point-row observations into profiles by (time, lat, lon)
  - Interpolates each profile from PRES(dbar) ≈ depth(m) to 15 target depths
  - Snaps to nearest 0.25° grid cell
  - Writes data/processed/argo_matchups.json

Also does a sanity check against INCOIS_ARGO (monthly, coarse) — not used
in training or quantitative evaluation.

Run:  python pipeline/argo_profiles.py
"""

import json
import sys
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


def snap_to_grid(lat_val: float, lon_val: float,
                 target_lat: np.ndarray, target_lon: np.ndarray) -> tuple:
    """Return nearest grid indices (i, j) and snapped (lat, lon)."""
    i = int(np.argmin(np.abs(target_lat - lat_val)))
    j = int(np.argmin(np.abs(target_lon - lon_val)))
    return i, j, float(target_lat[i]), float(target_lon[j])


def interp_profile(pres: np.ndarray, temp: np.ndarray, tgt_depths: np.ndarray) -> np.ndarray:
    """
    Interpolate a vertical profile to target depths.
    pres(dbar) ≈ depth(m) for upper ocean.
    Returns NaN array if fewer than 2 valid points.
    """
    valid = ~(np.isnan(pres) | np.isnan(temp))
    if valid.sum() < 2:
        return np.full(len(tgt_depths), np.nan)
    pres_v = pres[valid]
    temp_v = temp[valid]
    # Sort by depth ascending
    sort_idx = np.argsort(pres_v)
    pres_v = pres_v[sort_idx]
    temp_v = temp_v[sort_idx]
    f = interp1d(pres_v, temp_v, kind="linear", bounds_error=False,
                 fill_value=(temp_v[0], temp_v[-1]))
    return f(tgt_depths).astype(np.float32)


def main():
    cfg = load_config()
    c = cfg["data"]["argo_obs"]
    path = ROOT / c["path"]
    tgt_depths = np.array(cfg["target_depths"], dtype=float)

    res = cfg["target_grid"]["resolution"]
    d = cfg["domain"]
    target_lat = np.arange(d["lat_min"], d["lat_max"] + res / 2, res)
    target_lon = np.arange(d["lon_min"], d["lon_max"] + res / 2, res)

    out_path = ROOT / cfg["processed"]["argo_matchups"]
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        print(f"WARNING: ARGO file not found at {path}")
        # Write empty matchups file so backend doesn't crash
        with open(out_path, "w") as f:
            json.dump({"note": "ARGO file not found", "profiles": []}, f)
        print(f"  Wrote empty matchups to {out_path}")
        return

    print(f"Loading ARGO profiles from {path}...")
    with xr.open_dataset(path) as ds:
        print(f"  Variables: {list(ds.data_vars)}")
        print(f"  Dims: {dict(ds.dims)}")

        lats = ds[c["lat_dim"]].values
        lons = ds[c["lon_dim"]].values
        pres_all = ds[c["var_pres"]].values.astype(float)
        temp_all = ds[c["var_temp"]].values.astype(float)

        # Try to get time values
        try:
            times = ds[c["time_dim"]].values
        except Exception:
            times = np.array(["2020-01-01"] * len(lats))

    print(f"  Total observation rows: {len(lats)}")

    # Crop to NIO
    nio_mask = (
        (lats >= d["lat_min"]) & (lats <= d["lat_max"]) &
        (lons >= d["lon_min"]) & (lons <= d["lon_max"])
    )
    lats = lats[nio_mask]
    lons = lons[nio_mask]
    pres_all = pres_all[nio_mask]
    temp_all = temp_all[nio_mask]
    times = times[nio_mask]
    print(f"  NIO observations: {nio_mask.sum()}")

    # Group into profiles by (time, lat, lon) with rounding tolerance 0.1°
    profiles = {}
    for i in range(len(lats)):
        lat_r = round(float(lats[i]), 2)
        lon_r = round(float(lons[i]), 2)
        try:
            t_str = str(times[i])[:10]  # YYYY-MM-DD
        except Exception:
            t_str = "2020-01-01"
        key = (t_str, lat_r, lon_r)
        if key not in profiles:
            profiles[key] = {"pres": [], "temp": []}
        profiles[key]["pres"].append(float(pres_all[i]))
        profiles[key]["temp"].append(float(temp_all[i]))

    print(f"  Unique profiles: {len(profiles)}")

    matchups = []
    for (date, lat_val, lon_val), data in profiles.items():
        pres_arr = np.array(data["pres"])
        temp_arr = np.array(data["temp"])
        interp_temps = interp_profile(pres_arr, temp_arr, tgt_depths)

        gi, gj, snap_lat, snap_lon = snap_to_grid(lat_val, lon_val, target_lat, target_lon)

        matchup = {
            "date": date,
            "original_lat": lat_val,
            "original_lon": lon_val,
            "grid_lat": snap_lat,
            "grid_lon": snap_lon,
            "grid_i": gi,
            "grid_j": gj,
            "n_obs": len(pres_arr),
            "pres_range": [float(np.nanmin(pres_arr)), float(np.nanmax(pres_arr))],
            "temp_at_15depths": {
                f"{int(dep)}m": (float(t) if not np.isnan(t) else None)
                for dep, t in zip(tgt_depths, interp_temps)
            },
        }
        matchups.append(matchup)

    result = {
        "note": "ARGO profiles interpolated to 15 target depths. Used for validation only, NOT training.",
        "n_profiles": len(matchups),
        "target_depths_m": tgt_depths.tolist(),
        "profiles": matchups,
    }

    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"\n✓ Wrote {len(matchups)} ARGO matchup profiles to {out_path}")

    # Quick sanity check against INCOIS_ARGO (monthly coarse grid)
    incois_path = ROOT / cfg["data"]["incois_argo"]["path"]
    if incois_path.exists():
        print("\nSanity check: INCOIS_ARGO (monthly average)...")
        with xr.open_dataset(incois_path) as ds2:
            temp_incois = ds2[cfg["data"]["incois_argo"]["variable"]]
            print(f"  INCOIS ARGO TEMP range: {float(temp_incois.min()):.2f} – {float(temp_incois.max()):.2f} °C")
            print("  (Used for coarse sanity check only. NOT used in training or evaluation.)")
    else:
        print(f"\nINOCIS_ARGO file not found at {incois_path} (optional sanity check skipped)")

    print("\n✓ argo_profiles.py complete.")


if __name__ == "__main__":
    main()
