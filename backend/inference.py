"""
inference.py
------------
Model loading and inference utilities for the FastAPI backend.
Handles:
  - Lazy-loading model and baseline (cached after first call)
  - Full-grid reconstruction per date
  - Normalisation/denormalisation
  - LRU cache for reconstructed grids (keyed by date+depth)
"""

import json
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.oceanembed import build_model
from ml.baseline import DepthwiseLinearBaseline
from ml.dataset import CHANNEL_NAMES

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


def load_config():
    with open(ROOT / "pipeline" / "config.yaml") as f:
        return yaml.safe_load(f)


# ─── Singleton loaders ────────────────────────────────────────────────────────

_model = None
_baseline = None
_norm_stats = None
_land_mask = None
_config = None


def get_config():
    global _config
    if _config is None:
        _config = load_config()
    return _config


def get_model():
    global _model
    if _model is None:
        cfg = get_config()
        ckpt_path = ROOT / cfg["training"]["checkpoint"]
        if not ckpt_path.exists():
            return None
        try:
            m = build_model(cfg)
            ckpt = torch.load(ckpt_path, map_location="cpu")
            m.load_state_dict(ckpt["model_state_dict"])
            m.eval()
            _model = m
        except Exception as e:
            print(f"[inference] Warning: could not load model: {e}")
            _model = None
    return _model


def get_baseline():
    global _baseline
    if _baseline is None:
        cfg = get_config()
        bl_path = ROOT / cfg["training"]["baseline_checkpoint"]
        if bl_path.exists():
            try:
                _baseline = DepthwiseLinearBaseline.load(bl_path)
            except Exception as e:
                print(f"[inference] Warning: could not load baseline: {e}")
    return _baseline


def get_norm_stats():
    global _norm_stats
    if _norm_stats is None:
        cfg = get_config()
        norm_path = ROOT / cfg["processed"]["norm_stats"]
        if norm_path.exists():
            with open(norm_path) as f:
                _norm_stats = json.load(f)
    return _norm_stats


def get_land_mask() -> Optional[np.ndarray]:
    global _land_mask
    if _land_mask is None:
        cfg = get_config()
        mask_path = ROOT / cfg["processed"]["dir"] / "land_mask.npy"
        if mask_path.exists():
            _land_mask = np.load(mask_path).astype(bool)
    return _land_mask


# ─── Grid helpers ─────────────────────────────────────────────────────────────

def build_grid_coords():
    cfg = get_config()
    res = cfg["target_grid"]["resolution"]
    d = cfg["domain"]
    lats = np.arange(d["lat_min"], d["lat_max"] + res / 2, res)
    lons = np.arange(d["lon_min"], d["lon_max"] + res / 2, res)
    return lats.tolist(), lons.tolist()


def get_available_dates() -> list:
    cfg = get_config()
    proc_dir = ROOT / cfg["processed"]["dir"]
    files = sorted(proc_dir.glob("inputs_????????.nc"))
    dates = []
    for f in files:
        date_str = f.stem.replace("inputs_", "")
        # Format as YYYY-MM-DD
        dates.append(f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}")
    return dates


# ─── Reconstruction cache ─────────────────────────────────────────────────────

_reconstruction_cache: Dict[str, np.ndarray] = {}  # date_str → (15, H, W)
_glorys_cache: Dict[str, np.ndarray] = {}           # date_str → (15, H, W)
_surface_cache: Dict[str, dict] = {}                # date_str → {channel: (H, W)}


def _date_to_str(date: str) -> str:
    return date.replace("-", "")


def reconstruct_full(date: str) -> Optional[np.ndarray]:
    """
    Run model inference on the full grid for a given date.
    Returns (15, H, W) array in °C, or None if model/data not available.
    Result is cached in memory.
    """
    date_str = _date_to_str(date)
    if date_str in _reconstruction_cache:
        return _reconstruction_cache[date_str]

    model = get_model()
    cfg = get_config()
    proc_dir = ROOT / cfg["processed"]["dir"]
    inp_path = proc_dir / f"inputs_{date_str}.nc"

    if not inp_path.exists():
        return None
    if model is None:
        # Fall back to baseline if model not available
        return reconstruct_baseline_full(date)

    with xr.open_dataset(inp_path) as ds:
        arrays = []
        for ch in CHANNEL_NAMES:
            arr = ds[ch].values.astype(np.float32)
            arr = np.where(np.isnan(arr), 0.0, arr)
            arrays.append(arr)
        inputs = np.stack(arrays, axis=0)

    x = torch.from_numpy(inputs).unsqueeze(0)
    with torch.no_grad():
        pred = model(x)
    result = pred.squeeze(0).numpy()  # (15, H, W)

    _reconstruction_cache[date_str] = result
    return result


def reconstruct_baseline_full(date: str) -> Optional[np.ndarray]:
    """Run linear baseline for a given date."""
    date_str = _date_to_str(date)
    baseline = get_baseline()
    norm_stats = get_norm_stats()
    if baseline is None or norm_stats is None:
        return None

    cfg = get_config()
    proc_dir = ROOT / cfg["processed"]["dir"]
    inp_path = proc_dir / f"inputs_{date_str}.nc"
    if not inp_path.exists():
        return None

    with xr.open_dataset(inp_path) as ds:
        sst_norm = ds["sst"].values.astype(float)
    sst_real = sst_norm * norm_stats["sst"]["std"] + norm_stats["sst"]["mean"]
    return baseline.predict(sst_real, n_depths=15)


def get_glorys_full(date: str) -> Optional[np.ndarray]:
    """Load GLORYS target for a given date. Returns (15, H, W) or None."""
    date_str = _date_to_str(date)
    if date_str in _glorys_cache:
        return _glorys_cache[date_str]

    cfg = get_config()
    proc_dir = ROOT / cfg["processed"]["dir"]
    tgt_path = proc_dir / f"targets_{date_str}.nc"
    if not tgt_path.exists():
        return None

    with xr.open_dataset(tgt_path) as ds:
        tgt = ds["thetao"].values.astype(np.float32)
    _glorys_cache[date_str] = tgt
    return tgt


def get_surface_variable(date: str, variable: str) -> Optional[np.ndarray]:
    """
    Load a raw surface variable for the given date and return as (H, W).
    variable: 'sst' | 'sss' | 'ssh' | 'u_cur' | 'v_cur' | 'u_wind' | 'v_wind'
    Returns denormalised values where possible.
    """
    date_str = _date_to_str(date)
    cache_key = f"{date_str}_{variable}"
    if cache_key in _surface_cache:
        return _surface_cache[cache_key]

    cfg = get_config()
    proc_dir = ROOT / cfg["processed"]["dir"]
    inp_path = proc_dir / f"inputs_{date_str}.nc"
    if not inp_path.exists():
        return None

    # Map UI variable names to channel names
    var_map = {
        "sst": "sst", "sss": "sss", "ssh": "sla", "sla": "sla",
        "u_cur": "u_cur", "v_cur": "v_cur",
        "u_wind": "u_wind", "v_wind": "v_wind",
    }
    ch = var_map.get(variable)
    if ch is None:
        return None

    norm_stats = get_norm_stats()
    with xr.open_dataset(inp_path) as ds:
        arr = ds[ch].values.astype(np.float32)

    # Denormalise if we have stats
    if norm_stats and ch in norm_stats:
        arr = arr * norm_stats[ch]["std"] + norm_stats[ch]["mean"]

    land_mask = get_land_mask()
    if land_mask is not None:
        arr[~land_mask] = np.nan

    _surface_cache[cache_key] = arr
    return arr
