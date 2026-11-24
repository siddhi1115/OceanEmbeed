"""
baseline.py
-----------
Simple per-depth linear regression baseline for OceanEmbed.
Uses SST (channel 0 of the processed inputs) as the only predictor.

Provides a comparison target in the validation UI:
  "Can the neural network outperform a simple linear regression?"

Saves:  weights/baseline.pkl
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"
CHANNEL_NAMES = ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind", "lat", "lon"]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class DepthwiseLinearBaseline:
    """
    Per-depth, pixel-wise linear regression:
        T_pred(d, lat, lon) = a(d) * SST(lat, lon) + b(d)

    'a' and 'b' are scalar per depth (not per pixel) for simplicity.
    Fit on all available training days.
    """
    def __init__(self):
        self.slopes: Dict[int, float] = {}  # depth_idx → slope
        self.intercepts: Dict[int, float] = {}  # depth_idx → intercept
        self.fitted = False

    def fit(self, sst_flat: np.ndarray, targets_flat: Dict[int, np.ndarray]):
        """
        sst_flat : (N,) flattened SST values over ocean pixels and all days
        targets_flat : {depth_idx: (N,) flattened target temperatures}
        """
        for d_idx, temps in targets_flat.items():
            valid = ~np.isnan(temps) & ~np.isnan(sst_flat)
            if valid.sum() < 2:
                self.slopes[d_idx] = 0.0
                self.intercepts[d_idx] = float(np.nanmean(temps))
                continue
            # Ordinary least squares
            x = sst_flat[valid]
            y = temps[valid]
            x_mean, y_mean = x.mean(), y.mean()
            slope = np.sum((x - x_mean) * (y - y_mean)) / (np.sum((x - x_mean) ** 2) + 1e-9)
            intercept = y_mean - slope * x_mean
            self.slopes[d_idx] = float(slope)
            self.intercepts[d_idx] = float(intercept)
        self.fitted = True

    def predict(self, sst: np.ndarray, n_depths: int = 15) -> np.ndarray:
        """
        sst : (H, W) unnormalised SST in °C
        Returns : (n_depths, H, W) predicted temperatures
        """
        H, W = sst.shape
        out = np.zeros((n_depths, H, W), dtype=np.float32)
        for d_idx in range(n_depths):
            slope = self.slopes.get(d_idx, 0.0)
            intercept = self.intercepts.get(d_idx, 20.0)
            out[d_idx] = slope * sst + intercept
        return out

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"  ✓ Baseline saved to {path}")

    @staticmethod
    def load(path: Path) -> "DepthwiseLinearBaseline":
        with open(path, "rb") as f:
            return pickle.load(f)


def train_baseline(cfg: dict) -> DepthwiseLinearBaseline:
    """
    Load all processed days, extract SST and GLORYS targets,
    fit the baseline, and save to weights/baseline.pkl.
    """
    proc_dir = ROOT / cfg["processed"]["dir"]
    n_depths = cfg["model"]["n_depths"]

    # Load normalisation stats to recover real SST from normalised input
    norm_path = ROOT / cfg["processed"]["norm_stats"]
    with open(norm_path) as f:
        norm_stats = json.load(f)
    sst_mean = norm_stats["sst"]["mean"]
    sst_std = norm_stats["sst"]["std"]

    all_sst = []
    all_targets = {d: [] for d in range(n_depths)}

    input_files = sorted(proc_dir.glob("inputs_????????.nc"))
    for inp_path in input_files:
        date_str = inp_path.stem.replace("inputs_", "")
        tgt_path = proc_dir / f"targets_{date_str}.nc"
        if not tgt_path.exists():
            continue

        with xr.open_dataset(inp_path) as ds:
            sst_norm = ds["sst"].values.astype(float)
            # Denormalise
            sst_real = sst_norm * sst_std + sst_mean

        with xr.open_dataset(tgt_path) as ds:
            tgt = ds["thetao"].values.astype(float)  # (15, H, W)

        land_mask = ~np.isnan(sst_real)
        sst_flat = sst_real[land_mask].ravel()
        all_sst.append(sst_flat)
        for d_idx in range(n_depths):
            tgt_flat = tgt[d_idx][land_mask].ravel()
            all_targets[d_idx].append(tgt_flat)

    if not all_sst:
        raise FileNotFoundError("No processed files found. Run the pipeline first.")

    sst_all = np.concatenate(all_sst)
    targets_all = {d: np.concatenate(v) for d, v in all_targets.items()}

    baseline = DepthwiseLinearBaseline()
    baseline.fit(sst_all, targets_all)
    return baseline


if __name__ == "__main__":
    import sys
    cfg = load_config()
    print("Training depth-wise linear regression baseline...")
    baseline = train_baseline(cfg)
    out_path = ROOT / cfg["training"]["baseline_checkpoint"]
    baseline.save(out_path)
    print("\nBaseline slopes (depth_idx: slope):")
    for d, s in baseline.slopes.items():
        print(f"  depth {d:2d}: slope={s:.4f}  intercept={baseline.intercepts[d]:.2f}")
    print("\n✓ baseline.py complete.")
