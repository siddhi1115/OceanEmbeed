"""
evaluate.py
-----------
Runs the trained OceanEmbed model (and baseline) on all 5 days,
computes metrics vs GLORYS (and vs ARGO where available),
and writes backend/metrics.json.

Honesty guarantees:
  - split is always "in-sample" (5 days = no held-out test set)
  - data_period always from config
  - metrics.json is the ONLY source of truth for the frontend

Usage:
  python ml/evaluate.py
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import xarray as xr
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.oceanembed import build_model, model_info
from ml.baseline import DepthwiseLinearBaseline


def load_config():
    with open(ROOT / "pipeline" / "config.yaml") as f:
        return yaml.safe_load(f)


def load_model(cfg, checkpoint_path: Path):
    model = build_model(cfg)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found at {checkpoint_path}. "
            "Run ml/train.py first."
        )
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model


def run_inference(model, inp_path: Path) -> np.ndarray:
    """Run model on a full-grid input file. Returns (15, H, W)."""
    from ml.dataset import CHANNEL_NAMES
    with xr.open_dataset(inp_path) as ds:
        arrays = []
        for ch in CHANNEL_NAMES:
            arr = ds[ch].values.astype(np.float32)
            arr = np.where(np.isnan(arr), 0.0, arr)
            arrays.append(arr)
        inputs = np.stack(arrays, axis=0)  # (9, H, W)

    x = torch.from_numpy(inputs).unsqueeze(0)  # (1, 9, H, W)
    with torch.no_grad():
        pred = model(x)
    return pred.squeeze(0).numpy()  # (15, H, W)


def run_baseline_inference(baseline: DepthwiseLinearBaseline, inp_path: Path,
                           norm_stats: dict) -> np.ndarray:
    """Run linear baseline on a full-grid. Needs denormalised SST."""
    with xr.open_dataset(inp_path) as ds:
        sst_norm = ds["sst"].values.astype(float)
    sst_real = sst_norm * norm_stats["sst"]["std"] + norm_stats["sst"]["mean"]
    return baseline.predict(sst_real, n_depths=15)


def compute_metrics_per_depth(
    pred: np.ndarray, target: np.ndarray, land_mask: np.ndarray,
    target_depths: list,
) -> List[Dict]:
    """
    pred, target: (n_days, 15, H, W)
    Returns list of dicts, one per depth.
    """
    n_days, n_depths, H, W = pred.shape
    results = []
    for d_idx, depth in enumerate(target_depths):
        p = pred[:, d_idx, :, :].ravel()
        t = target[:, d_idx, :, :].ravel()
        mask = land_mask.ravel().astype(bool)
        p = p[mask]
        t = t[mask]
        valid = ~(np.isnan(p) | np.isnan(t))
        p = p[valid]
        t = t[valid]
        n = len(p)

        if n < 2:
            results.append({"depth_m": depth, "n_valid": n,
                            "rmse": None, "mae": None, "bias": None, "r2": None, "corr": None})
            continue

        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        mae = float(np.mean(np.abs(p - t)))
        bias = float(np.mean(p - t))
        ss_res = np.sum((p - t) ** 2)
        ss_tot = np.sum((t - t.mean()) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-9))
        corr = float(np.corrcoef(p, t)[0, 1]) if n > 1 else None

        results.append({
            "depth_m": depth,
            "n_valid": n,
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "bias": round(bias, 4),
            "r2": round(r2, 4),
            "corr": round(corr, 4) if corr is not None else None,
        })
    return results


def compute_region_metrics(pred, target, land_mask, lat, lon, regions, target_depths):
    """Compute RMSE/MAE per region (Arabian Sea, Bay of Bengal)."""
    results = {}
    for region_name, bounds in regions.items():
        lat_mask = (lat >= bounds["lat_min"]) & (lat <= bounds["lat_max"])
        lon_mask = (lon >= bounds["lon_min"]) & (lon <= bounds["lon_max"])
        # 2D spatial mask
        spatial_mask = np.outer(lat_mask, lon_mask) & land_mask

        all_pred, all_tgt = [], []
        for d_idx in range(len(target_depths)):
            p = pred[:, d_idx, :, :][:, spatial_mask]
            t = target[:, d_idx, :, :][:, spatial_mask]
            valid = ~(np.isnan(p) | np.isnan(t))
            all_pred.append(p[valid])
            all_tgt.append(t[valid])

        p_all = np.concatenate(all_pred)
        t_all = np.concatenate(all_tgt)
        if len(p_all) < 2:
            results[region_name] = {"rmse": None, "mae": None, "bias": None, "n": 0}
            continue

        results[region_name] = {
            "rmse": round(float(np.sqrt(np.mean((p_all - t_all) ** 2))), 4),
            "mae": round(float(np.mean(np.abs(p_all - t_all))), 4),
            "bias": round(float(np.mean(p_all - t_all)), 4),
            "n": int(len(p_all)),
        }
    return results


def compute_argo_matchups(pred_all: Dict, target_depths: list, argo_matchups_path: Path) -> list:
    """
    For each ARGO profile in matchups, extract the nearest grid cell prediction
    and GLORYS value and return a comparison list.
    """
    if not argo_matchups_path.exists():
        return []

    with open(argo_matchups_path) as f:
        argo = json.load(f)

    matchup_results = []
    for profile in argo.get("profiles", []):
        date = profile.get("date", "2020-01-01")
        date_str = date.replace("-", "")
        gi = profile.get("grid_i")
        gj = profile.get("grid_j")

        if gi is None or gj is None:
            continue

        pred_map = pred_all.get(date_str)
        tgt_map = pred_all.get(f"glorys_{date_str}")
        if pred_map is None:
            continue

        profile_pred = []
        profile_tgt = []
        profile_argo = []
        for d_idx, depth in enumerate(target_depths):
            depth_key = f"{int(depth)}m"
            argo_temp = profile["temp_at_15depths"].get(depth_key)
            pred_temp = float(pred_map[d_idx, gi, gj]) if pred_map is not None else None
            tgt_temp = float(tgt_map[d_idx, gi, gj]) if tgt_map is not None else None
            profile_pred.append(pred_temp)
            profile_tgt.append(tgt_temp)
            profile_argo.append(argo_temp)

        matchup_results.append({
            "date": date,
            "lat": profile["grid_lat"],
            "lon": profile["grid_lon"],
            "depths_m": target_depths,
            "predicted_C": [round(v, 3) if v is not None and not np.isnan(v) else None for v in profile_pred],
            "glorys_C": [round(v, 3) if v is not None and not np.isnan(v) else None for v in profile_tgt],
            "argo_C": [round(v, 3) if v is not None else None for v in profile_argo],
        })

    return matchup_results


def main():
    cfg = load_config()
    ec = cfg["evaluation"]
    target_depths = cfg["target_depths"]
    proc_dir = ROOT / cfg["processed"]["dir"]
    n_depths = len(target_depths)

    # Load land mask
    mask_path = proc_dir / "land_mask.npy"
    if mask_path.exists():
        land_mask = np.load(mask_path).astype(bool)
    else:
        print("Warning: land_mask.npy not found. Using all-ocean mask.")
        land_mask = np.ones((101, 241), dtype=bool)

    # Build target grid
    res = cfg["target_grid"]["resolution"]
    d = cfg["domain"]
    target_lat = np.arange(d["lat_min"], d["lat_max"] + res / 2, res)
    target_lon = np.arange(d["lon_min"], d["lon_max"] + res / 2, res)

    # Load model
    ckpt_path = ROOT / cfg["training"]["checkpoint"]
    print(f"Loading model from {ckpt_path}...")
    try:
        model = load_model(cfg, ckpt_path)
        info = model_info(model)
        print(f"  Parameters: {info['total_parameters']:,}")
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Load baseline
    baseline_path = ROOT / cfg["training"]["baseline_checkpoint"]
    baseline = None
    if baseline_path.exists():
        try:
            baseline = DepthwiseLinearBaseline.load(baseline_path)
            print(f"  Baseline loaded from {baseline_path}")
        except Exception as e:
            print(f"  Warning: could not load baseline: {e}")

    # Load norm stats
    norm_path = ROOT / cfg["processed"]["norm_stats"]
    with open(norm_path) as f:
        norm_stats = json.load(f)

    # Discover available processed days
    input_files = sorted(proc_dir.glob("inputs_????????.nc"))
    dates_avail = [f.stem.replace("inputs_", "") for f in input_files]
    print(f"\nEvaluating on {len(dates_avail)} days: {dates_avail}")

    pred_all = {}       # date_str → (15, H, W)
    baseline_all = {}
    target_all = {}     # date_str → (15, H, W)

    for date_str in dates_avail:
        inp_path = proc_dir / f"inputs_{date_str}.nc"
        tgt_path = proc_dir / f"targets_{date_str}.nc"

        # Model prediction
        print(f"  Inference {date_str}...", end=" ")
        pred = run_inference(model, inp_path)
        pred_all[date_str] = pred
        print(f"pred range: {np.nanmin(pred):.2f}–{np.nanmax(pred):.2f}°C")

        # Baseline prediction
        if baseline is not None:
            bl_pred = run_baseline_inference(baseline, inp_path, norm_stats)
            baseline_all[date_str] = bl_pred

        # GLORYS target
        if tgt_path.exists():
            with xr.open_dataset(tgt_path) as ds:
                tgt = ds["thetao"].values.astype(np.float32)
            target_all[date_str] = tgt
            pred_all[f"glorys_{date_str}"] = tgt  # for ARGO matchup lookup

    if not target_all:
        print("WARNING: No target files found. Metrics cannot be computed.")
        print("Run pipeline/targets.py first.")
        metrics = {
            "data_period": ec["data_period"],
            "n_samples": len(dates_avail),
            "split": ec["split_label"],
            "warning": "No GLORYS target files available; metrics not computed.",
            "per_depth": [],
            "per_depth_baseline": [],
            "by_region": {},
            "by_region_baseline": {},
            "argo_matchups": [],
            "model_info": info,
        }
    else:
        # Stack arrays: (n_days, 15, H, W)
        days = sorted(target_all.keys())
        pred_stack = np.stack([pred_all[d] for d in days], axis=0)
        tgt_stack = np.stack([target_all[d] for d in days], axis=0)

        print("\nComputing metrics vs GLORYS...")
        per_depth = compute_metrics_per_depth(pred_stack, tgt_stack, land_mask, target_depths)

        per_depth_baseline = []
        if baseline_all:
            bl_stack = np.stack([baseline_all[d] for d in days], axis=0)
            per_depth_baseline = compute_metrics_per_depth(bl_stack, tgt_stack, land_mask, target_depths)

        by_region = compute_region_metrics(
            pred_stack, tgt_stack, land_mask, target_lat, target_lon,
            cfg["regions"], target_depths
        )
        by_region_baseline = {}
        if baseline_all:
            by_region_baseline = compute_region_metrics(
                bl_stack, tgt_stack, land_mask, target_lat, target_lon,
                cfg["regions"], target_depths
            )

        # ARGO matchups (validation only)
        argo_matchups_path = ROOT / cfg["processed"]["argo_matchups"]
        print("Computing ARGO matchups...")
        argo_matchups = compute_argo_matchups(pred_all, target_depths, argo_matchups_path)
        print(f"  {len(argo_matchups)} ARGO profiles matched")

        # Print summary table
        print("\n── Metrics per depth (model vs GLORYS, in-sample) ──")
        print(f"{'Depth':>8} {'RMSE':>7} {'MAE':>7} {'Bias':>8} {'R²':>7} {'Corr':>7}")
        for m in per_depth:
            d = m['depth_m']
            rmse = f"{m['rmse']:.3f}" if m['rmse'] is not None else "N/A"
            mae = f"{m['mae']:.3f}" if m['mae'] is not None else "N/A"
            bias = f"{m['bias']:.3f}" if m['bias'] is not None else "N/A"
            r2 = f"{m['r2']:.3f}" if m['r2'] is not None else "N/A"
            corr = f"{m['corr']:.3f}" if m['corr'] is not None else "N/A"
            print(f"{d:>8}m {rmse:>7} {mae:>7} {bias:>8} {r2:>7} {corr:>7}")

        metrics = {
            "data_period": ec["data_period"],
            "n_samples": len(days),
            "split": ec["split_label"],
            "disclaimer": (
                "All metrics are IN-SAMPLE (training days = evaluation days). "
                "This prototype uses only 5 days of data. "
                "These metrics demonstrate the pipeline, not operational skill."
            ),
            "dates_evaluated": days,
            "per_depth": per_depth,
            "per_depth_baseline": per_depth_baseline,
            "by_region": by_region,
            "by_region_baseline": by_region_baseline,
            "argo_matchups": argo_matchups,
            "model_info": info,
            "training_checkpoint": str(ckpt_path),
        }

    # Write metrics.json
    out_path = ROOT / ec["metrics_output"]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2, default=str)

    print(f"\n✓ Metrics written to {out_path}")
    print(f"  split = '{metrics['split']}'")
    print(f"  n_samples = {metrics['n_samples']}")
    print("\n✓ evaluate.py complete.")


if __name__ == "__main__":
    main()
