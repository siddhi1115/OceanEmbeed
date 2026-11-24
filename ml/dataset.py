"""
dataset.py
----------
PyTorch Dataset for OceanEmbed.

Yields (input_patch, target_patch, land_mask_patch) tuples.
- Training mode: random 64×64 crops with optional flip augmentation
- Inference mode: returns full grid (no cropping)
"""

import json
import random
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr
import yaml


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "pipeline" / "config.yaml"

CHANNEL_NAMES = ["sst", "sss", "sla", "u_cur", "v_cur", "u_wind", "v_wind", "lat", "lon"]


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


class OceanDataset(Dataset):
    """
    Parameters
    ----------
    dates : list of "YYYY-MM-DD" strings
    patch_size : int — size of random crop (training). Use None for full grid.
    augment : bool — random horizontal/vertical flips
    n_patches_per_day : int — how many random patches to yield per day
    """
    def __init__(
        self,
        dates: Optional[List[str]] = None,
        patch_size: Optional[int] = 64,
        augment: bool = True,
        n_patches_per_day: int = 8,
    ):
        self.cfg = load_config()
        proc_dir = ROOT / self.cfg["processed"]["dir"]

        # Discover available dates from processed files
        if dates is None:
            input_files = sorted(proc_dir.glob("inputs_????????.nc"))
            self.dates = [f.stem.replace("inputs_", "") for f in input_files]
        else:
            self.dates = [d.replace("-", "") for d in dates]

        if not self.dates:
            raise FileNotFoundError(
                f"No processed input files found in {proc_dir}. "
                "Run pipeline/harmonize.py first."
            )

        self.proc_dir = proc_dir
        self.patch_size = patch_size
        self.augment = augment
        self.n_patches = n_patches_per_day
        self.full_mode = patch_size is None

        # Load land mask
        mask_path = proc_dir / "land_mask.npy"
        if mask_path.exists():
            self.land_mask = np.load(mask_path).astype(np.float32)
        else:
            # Infer from first input file
            with xr.open_dataset(proc_dir / f"inputs_{self.dates[0]}.nc") as ds:
                ref = ds["sst"].values
            self.land_mask = (~np.isnan(ref)).astype(np.float32)

        # Build index: (date_str, random_seed) for deterministic crop per __getitem__
        if self.full_mode:
            self.index = [(d, 0) for d in self.dates]
        else:
            self.index = [(d, i) for d in self.dates for i in range(self.n_patches)]

    def __len__(self):
        return len(self.index)

    def _load_day(self, date_str: str) -> Tuple[np.ndarray, np.ndarray]:
        """Load input and target arrays for one day. Returns (inputs, targets)."""
        inp_path = self.proc_dir / f"inputs_{date_str}.nc"
        tgt_path = self.proc_dir / f"targets_{date_str}.nc"

        with xr.open_dataset(inp_path) as ds:
            # Stack channels: (C, H, W)
            arrays = []
            for ch in CHANNEL_NAMES:
                arr = ds[ch].values.astype(np.float32)
                # Replace NaN with 0 (land mask handles masking in loss)
                arr = np.where(np.isnan(arr), 0.0, arr)
                arrays.append(arr)
            inputs = np.stack(arrays, axis=0)  # (9, H, W)

        if tgt_path.exists():
            with xr.open_dataset(tgt_path) as ds:
                targets = ds["thetao"].values.astype(np.float32)  # (15, H, W)
        else:
            # Targets not yet available — return NaN array (loss will be 0)
            H, W = inputs.shape[1], inputs.shape[2]
            targets = np.full((15, H, W), np.nan, dtype=np.float32)

        return inputs, targets

    def _random_crop(self, inputs, targets, land_mask, seed):
        """Extract a random patch_size×patch_size crop containing at least some ocean."""
        ps = self.patch_size
        H, W = inputs.shape[1], inputs.shape[2]
        rng = random.Random(seed)

        # Try a few random locations, prefer ones with ocean pixels
        best = None
        for _ in range(20):
            i0 = rng.randint(0, H - ps)
            j0 = rng.randint(0, W - ps)
            lm_crop = land_mask[i0:i0 + ps, j0:j0 + ps]
            if lm_crop.mean() > 0.1:  # At least 10% ocean
                best = (i0, j0)
                break
        if best is None:
            best = (0, 0)
        i0, j0 = best

        inp_patch = inputs[:, i0:i0 + ps, j0:j0 + ps]
        tgt_patch = targets[:, i0:i0 + ps, j0:j0 + ps]
        lm_patch = land_mask[i0:i0 + ps, j0:j0 + ps]
        return inp_patch, tgt_patch, lm_patch

    def _augment(self, inp, tgt, lm, seed):
        """Random horizontal/vertical flips."""
        rng = random.Random(seed + 1000)
        if rng.random() > 0.5:
            inp = inp[:, :, ::-1].copy()
            tgt = tgt[:, :, ::-1].copy()
            lm = lm[:, ::-1].copy()
        if rng.random() > 0.5:
            inp = inp[:, ::-1, :].copy()
            tgt = tgt[:, ::-1, :].copy()
            lm = lm[::-1, :].copy()
        return inp, tgt, lm

    def __getitem__(self, idx):
        date_str, patch_seed = self.index[idx]
        inputs, targets = self._load_day(date_str)

        if self.full_mode:
            lm = self.land_mask
            # Replace NaN in targets with 0 for tensor but mask is provided separately
            tgt_safe = np.where(np.isnan(targets), 0.0, targets)
            return (
                torch.from_numpy(inputs),
                torch.from_numpy(tgt_safe),
                torch.from_numpy(lm),
            )

        # Random crop
        inp_p, tgt_p, lm_p = self._random_crop(inputs, targets, self.land_mask, patch_seed)

        # Augment
        if self.augment:
            inp_p, tgt_p, lm_p = self._augment(inp_p, tgt_p, lm_p, patch_seed)

        # Keep NaN in targets for masked loss, but convert inf/large values
        tgt_p = np.clip(tgt_p, -5, 40)  # sane ocean temp range

        return (
            torch.from_numpy(inp_p.copy()),
            torch.from_numpy(tgt_p.copy()),
            torch.from_numpy(lm_p.copy()),
        )
