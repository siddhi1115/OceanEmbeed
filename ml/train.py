"""
train.py
--------
Train the OceanEmbed model on the 5-day prototype dataset.

Usage:
  python ml/train.py                    # use config.yaml defaults
  python ml/train.py --epochs 5         # quick smoke test
  python ml/train.py --T 2 --epochs 10

Saves:
  weights/model.pt          — best checkpoint (lowest training loss)
  weights/training_log.json — per-epoch loss history
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset import OceanDataset
from ml.oceanembed import OceanEmbed, OceanEmbedLoss, build_model, model_info
from ml.baseline import train_baseline


def load_config():
    with open(ROOT / "pipeline" / "config.yaml") as f:
        return yaml.safe_load(f)


def train_epoch(model, loader, optimizer, criterion, device, land_mask_global):
    model.train()
    total_loss = 0.0
    total_mse = 0.0
    total_smooth = 0.0
    n_batches = 0

    lm_device = torch.from_numpy(land_mask_global).bool().to(device)

    for inputs, targets, land_masks in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        pred = model(inputs)

        # Use batch land mask (patch-level)
        lm_batch = land_masks[:, None, :, :].expand_as(pred).bool()
        targets_masked = targets.clone()
        targets_masked[~lm_batch] = float("nan")

        loss, mse, smooth = criterion(pred, targets_masked, lm_batch[:, 0, :, :])
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        total_mse += mse.item()
        total_smooth += smooth.item()
        n_batches += 1

    if n_batches == 0:
        return 0.0, 0.0, 0.0
    return total_loss / n_batches, total_mse / n_batches, total_smooth / n_batches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "pipeline" / "config.yaml"))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--T", type=int, default=None, help="Temporal window (not used in T=1 mode)")
    parser.add_argument("--patch-size", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    tc = cfg["training"]
    mc = cfg["model"]

    # CLI overrides
    epochs = args.epochs or tc["epochs"]
    lr = args.lr or tc["lr"]
    patch_size = args.patch_size or mc["patch_size"]

    device = torch.device("cpu")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}  LR: {lr}  Patch size: {patch_size}")

    # Dataset
    print("\nLoading dataset...")
    try:
        dataset = OceanDataset(patch_size=patch_size, augment=True,
                               n_patches_per_day=tc.get("n_patches_per_day", 8))
    except FileNotFoundError as e:
        print(f"\nERROR: {e}")
        print("Please run the pipeline scripts first:")
        print("  python pipeline/harmonize.py")
        print("  python pipeline/targets.py")
        sys.exit(1)

    print(f"  Dataset size: {len(dataset)} patches from {len(dataset.dates)} days")
    print(f"  Days: {dataset.dates}")

    loader = DataLoader(
        dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc["num_workers"],
        pin_memory=False,
    )

    # Model
    model = build_model(cfg).to(device)
    info = model_info(model)
    print(f"\nModel: {info['architecture']}")
    print(f"  Parameters: {info['total_parameters']:,}")

    # Loss & optimizer
    criterion = OceanEmbedLoss(lambda_smooth=tc["lambda_smooth"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=tc["weight_decay"])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.1)

    # Land mask for loss
    land_mask_np = dataset.land_mask  # (H, W)

    # Training loop
    weights_dir = ROOT / "weights"
    weights_dir.mkdir(exist_ok=True)
    checkpoint_path = ROOT / tc["checkpoint"]
    log_path = weights_dir / "training_log.json"

    best_loss = float("inf")
    log = {"epochs": [], "best_epoch": -1, "config": {"epochs": epochs, "lr": lr}}

    print("\nStarting training...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss, mse, smooth = train_epoch(model, loader, optimizer, criterion, device, land_mask_np)
        scheduler.step()
        elapsed = time.time() - t0

        log["epochs"].append({
            "epoch": epoch,
            "total_loss": round(loss, 6),
            "mse_loss": round(mse, 6),
            "smooth_loss": round(smooth, 6),
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "time_s": round(elapsed, 2),
        })

        print(f"  Epoch {epoch:3d}/{epochs} | Loss={loss:.5f} MSE={mse:.5f} "
              f"Smooth={smooth:.5f} | LR={optimizer.param_groups[0]['lr']:.6f} | {elapsed:.1f}s")

        if loss < best_loss:
            best_loss = loss
            log["best_epoch"] = epoch
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss,
                "config": cfg,
                "model_info": info,
            }, checkpoint_path)
            print(f"    ✓ Saved best checkpoint (loss={best_loss:.5f})")

        # Save log periodically
        with open(log_path, "w") as f:
            json.dump(log, f, indent=2)

    print(f"\n✓ Training complete. Best loss: {best_loss:.5f} at epoch {log['best_epoch']}")
    print(f"  Checkpoint: {checkpoint_path}")

    # Train baseline
    print("\n──── Training linear regression baseline ────")
    try:
        baseline = train_baseline(cfg)
        baseline_path = ROOT / tc["baseline_checkpoint"]
        baseline.save(baseline_path)
    except FileNotFoundError as e:
        print(f"  Warning: {e}")
        print("  Baseline not trained (need normalisation stats from harmonize.py)")

    print("\n✓ train.py complete.")


if __name__ == "__main__":
    main()
