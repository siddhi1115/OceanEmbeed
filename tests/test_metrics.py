"""Tests for metric computation in evaluate.py."""
import sys
from pathlib import Path
import numpy as np
import pytest
import json

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


class TestMetrics:
    """Tests for evaluate.py metric computation functions."""

    def _make_stack(self, n_days=2, n_depths=15, H=10, W=10,
                    pred_offset=0.0):
        """Create synthetic pred/target stacks with known error."""
        rng = np.random.default_rng(42)
        target = rng.uniform(5, 30, (n_days, n_depths, H, W)).astype(np.float32)
        pred = target + pred_offset  # constant offset
        land_mask = np.ones((H, W), dtype=bool)
        land_mask[0, 0] = False  # one land pixel
        # Apply land mask
        for t in range(n_days):
            for d in range(n_depths):
                target[t, d, ~land_mask] = np.nan
                pred[t, d, ~land_mask] = np.nan
        return pred, target, land_mask

    def test_perfect_prediction_rmse_zero(self):
        """Perfect prediction (pred == target) should give RMSE = 0."""
        from ml.evaluate import compute_metrics_per_depth
        pred, target, land_mask = self._make_stack(pred_offset=0.0)
        results = compute_metrics_per_depth(pred, target, land_mask, TARGET_DEPTHS)
        for m in results:
            assert m["rmse"] == 0.0, f"RMSE should be 0 for perfect prediction at depth {m['depth_m']}m"

    def test_known_offset_rmse(self):
        """Constant offset prediction should give RMSE = offset."""
        from ml.evaluate import compute_metrics_per_depth
        offset = 2.0
        pred, target, land_mask = self._make_stack(pred_offset=offset)
        results = compute_metrics_per_depth(pred, target, land_mask, TARGET_DEPTHS)
        for m in results:
            assert m["rmse"] is not None
            np.testing.assert_allclose(
                m["rmse"], offset, atol=0.001,
                err_msg=f"RMSE mismatch at depth {m['depth_m']}m"
            )

    def test_known_bias(self):
        """Constant +2°C prediction should give Bias = +2."""
        from ml.evaluate import compute_metrics_per_depth
        offset = 2.0
        pred, target, land_mask = self._make_stack(pred_offset=offset)
        results = compute_metrics_per_depth(pred, target, land_mask, TARGET_DEPTHS)
        for m in results:
            np.testing.assert_allclose(
                m["bias"], offset, atol=0.001,
                err_msg=f"Bias mismatch at depth {m['depth_m']}m"
            )

    def test_all_15_depths_in_output(self):
        """Output should have one entry per target depth."""
        from ml.evaluate import compute_metrics_per_depth
        pred, target, land_mask = self._make_stack()
        results = compute_metrics_per_depth(pred, target, land_mask, TARGET_DEPTHS)
        assert len(results) == 15, f"Expected 15 depth entries, got {len(results)}"
        depths_out = [m["depth_m"] for m in results]
        assert depths_out == TARGET_DEPTHS

    def test_in_sample_flag_in_output(self):
        """metrics.json must contain split='in-sample'."""
        metrics_path = ROOT / "backend" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not yet generated; run ml/evaluate.py first")
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "split" in metrics, "metrics.json missing 'split' field"
        assert metrics["split"] == "in-sample", (
            f"Expected split='in-sample', got '{metrics['split']}'"
        )

    def test_metrics_have_data_period(self):
        """metrics.json must contain data_period field."""
        metrics_path = ROOT / "backend" / "metrics.json"
        if not metrics_path.exists():
            pytest.skip("metrics.json not yet generated; run ml/evaluate.py first")
        with open(metrics_path) as f:
            metrics = json.load(f)
        assert "data_period" in metrics, "metrics.json missing 'data_period' field"
        assert "2020" in metrics["data_period"], (
            f"data_period should mention 2020: {metrics['data_period']}"
        )

    def test_r2_perfect_prediction(self):
        """Perfect prediction → R² = 1.0 (approximately)."""
        from ml.evaluate import compute_metrics_per_depth
        pred, target, land_mask = self._make_stack(pred_offset=0.0)
        results = compute_metrics_per_depth(pred, target, land_mask, TARGET_DEPTHS)
        for m in results:
            if m["r2"] is not None:
                np.testing.assert_allclose(m["r2"], 1.0, atol=0.001)

    def test_land_mask_applied(self):
        """NaN land pixels should be excluded from metrics."""
        from ml.evaluate import compute_metrics_per_depth
        pred, target, land_mask = self._make_stack(pred_offset=5.0)
        # Mark half the grid as land
        large_mask = np.zeros((10, 10), dtype=bool)
        large_mask[5:, 5:] = True  # only 25 ocean pixels
        n_days, n_depths, H, W = pred.shape
        for t in range(n_days):
            for d in range(n_depths):
                pred[t, d, ~large_mask] = np.nan
                target[t, d, ~large_mask] = np.nan
        results = compute_metrics_per_depth(pred, target, large_mask, TARGET_DEPTHS)
        # RMSE should still be 5.0 (not affected by NaN land)
        for m in results:
            if m["n_valid"] > 0:
                np.testing.assert_allclose(m["rmse"], 5.0, atol=0.01)
