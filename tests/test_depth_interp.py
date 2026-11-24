"""Tests for depth interpolation utilities in targets.py and argo_profiles.py."""
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

TARGET_DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]


class TestDepthInterp:
    """Tests for 1-D profile interpolation."""

    def test_basic_interpolation(self):
        """Linear interpolation should return correct values at target depths."""
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([0., 100., 500., 1000.], dtype=float)
        # Linear profile: T = 30 - 0.02 * depth
        src_temps = 30.0 - 0.02 * src_depths
        tgt = np.array([0., 50., 100., 500., 1000.], dtype=float)
        result = interp_depths_1d(src_depths, src_temps, tgt)
        expected = 30.0 - 0.02 * tgt
        np.testing.assert_allclose(result, expected, atol=0.01, err_msg="Linear interp failed")

    def test_exact_points_preserved(self):
        """Values at exact source depths should be preserved exactly."""
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([0., 10., 50., 100., 200.], dtype=float)
        src_temps = np.array([28., 27., 25., 22., 18.], dtype=float)
        # Request interpolation at the exact source points
        result = interp_depths_1d(src_depths, src_temps, src_depths)
        np.testing.assert_allclose(result, src_temps, atol=0.001)

    def test_all_15_target_depths(self):
        """Should return exactly 15 values for the standard target depths."""
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([0., 10., 50., 100., 200., 500., 1000.], dtype=float)
        src_temps = np.array([29., 28., 25., 22., 18., 12., 8.], dtype=float)
        tgt = np.array(TARGET_DEPTHS, dtype=float)
        result = interp_depths_1d(src_depths, src_temps, tgt)
        assert len(result) == 15, f"Expected 15 values, got {len(result)}"
        assert not np.all(np.isnan(result)), "All values are NaN"

    def test_nan_input_returns_nan(self):
        """All-NaN profile should return all-NaN output."""
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([0., 100., 500.], dtype=float)
        src_temps = np.array([np.nan, np.nan, np.nan])
        result = interp_depths_1d(src_depths, src_temps, np.array([50., 200.]))
        assert np.all(np.isnan(result)), "Expected all NaN for NaN input"

    def test_fewer_than_2_valid_returns_nan(self):
        """Single valid point is insufficient for interpolation → NaN."""
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([0., 100., 500.], dtype=float)
        src_temps = np.array([28.0, np.nan, np.nan])
        result = interp_depths_1d(src_depths, src_temps, np.array([50.]))
        assert np.all(np.isnan(result)), "Expected NaN for <2 valid points"

    def test_boundary_extrapolation(self):
        """Depths outside source range should use nearest edge value (not blow up)."""
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([10., 100., 500.], dtype=float)
        src_temps = np.array([28., 22., 12.], dtype=float)
        # Requesting depth 0 (shallower than src_depths[0])
        result = interp_depths_1d(src_depths, src_temps, np.array([0., 1000.]))
        # Should not be NaN and should be within reasonable range
        assert not np.isnan(result[0]), "Shallow extrapolation returned NaN"
        assert not np.isnan(result[1]), "Deep extrapolation returned NaN"
        assert 5 <= result[0] <= 35, f"Extrapolated surface value out of range: {result[0]}"
        assert 0 <= result[1] <= 35, f"Extrapolated deep value out of range: {result[1]}"

    def test_monotone_decreasing_profile(self):
        """
        Standard ocean profile should stay within source temperature range
        after interpolation.
        """
        from pipeline.targets import interp_depths_1d
        src_depths = np.array([0., 5., 10., 20., 50., 100., 200., 500., 1000.], dtype=float)
        src_temps = np.array([29.5, 29.3, 29.0, 28.0, 25.0, 20.0, 15.0, 10.0, 7.0], dtype=float)
        tgt = np.array(TARGET_DEPTHS, dtype=float)
        result = interp_depths_1d(src_depths, src_temps, tgt)
        t_min, t_max = src_temps.min(), src_temps.max()
        assert np.all(result >= t_min - 0.1), f"Values below source minimum: {result.min():.2f}"
        assert np.all(result <= t_max + 0.1), f"Values above source maximum: {result.max():.2f}"

    def test_argo_profile_interp(self):
        """ARGO profile interpolation (from argo_profiles.py) should work correctly."""
        from pipeline.argo_profiles import interp_profile
        pres = np.array([0., 10., 25., 50., 75., 100., 150., 200., 300., 500.])
        temp = np.array([28.5, 28.2, 27.8, 26.5, 24.0, 21.0, 16.0, 13.0, 10.0, 8.0])
        tgt = np.array(TARGET_DEPTHS, dtype=float)
        result = interp_profile(pres, temp, tgt)
        assert len(result) == len(TARGET_DEPTHS)
        assert not np.any(np.isnan(result)), f"Unexpected NaN values: {result}"
        # Surface should be near 28.5
        assert abs(result[0] - 28.5) < 0.5, f"Surface temp off: {result[0]:.2f}"
