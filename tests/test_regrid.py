"""Tests for regridding utilities in harmonize.py."""
import sys
from pathlib import Path
import numpy as np
import pytest
import xarray as xr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def build_test_da(lat, lon, fill_value=25.0):
    """Create a simple DataArray with known values."""
    data = np.full((len(lat), len(lon)), fill_value, dtype=np.float32)
    return xr.DataArray(data, dims=["latitude", "longitude"],
                        coords={"latitude": lat, "longitude": lon})


class TestRegrid:
    """Tests for regrid_to_target."""

    def test_output_grid_shape(self):
        """Output should match target lat/lon dimensions."""
        from pipeline.harmonize import regrid_to_target, build_target_grid, load_config
        cfg = load_config()
        tgt_lat, tgt_lon = build_target_grid(cfg)
        src_lat = np.arange(5.0, 30.1, 0.1)
        src_lon = np.arange(45.0, 105.1, 0.1)
        da = build_test_da(src_lat, src_lon, fill_value=28.0)
        result = regrid_to_target(da, tgt_lat, tgt_lon)
        assert result.shape == (len(tgt_lat), len(tgt_lon)), (
            f"Expected {(len(tgt_lat), len(tgt_lon))}, got {result.shape}"
        )

    def test_output_grid_resolution(self):
        """Output latitude spacing should be 0.25°."""
        from pipeline.harmonize import regrid_to_target, build_target_grid, load_config
        cfg = load_config()
        tgt_lat, tgt_lon = build_target_grid(cfg)
        src_lat = np.arange(5.0, 30.1, 0.5)
        src_lon = np.arange(45.0, 105.1, 0.5)
        da = build_test_da(src_lat, src_lon)
        result = regrid_to_target(da, tgt_lat, tgt_lon)
        spacing = np.diff(result["latitude"].values)
        assert np.allclose(spacing, 0.25, atol=1e-6), f"Lat spacing not 0.25°: {spacing[:3]}"

    def test_constant_field_preserved(self):
        """A constant-value field should remain constant after interpolation."""
        from pipeline.harmonize import regrid_to_target, build_target_grid, load_config
        cfg = load_config()
        tgt_lat, tgt_lon = build_target_grid(cfg)
        src_lat = np.arange(5.0, 30.5, 0.5)
        src_lon = np.arange(45.0, 105.5, 0.5)
        da = build_test_da(src_lat, src_lon, fill_value=27.5)
        result = regrid_to_target(da, tgt_lat, tgt_lon)
        # Interior values should be close to 27.5
        interior = result.values[2:-2, 2:-2]
        assert np.allclose(interior, 27.5, atol=0.1), (
            f"Constant field not preserved. Max deviation: {np.abs(interior - 27.5).max():.4f}"
        )

    def test_nan_land_mask_preserved(self):
        """NaN values (land) should remain NaN after regridding."""
        from pipeline.harmonize import regrid_to_target, build_target_grid, load_config
        cfg = load_config()
        tgt_lat, tgt_lon = build_target_grid(cfg)
        src_lat = np.arange(5.0, 30.5, 0.5)
        src_lon = np.arange(45.0, 105.5, 0.5)
        data = np.full((len(src_lat), len(src_lon)), 25.0, dtype=np.float32)
        # Set a strip to NaN
        data[:, :5] = np.nan
        da = xr.DataArray(data, dims=["latitude", "longitude"],
                          coords={"latitude": src_lat, "longitude": src_lon})
        result = regrid_to_target(da, tgt_lat, tgt_lon)
        # First column of output (lon=45°) should be NaN
        first_col = result.values[:, 0]
        # At least some NaN values should propagate
        assert np.any(np.isnan(first_col)), "NaN land mask not preserved after regridding"

    def test_domain_bounds(self):
        """Output coordinates should be within NIO domain."""
        from pipeline.harmonize import regrid_to_target, build_target_grid, load_config
        cfg = load_config()
        tgt_lat, tgt_lon = build_target_grid(cfg)
        src_lat = np.arange(0.0, 35.0, 0.5)
        src_lon = np.arange(40.0, 110.0, 0.5)
        da = build_test_da(src_lat, src_lon)
        result = regrid_to_target(da, tgt_lat, tgt_lon)
        d = cfg["domain"]
        assert float(result["latitude"].min()) >= d["lat_min"] - 0.01
        assert float(result["latitude"].max()) <= d["lat_max"] + 0.01
        assert float(result["longitude"].min()) >= d["lon_min"] - 0.01
        assert float(result["longitude"].max()) <= d["lon_max"] + 0.01
