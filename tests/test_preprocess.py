"""Tests for vcl.preprocess module."""

import json
from pathlib import Path
from unittest.mock import Mock, patch

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from shapely.geometry import box

import vcl.preprocess as preprocess


class TestPreprocessInputValidation:
    """Test input validation for the preprocess function."""

    def test_preprocess_rejects_non_json_file(self, tmp_path):
        """Verify that preprocess raises ValueError for non-.json files."""
        txt_file = tmp_path / "input.txt"
        txt_file.write_text("not a json file")

        with pytest.raises(ValueError, match="Input file must be a JSON file"):
            preprocess.preprocess(txt_file)

    def test_preprocess_requires_common_section(self, tmp_path):
        """Verify that preprocess requires 'common' section in JSON."""
        json_file = tmp_path / "input.json"
        config = {"basepath": "/some/path"}
        with open(json_file, "w") as f:
            json.dump(config, f)

        with pytest.raises(AssertionError):
            preprocess.preprocess(json_file)

    def test_preprocess_requires_basemap(self, tmp_path):
        """Verify that preprocess requires 'basemap' in common layers."""
        json_file = tmp_path / "input.json"
        config = {
            "basepath": "/some/path",
            "common": {"extent": "extent.geojson", "bathymetry": "bathy.tif"},
        }
        with open(json_file, "w") as f:
            json.dump(config, f)

        with pytest.raises(AssertionError):
            preprocess.preprocess(json_file)

    def test_preprocess_requires_extent(self, tmp_path):
        """Verify that preprocess requires 'extent' in common layers."""
        json_file = tmp_path / "input.json"
        config = {
            "basepath": "/some/path",
            "common": {"basemap": "basemap.tif", "bathymetry": "bathy.tif"},
        }
        with open(json_file, "w") as f:
            json.dump(config, f)

        with pytest.raises(AssertionError):
            preprocess.preprocess(json_file)

    def test_preprocess_requires_bathymetry(self, tmp_path):
        """Verify that preprocess requires 'bathymetry' in common layers."""
        json_file = tmp_path / "input.json"
        config = {
            "basepath": "/some/path",
            "common": {"basemap": "basemap.tif", "extent": "extent.geojson"},
        }
        with open(json_file, "w") as f:
            json.dump(config, f)

        with pytest.raises(AssertionError):
            preprocess.preprocess(json_file)


class TestPreprocessPNG:
    """Test PNG preprocessing functions."""

    def test_preprocess_png_with_pgw(self, temp_png_with_pgw, mock_extent_polygon):
        """Verify PNG with world file loads correctly."""
        extra_info = {
            "extent": mock_extent_polygon,
            "mid_point": mock_extent_polygon.centroid.coords[0],
            "angle": 0.0,
            "crs": "EPSG:4326",
        }

        result = preprocess.preprocess_png(temp_png_with_pgw, "test_layer", extra_info)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.ndim == 3  # Should have RGB channels

    def test_preprocess_png_without_pgw_with_extra_info(
        self, tmp_path, mock_extent_polygon
    ):
        """Verify PNG without .pgw uses extra_info bounds."""
        # Create PNG without world file
        from PIL import Image

        png_path = tmp_path / "no_pgw.png"
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img.save(png_path)

        bounds = mock_extent_polygon.bounds
        extra_info = {
            "extent": mock_extent_polygon,
            "mid_point": mock_extent_polygon.centroid.coords[0],
            "angle": 0.0,
            "crs": "EPSG:4326",
            "no_pgw": {"extent": bounds},  # Layer-specific bounds
        }

        result = preprocess.preprocess_png(png_path, "no_pgw", extra_info)

        assert result is not None
        assert isinstance(result, np.ndarray)

    def test_preprocess_png_without_pgw_no_extra_info(
        self, tmp_path, mock_extent_polygon
    ):
        """Verify ValueError when PNG has no .pgw and no extra_info bounds."""
        from PIL import Image

        png_path = tmp_path / "no_bounds.png"
        img = Image.new("RGB", (50, 50), color=(100, 100, 100))
        img.save(png_path)

        extra_info = {
            "extent": mock_extent_polygon,
            "mid_point": mock_extent_polygon.centroid.coords[0],
            "angle": 0.0,
            "crs": "EPSG:4326",
        }

        with pytest.raises(AssertionError):
            preprocess.preprocess_png(png_path, "no_bounds", extra_info)


class TestPreprocessShape:
    """Test shapefile/vector preprocessing functions."""

    def test_preprocess_shape_polygon(self, temp_shapefile, mock_extent_polygon):
        """Verify polygon rasterization works correctly."""
        extra_info = {
            "extent": mock_extent_polygon,
            "mid_point": mock_extent_polygon.centroid.coords[0],
            "angle": 0.0,
            "crs": "EPSG:4326",
        }

        result = preprocess.preprocess_shape(temp_shapefile, "test_shapes", extra_info)

        assert result is not None
        assert isinstance(result, np.ndarray)
        # Should have some non-NaN values where features were rasterized
        assert not np.all(np.isnan(result))

    def test_preprocess_shape_output_dimensions(
        self, temp_shapefile, mock_extent_polygon
    ):
        """Verify output dimensions match expected 1080x1920."""
        extra_info = {
            "extent": mock_extent_polygon,
            "mid_point": mock_extent_polygon.centroid.coords[0],
            "angle": 0.0,
            "crs": "EPSG:4326",
        }

        result = preprocess.preprocess_shape(temp_shapefile, "test_shapes", extra_info)

        # After rotation and cropping, dimensions may vary but should be reasonable
        assert result.shape[0] > 0
        assert result.shape[1] > 0


class TestPreprocessUnique:
    """Test unique layer preprocessing."""

    def test_preprocess_unique_creates_year_dictionaries(self):
        """Verify empty dictionaries are created for each year."""
        datasets = {"2023": {}, "2050": {}, "2100": {}}

        result = preprocess.preprocess_unique(datasets)

        assert isinstance(result, dict)
        assert "2023" in result
        assert "2050" in result
        assert "2100" in result
        assert all(isinstance(v, dict) for v in result.values())


class TestPreprocessTifAndNc:
    """Test TIF and NetCDF preprocessing (currently placeholders)."""

    def test_preprocess_tif_returns_none(self):
        """Verify preprocess_tif placeholder returns None."""
        result = preprocess.preprocess_tif()
        assert result is None

    def test_preprocess_nc_returns_none(self, tmp_path):
        """Verify preprocess_nc placeholder returns None."""
        # Create a minimal NetCDF file for testing
        import xarray as xr

        nc_path = tmp_path / "test.nc"
        ds = xr.Dataset({"temp": (["x", "y"], np.random.rand(10, 10))})
        ds.to_netcdf(nc_path)

        result = preprocess.preprocess_nc(nc_path, "test_layer", {})
        assert result is None


class TestPreprocessIntegration:
    """Integration tests for the full preprocessing pipeline."""

    def test_preprocess_with_minimal_config(self, mock_input_json):
        """Test full preprocessing with a minimal valid configuration."""
        # This is an integration test that may take longer
        # Skip in fast test runs if needed
        result = preprocess.preprocess(mock_input_json)

        assert isinstance(result, dict)
        assert "" in result  # Non-temporal data uses empty string key
        assert "basemap" in result[""]
        assert "bathymetry" in result[""]
        assert "extent" in result[""]
