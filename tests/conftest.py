"""Pytest configuration and shared fixtures for VCL tests."""

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from affine import Affine
from rasterio.crs import CRS
from shapely.geometry import Polygon, box


@pytest.fixture
def mock_extent_polygon():
    """Create a simple rectangular polygon for testing.

    Returns a polygon representing a small area suitable for testing
    geospatial transformations.
    """
    return box(4.0, 52.0, 4.5, 52.5)


@pytest.fixture
def mock_basemap_raster():
    """Create a simple RGB raster array for basemap testing.

    Returns:
        numpy.ndarray: 100x100x3 RGB array with gradient pattern
    """
    # Create a simple gradient pattern
    raster = np.zeros((100, 100, 3), dtype=np.uint8)
    raster[:, :, 0] = np.linspace(0, 255, 100).reshape(1, -1)  # Red gradient
    raster[:, :, 1] = np.linspace(0, 255, 100).reshape(-1, 1)  # Green gradient
    raster[:, :, 2] = 128  # Constant blue
    return raster


@pytest.fixture
def mock_bathymetry_raster():
    """Create a simple bathymetry (depth) raster for testing.

    Returns:
        numpy.ndarray: 100x100 array with depth values from -4000 to 0
    """
    # Create a depth gradient from -4000m (deep) to 0m (surface)
    depth = np.linspace(-4000, 0, 100 * 100).reshape(100, 100)
    return depth.astype(np.float32)


@pytest.fixture
def temp_geotiff(tmp_path, mock_basemap_raster, mock_extent_polygon):
    """Create a temporary GeoTIFF file with georeferencing.

    Args:
        tmp_path: pytest temporary directory fixture
        mock_basemap_raster: RGB raster data
        mock_extent_polygon: Extent for georeferencing

    Returns:
        Path: Path to the created GeoTIFF file
    """
    filepath = tmp_path / "test_basemap.tif"
    bounds = mock_extent_polygon.bounds  # (minx, miny, maxx, maxy)

    # Create affine transform
    width, height = mock_basemap_raster.shape[1], mock_basemap_raster.shape[0]
    transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], width, height
    )

    # Write GeoTIFF
    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=3,
        dtype=mock_basemap_raster.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
    ) as dst:
        # Write RGB bands
        for i in range(3):
            dst.write(mock_basemap_raster[:, :, i], i + 1)

    return filepath


@pytest.fixture
def temp_bathymetry_geotiff(tmp_path, mock_bathymetry_raster, mock_extent_polygon):
    """Create a temporary bathymetry GeoTIFF file.

    Args:
        tmp_path: pytest temporary directory fixture
        mock_bathymetry_raster: Depth raster data
        mock_extent_polygon: Extent for georeferencing

    Returns:
        Path: Path to the created GeoTIFF file
    """
    filepath = tmp_path / "test_bathymetry.tif"
    bounds = mock_extent_polygon.bounds

    width, height = mock_bathymetry_raster.shape[1], mock_bathymetry_raster.shape[0]
    transform = rasterio.transform.from_bounds(
        bounds[0], bounds[1], bounds[2], bounds[3], width, height
    )

    with rasterio.open(
        filepath,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=1,
        dtype=mock_bathymetry_raster.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform,
        nodata=-9999,
    ) as dst:
        dst.write(mock_bathymetry_raster, 1)

    return filepath


@pytest.fixture
def temp_png_with_pgw(tmp_path, mock_extent_polygon):
    """Create a PNG with accompanying world file (.pgw).

    Args:
        tmp_path: pytest temporary directory fixture
        mock_extent_polygon: Extent for world file

    Returns:
        Path: Path to the PNG file (world file created alongside)
    """
    png_path = tmp_path / "test_layer.png"
    pgw_path = tmp_path / "test_layer.pgw"

    # Create a simple test image
    from PIL import Image

    img = Image.new("RGB", (100, 100), color=(200, 150, 100))
    img.save(png_path)

    # Create world file
    bounds = mock_extent_polygon.bounds
    pixel_width = (bounds[2] - bounds[0]) / 100
    pixel_height = (bounds[3] - bounds[1]) / 100

    # World file format: pixel_width, 0, 0, -pixel_height, x_origin, y_origin
    world_content = f"""{pixel_width}
0.0
0.0
{-pixel_height}
{bounds[0]}
{bounds[3]}
"""
    pgw_path.write_text(world_content)

    return png_path


@pytest.fixture
def temp_shapefile(tmp_path, mock_extent_polygon):
    """Create a temporary shapefile with test geometries.

    Args:
        tmp_path: pytest temporary directory fixture
        mock_extent_polygon: Extent containing the features

    Returns:
        Path: Path to the shapefile
    """
    filepath = tmp_path / "test_shapes.shp"

    # Create some test polygons within the extent
    bounds = mock_extent_polygon.bounds
    mid_x = (bounds[0] + bounds[2]) / 2
    mid_y = (bounds[1] + bounds[3]) / 2

    # Create test features
    features = [
        {
            "geometry": box(bounds[0], bounds[1], mid_x, mid_y),
            "cmap_id": 1,
            "name": "Area 1",
        },
        {
            "geometry": box(mid_x, mid_y, bounds[2], bounds[3]),
            "cmap_id": 2,
            "name": "Area 2",
        },
    ]

    gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
    gdf.to_file(filepath)

    return filepath


@pytest.fixture
def mock_input_json(tmp_path, temp_geotiff, temp_bathymetry_geotiff, temp_shapefile):
    """Create a temporary input.json configuration file.

    Args:
        tmp_path: pytest temporary directory fixture
        temp_geotiff: Path to basemap GeoTIFF
        temp_bathymetry_geotiff: Path to bathymetry GeoTIFF
        temp_shapefile: Path to extent shapefile

    Returns:
        Path: Path to the input.json file
    """
    # Create extent GeoJSON
    extent_path = tmp_path / "extent.geojson"
    extent_gdf = gpd.GeoDataFrame(
        [{"geometry": box(4.0, 52.0, 4.5, 52.5)}], crs="EPSG:4326"
    )
    extent_gdf.to_file(extent_path, driver="GeoJSON")

    config = {
        "basepath": str(tmp_path),
        "crs": "EPSG:4326",
        "common": {
            "basemap": temp_geotiff.name,
            "bathymetry": temp_bathymetry_geotiff.name,
            "extent": extent_path.name,
            "layers": {"test_layer": temp_shapefile.name},
            "stats": {},
            "animations": {},
            "particles": {},
            "interactivity": {},
        },
    }

    json_path = tmp_path / "input.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)

    return json_path


@pytest.fixture
def mock_preprocessed_data(
    mock_basemap_raster, mock_bathymetry_raster, mock_extent_polygon
):
    """Create a sample preprocessed dataset structure.

    Returns:
        dict: Preprocessed dataset with typical structure
    """
    return {
        "": {  # Empty year key for non-temporal data
            "basemap": mock_basemap_raster,
            "bathymetry": mock_bathymetry_raster[:, :, np.newaxis],  # Add dimension
            "extent": mock_extent_polygon,
            "layers": {},
            "stats": {},
            "animations": {},
            "particles": {},
            "interactivity": {},
        }
    }
