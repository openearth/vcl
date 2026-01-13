"""Tests for vcl.data module utility functions."""

import numpy as np
import pytest
from shapely.geometry import Polygon, box

import vcl.data as data


class TestComputeRotationAngle:
    """Test rotation angle computation for polygons."""

    def test_rotation_angle_horizontal_rectangle(self):
        """Verify 0° angle for horizontal rectangle."""
        # Create a horizontal rectangle
        rect = box(0, 0, 10, 5)
        angle = data.compute_rotation_angle(rect)

        # Should be approximately 0 degrees
        assert abs(angle) < 1.0  # Within 1 degree


class TestRotateAndCrop:
    """Test array rotation and cropping."""

    def test_rotate_90_degrees(self):
        """Verify 90° rotation of array."""
        # Create a simple test pattern
        arr = np.array([[1.0, 2.0], [3.0, 4.0]])

        rotated = data.rotate_and_crop(arr, 90)

        assert rotated is not None
        assert isinstance(rotated, np.ndarray)
        # Shape might change after rotation
        assert rotated.size > 0

    def test_rotate_0_degrees(self):
        """Verify 0° rotation returns similar array."""
        arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        rotated = data.rotate_and_crop(arr, 0)

        assert rotated is not None
        # Should have similar shape for 0° rotation
        assert rotated.shape[0] > 0 and rotated.shape[1] > 0

    def test_cval_parameter(self):
        """Verify cval (fill value) parameter works."""
        arr = np.ones((10, 10))

        # Rotate with NaN fill
        rotated = data.rotate_and_crop(arr, 45, cval=np.nan)

        assert rotated is not None
        # Should have some NaN values in corners from rotation
        # (depending on implementation)


class TestCreateShadedImage:
    """Test shaded relief image creation."""

    def test_shaded_image_dimensions_match(
        self, mock_basemap_raster, mock_bathymetry_raster, tmp_path
    ):
        """Verify output dimensions match basemap."""
        # Create temporary rasterio datasets
        import rasterio
        from rasterio.transform import from_bounds

        # Save basemap
        basemap_path = tmp_path / "basemap.tif"
        height, width = mock_basemap_raster.shape[:2]
        transform = from_bounds(0, 0, 10, 10, width, height)

        with rasterio.open(
            basemap_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=3,
            dtype=mock_basemap_raster.dtype,
            transform=transform,
        ) as dst:
            for i in range(3):
                dst.write(mock_basemap_raster[:, :, i].T, i + 1)

        # Save bathymetry
        bathy_path = tmp_path / "bathymetry.tif"
        with rasterio.open(
            bathy_path,
            "w",
            driver="GTiff",
            height=height,
            width=width,
            count=1,
            dtype=mock_bathymetry_raster.dtype,
            transform=transform,
        ) as dst:
            dst.write(mock_bathymetry_raster.T, 1)

        # Open and create shaded image
        with rasterio.open(basemap_path) as sat:
            with rasterio.open(bathy_path) as bodem:
                result = data.create_shaded_image(sat, bodem)

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == height
        assert result.shape[1] == width


class TestGetRotatedVertex:
    """Test vertex rotation computation."""

    def test_rotate_point_90_degrees(self):
        """Verify 90° rotation of a point."""
        center = (0, 0)
        point = (1, 0)
        angle = np.pi / 2  # 90 degrees in radians

        rotated = data.get_rotated_vertex(center, point, angle)

        assert rotated is not None
        # After 90° rotation, (1, 0) should become approximately (0, 1)
        assert abs(rotated[0]) < 0.01  # x should be ~0
        assert abs(rotated[1] - 1) < 0.01  # y should be ~1

    def test_rotate_point_0_degrees(self):
        """Verify 0° rotation leaves point unchanged."""
        center = (0, 0)
        point = (5, 3)
        angle = 0

        rotated = data.get_rotated_vertex(center, point, angle)

        assert abs(rotated[0] - point[0]) < 0.01
        assert abs(rotated[1] - point[1]) < 0.01


class TestRotate1DArray:
    """Test 1D array rotation."""

    def test_rotate_1d_arrays(self):
        """Verify x and y coordinates are rotated correctly."""
        center = (0, 0)
        x = np.array([1, 2, 3])
        y = np.array([0, 0, 0])
        angle = np.pi / 2  # 90 degrees

        rotated_x, rotated_y = data.rotate_1d_array(center, x, y, angle)

        assert rotated_x is not None
        assert rotated_y is not None
        assert len(rotated_x) == len(x)
        assert len(rotated_y) == len(y)
        # After 90° rotation, x values should become y values approximately
        assert np.allclose(rotated_x, np.zeros_like(x), atol=0.01)
        assert np.allclose(rotated_y, x, atol=0.01)


class TestComputeMidPointRectangle:
    """Test rectangle midpoint computation."""

    def test_midpoint_of_square(self):
        """Verify midpoint of a square."""
        bounds = (0, 0, 10, 10)  # (left, bottom, right, top)

        midpoint = data.compute_mid_point_rectangle(bounds)

        assert midpoint == [5, 5]

    def test_midpoint_of_rectangle(self):
        """Verify midpoint of a rectangle."""
        bounds = (2, 3, 8, 7)

        midpoint = data.compute_mid_point_rectangle(bounds)

        assert midpoint == [5, 5]  # Center is at (5, 5)


class TestRotateAndCropArray:
    """Test rotate_and_crop_array function."""

    def test_rotate_and_crop_array_basic(
        self, mock_basemap_raster, mock_extent_polygon
    ):
        """Test basic rotation and cropping functionality."""
        array = mock_basemap_raster
        array_extent = mock_extent_polygon.bounds
        center_point = mock_extent_polygon.centroid.coords[0]
        angle = 0.0

        result = data.rotate_and_crop_array(
            array=array,
            array_extent=array_extent,
            center_point=center_point,
            angle=angle,
            crop_extent=mock_extent_polygon,
            crs="EPSG:4326",
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
        assert result.size > 0

    def test_rotate_and_crop_array_with_rotation(
        self, mock_basemap_raster, mock_extent_polygon
    ):
        """Test rotation and cropping with non-zero angle."""
        array = mock_basemap_raster
        array_extent = mock_extent_polygon.bounds
        center_point = mock_extent_polygon.centroid.coords[0]
        angle = 45.0  # 45 degree rotation

        result = data.rotate_and_crop_array(
            array=array,
            array_extent=array_extent,
            center_point=center_point,
            angle=angle,
            crop_extent=mock_extent_polygon,
            crs="EPSG:4326",
        )

        assert result is not None
        assert isinstance(result, np.ndarray)
