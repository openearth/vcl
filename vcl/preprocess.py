"""Data preprocessing module for VCL (Virtual Coastal Landscape).

This module provides functionality to preprocess various geospatial data formats including
PNG, TIF, NetCDF, and shapefiles for visualization in the VCL display system. It handles
data rotation, cropping, transformation, and rasterization to prepare data for rendering.

The preprocessing pipeline processes three main types of data:
- Common layers: basemap, bathymetry, extent, and other shared layers
- Unique layers: year-specific data that varies across different time periods
- Essential layers: core data required for all visualizations

Typical usage:
    input_file = Path("path/to/input.json")
    datasets = preprocess(input_file)
"""

import json
import logging
import warnings
from pathlib import Path
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely
import xarray as xr
from rasterio.errors import NotGeoreferencedWarning
from tqdm import tqdm

import vcl.data

logger = logging.getLogger(__name__)


def preprocess(input_file: Union[str, Path]):
    """Main preprocessing function that orchestrates the entire preprocessing pipeline.

    This function reads a JSON configuration file and processes both common and unique
    layers according to the specifications. It validates that required layers (basemap,
    extent, bathymetry) are present and combines the preprocessed data into a unified
    dictionary structure.

    Args:
        input_file: Path to the JSON configuration file containing layer definitions,
                   file paths, and preprocessing parameters.

    Returns:
        dict: Dictionary mapping years (or empty string for non-temporal data) to
             preprocessed datasets. Each dataset contains all common layers plus any
             year-specific unique layers.

    Raises:
        ValueError: If the input file is not a JSON file or if required layers
                   (common, basemap, extent, bathymetry) are missing.

    Example:
        >>> datasets = preprocess("config.json")
        >>> # Access 2050 data
        >>> data_2050 = datasets["2050"]
    """
    input_file = Path(input_file)
    file_extension = input_file.suffix
    if file_extension != ".json":
        raise ValueError("Input file must be a JSON file")
    with open(input_file, "r") as f:
        input_dict = json.load(f)

    assert "common" in input_dict.keys(), ValueError(
        "Input file must contain common layers"
    )
    assert "basemap" in input_dict["common"].keys(), ValueError(
        "basemap must be included in common layers"
    )
    assert "extent" in input_dict["common"].keys(), ValueError(
        "extent must be included in common layers"
    )
    assert "bathymetry" in input_dict["common"].keys(), ValueError(
        "bathymetry must be included in common layers"
    )

    base_path = input_dict.get("basepath")
    base_path = Path(base_path)
    crs = input_dict.get("crs")

    common_layers = input_dict.get("common")
    unique_layers = input_dict.get("unique")

    if common_layers:
        logger.info("-" * 60)
        logger.info("Preprocessing common layers...")
        logger.info("-" * 60)
        preprocessed_common_datasets = preprocess_common(
            common_layers, base_path=base_path, crs=crs
        )
    if unique_layers:
        preprocessed_unique_datasets = preprocess_unique(unique_layers)

    preprocessed_datasets = {}
    if unique_layers:
        for year in preprocessed_unique_datasets.keys():
            preprocessed_datasets[year] = {
                **preprocessed_common_datasets,
                **preprocessed_unique_datasets[year],
            }
    else:
        preprocessed_datasets[""] = preprocessed_common_datasets

    return preprocessed_datasets


def preprocess_common(
    datasets, base_path: Union[str, Path] = "", crs: str = "EPSG:4326"
):
    """Preprocess common layers that are shared across all time periods.

    This function processes all non-temporal data including basemap, bathymetry,
    extent polygon, layers, statistics, animations, particles, and interactivity
    polygons. It handles rotation, cropping, and coordinate transformations.

    Args:
        datasets: Dictionary containing paths and configuration for all layers to process.
        base_path: Base directory path for resolving relative file paths. Defaults to
                  current directory.
        crs: Coordinate Reference System as an EPSG string (e.g., "EPSG:4326").
            Defaults to "EPSG:4326" (WGS84).

    Returns:
        dict: Dictionary containing all preprocessed common data, including:
            - basemap: Processed basemap raster
            - bathymetry: Processed bathymetry raster
            - extent: Geometry defining the area of interest
            - layers: Dictionary of preprocessed data layers
            - stats: Statistics and info panels for each layer
            - animations: Temporal animation data
            - particles: Particle simulation data
            - interactivity: Interactive polygon regions

    Note:
        The extent polygon is used to compute rotation angle and midpoint for
        aligning all data to a common reference frame.
    """
    base_path = Path(base_path)
    # Create dictionary to store processed data and values
    preprocessed = {}

    extent = gpd.read_file(base_path / datasets["extent"]).iloc[0]["geometry"]
    angle = vcl.data.compute_rotation_angle(extent)
    mid_point = extent.centroid.coords[0]

    logger.info("Preprocessing basemap and bathymetry...")
    preprocessed = preprocess_essentials(
        datasets=datasets, preprocessed=preprocessed, base_path=base_path, crs=crs
    )

    extra_info = {"extent": extent, "mid_point": mid_point, "angle": angle, "crs": crs}
    extra_info = datasets.get("extra_info", {}) | extra_info

    logger.info("Preprocessing layers...")
    pbar = tqdm(datasets["layers"], unit="layer")
    for layer in pbar:
        pbar.set_description(f"Processing: {layer}")

        layer_path = base_path / datasets["layers"][layer]
        layer_path = base_path / layer_path
        file_extension = layer_path.suffix

        if file_extension == ".png":
            layer_data = preprocess_png(
                file_path=layer_path, layer=layer, extra_info=extra_info
            )
        elif file_extension == ".tif":
            layer_data = preprocess_tif()
        elif file_extension == ".nc":
            layer_data = preprocess_nc()
        elif file_extension in [".shp", ".gpkg", ".geojson"]:
            layer_data = preprocess_shape(
                file_path=layer_path, layer=layer, extra_info=extra_info
            )
        else:
            raise ValueError(f"Layer {layer} must be of type *.png, *.tif or *.nc")

        preprocessed[layer] = layer_data

    logger.info("Preprocessing info...")
    preprocessed["stats"] = {}
    pbar = tqdm(datasets["stats"], unit="layer")
    for layer in pbar:
        pbar.set_description(f"Processing: {layer}")
        stats_types = datasets["stats"][layer]
        # preprocessed["stats"][layer] = {}
        preprocessed["stats"][layer] = []
        for stat_type, layer_paths in stats_types.items():
            if stat_type == "image":
                for layer_path in layer_paths:
                    data = plt.imread(base_path / layer_path)
                    preprocessed["stats"][layer].append(("image", data))

    logger.info("Preprocessing animations...")
    pbar = tqdm(datasets["animations"], unit="layer")
    preprocessed["animations"] = {}
    for layer in pbar:
        pbar.set_description(f"Processing: {layer}")
        file_dir = base_path / datasets["animations"][layer]
        assert (base_path / file_dir).is_dir(), f"Directory {file_dir} does not exist."
        animation_data = []
        for layer_path in sorted(file_dir.glob("*")):
            year = layer_path.stem[:4].split("_")[0]
            if layer_path.suffix == ".png":
                layer_data = preprocess_png(
                    file_path=layer_path, layer=layer, extra_info=extra_info
                )
            animation_data.append({"frame": layer_data, "text": year})
        preprocessed["animations"][layer] = animation_data

    logger.info("Preprocessing particles...")
    preprocessed["particles"] = {}
    pbar = tqdm(datasets["particles"], unit="layer")
    for layer in pbar:
        layer_path = base_path / datasets["particles"][layer]
        particle_data = preprocess_particles(
            file_path=layer_path, layer=layer, extra_info=extra_info
        )
        preprocessed["particles"][layer] = particle_data

    logger.info("Preprocessing interactivity polygons...")
    preprocessed["interactivity"] = {}
    pbar = tqdm(datasets["interactivity"], unit="layer")
    for layer in pbar:
        layer_path = base_path / datasets["interactivity"][layer]
        layer_data = gpd.read_file(layer_path)
        preprocessed["interactivity"][layer] = layer_data

    return preprocessed


def preprocess_unique(datasets):
    """Preprocess unique layers that vary across different time periods.

    This function creates a structure for year-specific data. Currently a placeholder
    that initializes empty dictionaries for each year.

    Args:
        datasets: Dictionary with years as keys and layer specifications as values.

    Returns:
        dict: Dictionary mapping years to empty preprocessed dataset dictionaries.
             These will be populated with year-specific data.

    Note:
        This is currently a stub implementation. Year-specific preprocessing
        logic should be added here as needed.
    """
    preprocessed_datasets = {}
    for year in datasets.keys():
        preprocessed_datasets[year] = {}

    return preprocessed_datasets


def preprocess_essentials(
    datasets: dict,
    preprocessed: dict,
    base_path: Union[str, Path] = "",
    crs: str = "EPSG:4326",
):
    """Preprocess essential layers required for all visualizations.

    This function processes the core dataset components: basemap, bathymetry, and extent.
    It creates a shaded relief basemap, applies rotation and cropping transformations
    to align all data to the extent polygon.

    Args:
        datasets: Dictionary containing paths to essential layers (basemap, bathymetry, extent).
        preprocessed: Dictionary to store preprocessed results.
        base_path: Base directory path for resolving relative file paths. Defaults to
                  current directory.
        crs: Coordinate Reference System as an EPSG string. Defaults to "EPSG:4326".

    Returns:
        dict: The preprocessed dictionary updated with:
            - extent: Geometry polygon defining the area of interest
            - basemap: Shaded relief basemap aligned to extent
            - bathymetry: Bathymetry raster aligned to extent

    Note:
        The basemap is created by combining the base raster with bathymetric shading
        to create a visually appealing 3D relief effect.
    """
    base_path = Path(base_path)

    extent = gpd.read_file(base_path / datasets["extent"]).iloc[0]["geometry"]
    preprocessed["extent"] = extent
    angle = vcl.data.compute_rotation_angle(extent)
    mid_point = extent.centroid.coords[0]

    basemap = rasterio.open(base_path / datasets["basemap"])
    basemap_bounds = basemap.bounds
    bathymetry = rasterio.open(base_path / datasets["bathymetry"])

    basemap, basemap_bounds = vcl.data.create_shaded_image(basemap, bathymetry)
    print(extent)
    print(basemap_bounds)
    print(angle)

    basemap = vcl.data.rotate_and_crop_array(
        array=basemap,
        array_extent=basemap_bounds,
        center_point=mid_point,
        angle=angle,
        crop_extent=extent,
        crs=crs,
    )

    bathymetry = vcl.data.rotate_and_crop_array(
        array=np.transpose(bathymetry.read(), (1, 2, 0)),
        array_extent=bathymetry.bounds,
        center_point=mid_point,
        angle=angle,
        crop_extent=extent,
        crs=crs,
    )

    preprocessed["basemap"] = basemap
    preprocessed["bathymetry"] = bathymetry

    return preprocessed


def preprocess_png(file_path: Path, layer: str, extra_info: dict):
    """Preprocess PNG raster files with optional georeferencing.

    This function loads PNG images and transforms them to match the target extent.
    It handles both georeferenced PNGs (with .pgw world files) and non-georeferenced
    images where bounds must be specified in extra_info.

    Args:
        file_path: Path to the PNG file to process.
        layer: Name/identifier of the layer being processed.
        extra_info: Dictionary containing transformation parameters:
            - extent: Target geometry extent
            - mid_point: Center point for rotation
            - angle: Rotation angle in degrees
            - crs: Coordinate reference system
            - [layer]: Optional layer-specific extent if no .pgw file exists

    Returns:
        numpy.ndarray: Processed raster array aligned and cropped to the target extent,
                      filled to match the bounding box dimensions.

    Raises:
        ValueError: If the PNG lacks a .pgw file and no bounds are provided in extra_info.

    Note:
        The function rotates and crops the raster to align with the extent polygon,
        then fills any gaps to ensure the output matches the full bounding box.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=NotGeoreferencedWarning)
        with rasterio.open(file_path) as src:
            bounds = src.bounds
            data = src.read()

    if not file_path.with_suffix(".pgw").exists():
        layer_info = extra_info.get(layer, None)
        assert layer_info is not None, ValueError(
            f"Bounds for layer {layer} not found. Please add pgw file or add bounds as extra info."
        )
        bounds = layer_info.get("extent", None)
        assert bounds is not None, ValueError(
            f"Bounds for layer {layer} not found. Please add pgw file or add bounds as extra info."
        )

    cropped_data = vcl.data.rotate_and_crop_array(
        array=np.transpose(data, (1, 2, 0)),
        array_extent=bounds,
        center_point=extra_info["mid_point"],
        angle=extra_info["angle"],
        crop_extent=extra_info["extent"],
        crs=extra_info["crs"],
    )
    extent_bbox = extra_info["extent"].bounds
    cropped_bounds = (
        max(bounds[0], extent_bbox[0]),
        max(bounds[1], extent_bbox[1]),
        min(bounds[2], extent_bbox[2]),
        min(bounds[3], extent_bbox[3]),
    )

    filled_data = vcl.data.fill_array_to_bbox(
        array=cropped_data, array_extent=cropped_bounds, bbox=extent_bbox
    )

    return filled_data


def preprocess_tif():
    """Preprocess TIF/GeoTIFF raster files.

    Placeholder function for TIF file processing. Currently not implemented.

    Returns:
        None

    Note:
        This function needs to be implemented to handle GeoTIFF preprocessing.
        Implementation should follow the pattern of preprocess_png().
    """
    return


def preprocess_nc(file_path: Path, layer: str, extra_info: dict):
    """Preprocess NetCDF files.

    Placeholder function for NetCDF file processing. Opens the dataset but does
    not currently process it.

    Args:
        file_path: Path to the NetCDF (.nc) file.
        layer: Name/identifier of the layer being processed.
        extra_info: Dictionary containing transformation parameters (currently unused).

    Returns:
        None

    Note:
        This function needs to be implemented to extract and process relevant
        variables from NetCDF datasets.
    """
    ds = xr.open_dataset(file_path)
    return


def preprocess_shape(file_path: Path, layer: str, extra_info: dict):
    """Preprocess vector shapefiles, GeoPackages, and GeoJSON files.

    This function converts vector geometries (polygons, lines, points) to rasterized
    arrays suitable for display. LineStrings are buffered to ensure visibility.
    Features are rasterized using their cmap_id attribute for color mapping.

    Args:
        file_path: Path to the vector file (.shp, .gpkg, or .geojson).
        layer: Name/identifier of the layer being processed.
        extra_info: Dictionary containing transformation parameters:
            - extent: Target geometry extent for bounds
            - mid_point: Center point for rotation
            - angle: Rotation angle in degrees
            - crs: Coordinate reference system

    Returns:
        numpy.ndarray: Rasterized and rotated array (1080x1920) with feature values
                      from the cmap_id column. Non-feature pixels contain NaN.

    Note:
        - LineStrings and MultiLineStrings are buffered by 0.1 units for visibility
        - Output resolution is fixed at 1920x1080
        - The cmap_id column must exist in the GeoDataFrame for proper colorization
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="driver GPKG does not support open option CRS"
        )
        gdf = gpd.read_file(file_path, crs=extra_info["crs"])
    array_extent = extra_info["extent"].bounds
    xmin, ymin, xmax, ymax = array_extent

    width = 1920
    height = 1080

    transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, width, height)

    array = np.full((height, width), fill_value=np.nan)

    # Apply buffer to lines only
    shapes = (
        (
            (
                geom.buffer(0.1)
                if isinstance(geom, (shapely.LineString, shapely.MultiLineString))
                else geom
            ),
            value,
        )
        for geom, value in zip(gdf.geometry, gdf.cmap_id)
    )

    # Rasterize directly into the array
    array = rasterio.features.rasterize(
        shapes=shapes,
        out_shape=array.shape,
        fill=np.nan,
        transform=transform,
        dtype=np.float32,
    )

    cropped_array = vcl.data.rotate_and_crop_array(
        array=array,
        array_extent=array_extent,
        center_point=extra_info["mid_point"],
        angle=extra_info["angle"],
        crop_extent=extra_info["extent"],
        crs=extra_info["crs"],
    )

    return cropped_array


def preprocess_particles(file_path: Path, layer: str, extra_info: dict):
    """Preprocess particle simulation data from NetCDF files.

    This function loads particle or flow field data (typically ocean currents) from
    NetCDF files and filters the data to only include points within the area of interest.
    It extracts mesh cell positions and velocity vectors.

    Args:
        file_path: Path to the NetCDF file containing particle/current data.
        layer: Name/identifier of the layer being processed.
        extra_info: Dictionary containing transformation parameters:
            - extent: Geometry defining the area of interest for spatial filtering

    Returns:
        dict: Dictionary containing filtered particle data with keys:
            - face_x: X-coordinates of mesh cell centers within extent
            - face_y: Y-coordinates of mesh cell centers within extent
            - ucx: U-component (x-direction) of currents for all time steps
            - ucy: V-component (y-direction) of currents for all time steps

    Note:
        The NetCDF file is expected to have variables:
        - Mesh_face_x, Mesh_face_y: Spatial coordinates
        - currents_u, currents_v: Velocity components
    """
    ds = xr.open_dataset(file_path)
    x_values = ds["Mesh_face_x"].values
    y_values = ds["Mesh_face_y"].values
    cur_x = ds["currents_u"].values
    cur_y = ds["currents_v"].values

    xmin, ymin, xmax, ymax = extra_info["extent"].bounds

    mask = (
        (x_values >= xmin)
        & (x_values <= xmax)
        & (y_values >= ymin)
        & (y_values <= ymax)
    )

    x_values = x_values[mask]
    y_values = y_values[mask]
    cur_x = cur_x[:, mask]
    cur_y = cur_y[:, mask]
    # points = np.vstack((x_values, y_values)).T

    particle_data = {"face_x": x_values, "face_y": y_values, "ucx": cur_x, "ucy": cur_y}
    return particle_data
