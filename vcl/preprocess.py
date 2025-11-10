import json
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import rasterio
import shapely
import xarray as xr
import matplotlib.pyplot as plt

import vcl.data

data_dir = data_dir = Path("~/data/vcl/gnsbi").expanduser()


def preprocess(input_file: Union[str, Path]):
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
    base_path = Path(base_path)
    # Create dictionary to store processed data and values
    preprocessed = {}

    extent = gpd.read_file(base_path / datasets["extent"]).iloc[0]["geometry"]
    angle = vcl.data.compute_rotation_angle(extent)
    mid_point = extent.centroid.coords[0]

    preprocessed = preprocess_essentials(
        datasets=datasets, preprocessed=preprocessed, base_path=base_path, crs=crs
    )

    extra_info = {"extent": extent, "mid_point": mid_point, "angle": angle, "crs": crs}
    extra_info = datasets.get("extra_info", {}) | extra_info

    for layer, layer_path in datasets["layers"].items():
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

    preprocessed["stats"] = {}
    for layer, stats_types in datasets["stats"].items():
        # preprocessed["stats"][layer] = {}
        preprocessed["stats"][layer] = []
        for stat_type, layer_paths in stats_types.items():
            if stat_type == "image":
                for layer_path in layer_paths:
                    data = plt.imread(base_path / layer_path)
                    preprocessed["stats"][layer].append(("image", data))

    preprocessed["animations"] = {}
    for layer, file_dir in datasets["animations"].items():
        file_dir = base_path / file_dir
        assert (base_path / file_dir).is_dir(), f"Directory {file_dir} does not exist."
        animation_data = []
        for layer_path in sorted(file_dir.glob("*")):
            year = layer_path.stem[:4]
            if layer_path.suffix == ".png":
                layer_data = preprocess_png(
                    file_path=layer_path, layer=layer, extra_info=extra_info
                )
            animation_data.append({"frame": layer_data, "text": year})
        preprocessed["animations"][layer] = animation_data

    return preprocessed


def preprocess_unique(datasets):
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
    base_path = Path(base_path)

    extent = gpd.read_file(base_path / datasets["extent"]).iloc[0]["geometry"]
    preprocessed["extent"] = extent
    angle = vcl.data.compute_rotation_angle(extent)
    mid_point = extent.centroid.coords[0]

    basemap = rasterio.open(base_path / datasets["basemap"])
    basemap_bounds = basemap.bounds
    bathymetry = rasterio.open(base_path / datasets["bathymetry"])

    basemap = vcl.data.create_shaded_image(basemap, bathymetry)

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
    with rasterio.open(file_path) as src:
        bounds = src.bounds
        data = src.read()
    if not file_path.with_suffix(".pgw").exists():
        layer_info = extra_info.get(layer, None)
        assert layer_info is not None, ValueError(
            "Bounds for layer {layer} not found. Please add pgw file or add bounds as extra info."
        )
        bounds = layer_info.get("extent", None)
        assert bounds is not None, ValueError(
            "Bounds for layer {layer} not found. Please add pgw file or add bounds as extra info."
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
    return


def preprocess_nc():
    return


def preprocess_shape(file_path: Path, layer: str, extra_info: dict):
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
