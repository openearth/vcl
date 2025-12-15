import json
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
        print("-" * 60)
        print("Preprocessing common layers...")
        print("-" * 60)
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

    print("Preprocessing basemap and bathymetry...")
    preprocessed = preprocess_essentials(
        datasets=datasets, preprocessed=preprocessed, base_path=base_path, crs=crs
    )

    extra_info = {"extent": extent, "mid_point": mid_point, "angle": angle, "crs": crs}
    extra_info = datasets.get("extra_info", {}) | extra_info

    print("Preprocessing layers...")
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

    print("Preprocessing info...")
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

    print("Preprocessing animations...")
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

    print("Preprocessing particles...")
    preprocessed["particles"] = {}
    pbar = tqdm(datasets["particles"], unit="layer")
    for layer in pbar:
        layer_path = base_path / datasets["particles"][layer]
        particle_data = preprocess_particles(
            file_path=layer_path, layer=layer, extra_info=extra_info
        )
        preprocessed["particles"][layer] = particle_data

    print("Preprocessing interactivity polygons...")
    preprocessed["interactivity"] = {}
    pbar = tqdm(datasets["interactivity"], unit="layer")
    for layer in pbar:
        layer_path = base_path / datasets["interactivity"][layer]
        layer_data = gpd.read_file(layer_path)
        preprocessed["interactivity"][layer] = layer_data

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
    return


def preprocess_nc(file_path: Path, layer: str, extra_info: dict):
    ds = xr.open_dataset(file_path)
    return


def preprocess_shape(file_path: Path, layer: str, extra_info: dict):
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
