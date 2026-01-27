from pathlib import Path

import geopandas as gpd
import matplotlib.image as mpimg
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr

from typing import Union


def load():
    # p_drive_dir = Path(r"P:\11209197-virtualclimatelab\01_data\Delft3D")
    data_dir = Path("~/data/vcl/gnsbi").expanduser()

    # Extents of what we want to show
    extent = (
        gpd.read_file(data_dir / "aoi-bbox-1-to-2.gpkg")
        .to_crs(epsg=4326)
        .iloc[0]
        .geometry
    )

    # Open arial photo of Terschelling (+ surrounding area)
    sat = rasterio.open(data_dir / "world_ocean_basemap.tif")
    ds_b0 = rasterio.open(data_dir / "bathymetry-elevation-combined-smoothed.tif")

    common_datasets = {
        "extent": extent,
        "sat": sat,
        "ds_b0": ds_b0,
    }

    unique_datasets = {"2023": {}}
    return common_datasets, unique_datasets


def load_preprocessed(data_path: Union[str, Path]):
    datasets = np.load(data_path, allow_pickle=True)
    return datasets
