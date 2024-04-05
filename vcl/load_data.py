from pathlib import Path
import matplotlib.image as mpimg
import numpy as np

import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import rasterio


def load():
    p_drive_dir = Path(r"P:\11209197-virtualclimatelab\01_data\Delft3D")
    data_dir = Path("~/data/vcl/dataset").expanduser()

    # Dataset for ground water model (concentrations)
    # Dataset ds is ordered with (z,y,x) coordinates
    ds = xr.open_dataset(data_dir.joinpath("concentratie_data_gw_model.nc"))
    ds_n = xr.open_dataset(data_dir.joinpath("conc_nieuw.nc"))

    # Dataset of bathymetry
    ds_b0 = rasterio.open(data_dir / "originele_bodem.tif")
    ds_b0_n = rasterio.open(data_dir / "nieuwe_bodem_v2.tif")

    # North sea and Wad sea dataset
    ds_wl = xr.open_dataset(p_drive_dir / "wadsea_0000_map.nc")
    # Extents of what we want to show
    extent = (
        gpd.read_file(data_dir / "bounding_box.shp").to_crs(epsg=28992).iloc[0].geometry
    )

    # Open arial photo of Terschelling (+ surrounding area)
    sat = rasterio.open(data_dir / "test3.tif")

    GSR = rasterio.open(data_dir / "RecreatiezonderRot.png")
    GVG = rasterio.open(data_dir / "GVGzonderrotatie.png")
    ecotopen = rasterio.open(data_dir / "Ecotopen zonder rotatie en legend.png")

    animation_files = list(Path(data_dir / "Historische_ontwikkeling").glob("*.tiff"))

    common_datasets = {
        "ds_b0": ds_b0,
        "extent": extent,
        "sat": sat,
        "GSR": GSR,
        "GVG": GVG,
        "ecotoop": ecotopen,
        "animation_files": animation_files,
        "ds_wl": ds_wl,
    }

    unique_datasets = {
        "2023": {
            "extent": extent,
            "ds": ds,
            "ds_b0": ds_b0,
        },
        "2050": {
            "extent": extent,
            "ds": ds_n,
            "ds_b0": ds_b0_n,
        },
    }
    return common_datasets, unique_datasets
