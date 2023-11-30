from pathlib import Path
import matplotlib.image as mpimg
import numpy as np

import rioxarray as rxr
import xarray as xr
import geopandas as gpd
import rasterio


def load():
    data_dir = Path("~/data/vcl/dataset").expanduser()

    # Dataset for ground water model (concentrations)
    # Dataset ds is ordered with (z,y,x) coordinates
    ds = xr.open_dataset(data_dir.joinpath("concentratie_data_gw_model.nc"))
    ds_n = xr.open_dataset(data_dir.joinpath("conc_nieuw.nc"))

    # Dataset of bathymetry
    ds_b0 = rxr.open_rasterio(data_dir.joinpath("originele_bodem.tif"))
    ds_b0_n = rxr.open_rasterio(data_dir.joinpath("nieuwe_bodem_v2.tif"))

    ds_b0 = rasterio.open(data_dir / "originele_bodem.tif")

    # Extents of what we want to show
    extent_klein = (
        gpd.read_file(data_dir / "afmetingen_krappebox.shp")
        .to_crs(epsg=28992)
        .iloc[0]
        .geometry
    )
    extent_groot = (
        gpd.read_file(data_dir / "afmetingen_box.shp")
        .to_crs(epsg=28992)
        .iloc[0]
        .geometry
    )

    # Read satellite image with surrounding sea
    sat = mpimg.imread(data_dir.joinpath("terschelling-sat2.png"))
    sat_k = mpimg.imread(data_dir.joinpath("terschelling-sat-klein.png"))
    sat_g = mpimg.imread(data_dir.joinpath("terschelling-sat-groot.png"))

    sat = rasterio.open(data_dir / "test3.tif")

    return {
        "ds": ds,
        "ds_n": ds_n,
        "ds_b0": ds_b0,
        "ds_b0_n": ds_b0_n,
        "extent_klein": extent_klein,
        "extent_groot": extent_groot,
        "sat": sat,
        "sat_k": sat_k,
        "sat_g": sat_g,
    }
