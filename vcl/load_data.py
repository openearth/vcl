from pathlib import Path

import geopandas as gpd
import matplotlib.image as mpimg
import numpy as np
import rasterio
import rioxarray as rxr
import xarray as xr


def load():
    # p_drive_dir = Path(r"P:\11209197-virtualclimatelab\01_data\Delft3D")
    data_dir = Path("~/data/vcl/dataset").expanduser()

    # Dataset for ground water model (concentrations)
    # Dataset ds is ordered with (z,y,x) coordinates
    ds = xr.open_dataset(data_dir.joinpath("concentratie_data_gw_model.nc"))
    ds_n = xr.open_dataset(data_dir.joinpath("conc_nieuw.nc"))

    # Dataset of bathymetry
    # ds_b0 = rasterio.open(data_dir / "terschelling_maquette_def.tif")
    ds_b0 = rasterio.open(data_dir / "originele_bodem.tif")
    ds_b0_n = rasterio.open(data_dir / "nieuwe_bodem_v2.tif")

    # North sea and Wad sea dataset
    ds_wl = xr.open_dataset(data_dir / "wadsea_small.nc")
    # Extents of what we want to show
    extent = (
        gpd.read_file(data_dir / "bounding_box.shp").to_crs(epsg=28992).iloc[0].geometry
    )
    # extent = (
    #     gpd.read_file(data_dir / "vaklodingen_outline.shp")
    #     .to_crs(epsg=28992)
    #     .iloc[0]
    #     .geometry
    # )

    # Open arial photo of Terschelling (+ surrounding area)
    sat = rasterio.open(data_dir / "test3.tif")

    GXG = rasterio.open(
        data_dir / "Freatische GXG/Referentie/Zomer_grondwaterstand_m_mv.tif"
    )
    GXG_n = rasterio.open(
        data_dir / "Freatische GXG/2100/Zomer_grondwaterstand_m_mv.tif"
    )

    GSR = rasterio.open(data_dir / "RecreatiezonderRot.png")
    GVG = rasterio.open(data_dir / "GVGzonderrotatie.png")
    ecotopen = rasterio.open(data_dir / "Ecotopen zonder rotatie en legend.png")
    floodmap = rasterio.open(data_dir / "Water_mask_difference_20230915_20240309.tif")

    animation_files = list(Path(data_dir / "Historische_ontwikkeling").glob("*.tiff"))

    common_datasets = {
        "ds_b0": ds_b0,
        "extent": extent,
        "sat": sat,
        "GSR": GSR,
        "GVG": GVG,
        "ecotoop": ecotopen,
        "floodmap": floodmap,
        "animation_files": animation_files,
        "ds_wl": ds_wl,
    }

    unique_datasets = {
        "2023": {
            "extent": extent,
            "ds": ds,
            "ds_b0": ds_b0,
            "GXG": GXG,
            "ssp": {"laag": ds, "hoog": ds},
        },
        "2050": {
            "extent": extent,
            "ds": ds_n,
            "ds_b0": ds_b0,
            "GXG": GXG_n,
            "ssp": {"laag": ds_n - 1, "hoog": ds_n * 2},
        },
        "2100": {
            "extent": extent,
            "ds": ds_n + 2,
            "ds_b0": ds_b0_n,
            "GXG": GXG,
            "ssp": {"laag": ds_n * -1, "hoog": ds_n * 0.5},
        },
    }
    return common_datasets, unique_datasets


def load_preprocessed():
    data_dir = Path("~/data/vcl/dataset").expanduser()
    datasets = np.load(data_dir / "preprocessed-data.npy", allow_pickle=True).item()
    return datasets
