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

    ds_ref_2023 = xr.open_dataset(data_dir.joinpath("conc_reference_2021_av.nc"))
    ds_ref_2050 = xr.open_dataset(data_dir.joinpath("conc_reference_2050_av.nc"))
    ds_ref_2100 = xr.open_dataset(data_dir.joinpath("conc_reference_2100_av.nc"))

    ds_hd_2023 = xr.open_dataset(data_dir.joinpath("conc_Hd_2021_av.nc"))
    ds_hd_2050 = xr.open_dataset(data_dir.joinpath("conc_Hd_2050_av.nc"))
    ds_hd_2100 = xr.open_dataset(data_dir.joinpath("conc_Hd_2100_av.nc"))

    ds_hn_2023 = xr.open_dataset(data_dir.joinpath("conc_Hn_2021_av.nc"))
    ds_hn_2050 = xr.open_dataset(data_dir.joinpath("conc_Hn_2050_av.nc"))
    ds_hn_2100 = xr.open_dataset(data_dir.joinpath("conc_Hn_2100_av.nc"))

    gxgs = ["glg", "gvg", "ghg"]
    gxg_ds = {}

    for gxg in gxgs:
        gxg_ref_2023 = xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_reference_2016_2023.tif"
        )
        gxg_ref_2050 = xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_reference_2047_2054.tif"
        )
        gxg_ref_2100 = xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_reference_2097_2104.tif"
        )

        gxg_ds[f"{gxg}_ref_2023"] = gxg_ref_2023
        gxg_ds[f"{gxg}_ref_2050"] = gxg_ref_2050
        gxg_ds[f"{gxg}_ref_2100"] = gxg_ref_2100

        gxg_ds[f"{gxg}_hn_2023"] = gxg_ref_2023
        gxg_ds[f"{gxg}_hd_2023"] = gxg_ref_2023

        gxg_ds[f"{gxg}_hn_2050"] = gxg_ref_2050 - xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_Hn_2047_2054.tif"
        )
        gxg_ds[f"{gxg}_hd_2050"] = gxg_ref_2050 - xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_Hd_2047_2054.tif"
        )

        gxg_ds[f"{gxg}_hn_2100"] = gxg_ref_2100 - xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_Hn_2097_2104.tif"
        )
        gxg_ds[f"{gxg}_hd_2100"] = gxg_ref_2100 - xr.open_dataset(
            data_dir / f"Freatische GXG/{gxg}_Hd_2097_2104.tif"
        )

    aoi = gpd.read_file(data_dir / "Freatische GXG/aoi.shp")

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

    GSR = rasterio.open(data_dir / "RecreatiezonderRot.png")
    ecotopen = rasterio.open(data_dir / "Ecotopen zonder rotatie en legend.png")

    GSR_2023 = rasterio.open(data_dir / "terrein-2023.png")
    GSR_2050 = rasterio.open(data_dir / "terrein-2050.png")
    GSR_2100 = rasterio.open(data_dir / "terrein-2100.png")

    vogels_2023 = rasterio.open(data_dir / "weidevogels-2023.png")
    vogels_2050 = rasterio.open(data_dir / "weidevogels-2050.png")
    vogels_2100 = rasterio.open(data_dir / "weidevogels-2100.png")

    floodmap = rasterio.open(data_dir / "Water_mask_difference_20230915_20240309.tif")

    animation_files = list(Path(data_dir / "Historische_ontwikkeling").glob("*.tiff"))

    common_datasets = {
        "ds_b0": ds_b0,
        "extent": extent,
        "sat": sat,
        "GSR": GSR_2023,
        "ecotoop": ecotopen,
        "floodmap": floodmap,
        "animation_files": animation_files,
        "ds_wl": ds_wl,
    }

    unique_datasets = {
        "2023": {
            "extent": extent,
            "ds": ds_hd_2023,
            "ds_b0": ds_b0,
            "GSR": GSR_2023,
            "vogels": vogels_2023,
            "GLG": {
                "ref": gxg_ds["glg_ref_2023"],
                "nat": gxg_ds["glg_hn_2023"],
                "droog": gxg_ds["glg_hd_2023"],
            },
            "GVG": {
                "ref": gxg_ds["gvg_ref_2023"],
                "nat": gxg_ds["gvg_hn_2023"],
                "droog": gxg_ds["gvg_hd_2023"],
            },
            "GHG": {
                "ref": gxg_ds["ghg_ref_2023"],
                "nat": gxg_ds["ghg_hn_2023"],
                "droog": gxg_ds["ghg_hd_2023"],
            },
            "ssp": {"ref": ds_ref_2023, "nat": ds_hn_2023, "droog": ds_hd_2023},
            "aoi": aoi,
        },
        "2050": {
            "extent": extent,
            "ds": ds_hd_2050,
            "ds_b0": ds_b0,
            "GSR": GSR_2050,
            "vogels": vogels_2050,
            "GLG": {
                "ref": gxg_ds["glg_ref_2050"],
                "nat": gxg_ds["glg_hn_2050"],
                "droog": gxg_ds["glg_hd_2050"],
            },
            "GVG": {
                "ref": gxg_ds["gvg_ref_2050"],
                "nat": gxg_ds["gvg_hn_2050"],
                "droog": gxg_ds["gvg_hd_2050"],
            },
            "GHG": {
                "ref": gxg_ds["ghg_ref_2050"],
                "nat": gxg_ds["ghg_hn_2050"],
                "droog": gxg_ds["ghg_hd_2050"],
            },
            "ssp": {"ref": ds_ref_2050, "nat": ds_hn_2050, "droog": ds_hd_2050},
            "aoi": aoi,
        },
        "2100": {
            "extent": extent,
            "ds": ds_hd_2100,
            "ds_b0": ds_b0_n,
            "GSR": GSR_2100,
            "vogels": vogels_2100,
            "GLG": {
                "ref": gxg_ds["glg_ref_2100"],
                "nat": gxg_ds["glg_hn_2100"],
                "droog": gxg_ds["glg_hd_2100"],
            },
            "GVG": {
                "ref": gxg_ds["gvg_ref_2100"],
                "nat": gxg_ds["gvg_hn_2100"],
                "droog": gxg_ds["gvg_hd_2100"],
            },
            "GHG": {
                "ref": gxg_ds["ghg_ref_2100"],
                "nat": gxg_ds["ghg_hn_2100"],
                "droog": gxg_ds["ghg_hd_2100"],
            },
            "ssp": {"ref": ds_ref_2100, "nat": ds_hn_2100, "droog": ds_hd_2100},
            "aoi": aoi,
        },
    }
    return common_datasets, unique_datasets


def load_preprocessed():
    data_dir = Path("~/data/vcl/dataset").expanduser()
    datasets = np.load(data_dir / "preprocessed-data.npy", allow_pickle=True).item()
    return datasets
