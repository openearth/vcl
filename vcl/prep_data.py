from pathlib import Path

import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import shapely
from matplotlib.colors import LightSource, ListedColormap
from pyproj import Transformer

import vcl.data
import vcl.display
import vcl.load_data


data_dir = data_dir = Path("~/data/vcl/dataset").expanduser()


def preprocess_common(datasets):
    # Get bathymetry datasets
    ds_b0 = datasets["ds_b0"]
    ecotoop = datasets["ecotoop"]
    GSR = datasets["GSR"]
    GVG = datasets["GVG"]
    floodmap = datasets["floodmap"]

    ds_wl = datasets["ds_wl"]
    # Create dictionary to store processed data and values
    preprocessed = {}

    # Get the extent we want to show
    preprocessed["extent"] = datasets[f"extent"]

    # Add satellite image
    sat = datasets["sat"]

    # Get other datasets and their bounds
    (
        preprocessed["ecotoop"],
        preprocessed["ecotoop_extent"],
    ) = vcl.data.prepare_rasterio_image(ecotoop)
    (
        preprocessed["GSR"],
        preprocessed["GSR_extent"],
    ) = vcl.data.prepare_rasterio_image(GSR)
    (
        preprocessed["GVG"],
        preprocessed["GVG_extent"],
    ) = vcl.data.prepare_rasterio_image(GVG)

    preprocessed["floodmap"], preprocessed["floodmap_extent"] = (
        vcl.data.prepare_rasterio_image(floodmap, Transformer.from_crs(32631, 28992))
    )

    # Compute rotation angle of the extent as well as centre point of the extent
    preprocessed["angle"] = vcl.data.compute_rotation_angle(preprocessed["extent"])
    preprocessed["mid_point"] = preprocessed["extent"].centroid.coords[0]

    # Create shaded image from satellite and bathymetry
    preprocessed["sat"] = vcl.data.create_shaded_image(sat, ds_b0)
    # Compute combined bounds of satellite image and the bathymetry
    preprocessed["plt_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)
    # Get plot lims of rotated extent (rotated such that extent has an angle of 0 with the horizontal axis)
    preprocessed["plt_lims"] = vcl.data.get_plot_lims(
        shapely.affinity.rotate(preprocessed["extent"], -preprocessed["angle"])
    )

    # Set new extent
    preprocessed["sat_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)

    preprocessed["animation_data"] = datasets["animation_files"]

    tidal_flows = {}
    tidal_flows["face_x"] = ds_wl.mesh2d_face_x.values
    tidal_flows["face_y"] = ds_wl.mesh2d_face_y.values
    tidal_flows["face_x"], tidal_flows["face_y"] = vcl.data.rotate_1d_array(
        preprocessed["mid_point"],
        tidal_flows["face_x"],
        tidal_flows["face_y"],
        -np.deg2rad(preprocessed["angle"]),
    )
    tidal_flows["ucx"] = ds_wl.mesh2d_ucx.values
    tidal_flows["ucy"] = ds_wl.mesh2d_ucy.values
    preprocessed["tidal_flows"] = tidal_flows

    sat.close()
    ecotoop.close()
    GSR.close()
    GVG.close()

    return preprocessed


def preprocess_unique(datasets):
    preprocessed_datasets = {}
    for year in datasets.keys():
        # Get bathymetry datasets
        ds_b0 = datasets[year]["ds_b0"]

        # Replace -9999 with second lowest value
        bodem = ds_b0.read(1)
        bodem[np.where(bodem == -9999)] = -43.8

        GXG = datasets[year]["GXG"]

        # Create dictionary to store processed data and values
        preprocessed = {}

        # Get the extent we want to show
        preprocessed["extent"] = datasets[year][f"extent"]

        # Add bathymetry and its bounds from the dataset
        preprocessed["bodem"] = bodem
        preprocessed["bodem_bounds"] = ds_b0.bounds

        # Compute rotation angle of the extent as well as centre point of the extent
        preprocessed["angle"] = vcl.data.compute_rotation_angle(preprocessed["extent"])
        preprocessed["mid_point"] = preprocessed["extent"].centroid.coords[0]
        bodem_mid_point = vcl.data.compute_mid_point_rectangle(
            preprocessed["bodem_bounds"]
        )

        # Get plot lims of rotated extent (rotated such that extent has an angle of 0 with the horizontal axis)
        preprocessed["plt_lims"] = vcl.data.get_plot_lims(
            shapely.affinity.rotate(preprocessed["extent"], -preprocessed["angle"])
        )

        # Rotate bathymetry
        preprocessed["rotated_bodem"] = vcl.data.rotate_and_crop(
            preprocessed["bodem"], -preprocessed["angle"], cval=-43.8
        )
        (
            preprocessed["rotated_bodem"],
            preprocessed["dx_bodem"],
            preprocessed["dy_bodem"],
        ) = vcl.data.fit_array_to_bounds(
            preprocessed["rotated_bodem"],
            preprocessed["bodem_bounds"],
            bodem_mid_point,
            preprocessed["plt_lims"],
            -preprocessed["angle"],
        )

        preprocessed["smooth_bodem"] = np.zeros_like(preprocessed["rotated_bodem"])

        y_bodem = np.linspace(
            preprocessed["plt_lims"][1],
            preprocessed["plt_lims"][3],
            preprocessed["rotated_bodem"].shape[0],
        )
        for i in range(preprocessed["smooth_bodem"].shape[1]):
            bodem_smoother = scipy.interpolate.UnivariateSpline(
                y_bodem, preprocessed["rotated_bodem"][:, i]
            )
            smooth_bodem = bodem_smoother(y_bodem)
            preprocessed["smooth_bodem"][:, i] = smooth_bodem

        # Each z layer of concentration dataset needs to be rotated and cropped seperately to account for the slices in x and y direction
        # Create dummy array of one layer and apply the function to it to obtain its shape
        # Note that the first dimension of ds_n is time instead of z
        for scenario in datasets[year]["ssp"].keys():
            print(year, scenario)
            preprocessed[f"ssp_{scenario}"] = {}
            # Get salt concentration datasets
            ds = datasets[year]["ssp"][scenario]
            datasets[year]["ssp"][scenario] = None

            # Replace negative concentrations (due to model errors) with 0
            # ds = ds.where((ds.conc >= 0) | ds.isnull(), other=0)
            # ds_n = ds_n.where((ds_n.conc >= 0) | ds_n.isnull(), other=0)
            conc_bounds = (
                ds.conc.x.values[0],
                ds.conc.y.values[-1],
                ds.conc.x.values[-1],
                ds.conc.y.values[0],
            )
            if year == "2023":
                conc = ds.conc.values

            else:
                conc = ds.conc.values[0, ...]

            # Replace negative concentrations (due to model errors) with 0
            conc[np.where(conc < 0)] = 0

            dummy = vcl.data.rotate_and_crop(conc[0, :, :], -preprocessed["angle"])

            rot_ds = np.zeros((conc.shape[0], dummy.shape[0], dummy.shape[1]))
            preprocessed[f"ssp_{scenario}"]["conc"] = np.zeros(
                (conc.shape[0], dummy.shape[0], dummy.shape[1])
            )

            for i in range(rot_ds.shape[0]):
                preprocessed[f"ssp_{scenario}"]["conc"][i, ...] = (
                    vcl.data.rotate_and_crop(conc[i, ...], -preprocessed["angle"])
                )

            (
                preprocessed[f"ssp_{scenario}"]["conc"],
                preprocessed[f"ssp_{scenario}"]["dx_conc"],
                preprocessed[f"ssp_{scenario}"]["dy_conc"],
            ) = vcl.data.fit_array_to_bounds(
                preprocessed[f"ssp_{scenario}"]["conc"],
                conc_bounds,
                preprocessed["mid_point"],
                preprocessed["plt_lims"],
                -preprocessed["angle"],
            )

            # Set new extent
            preprocessed[f"ssp_{scenario}"]["top_contour_extent"] = (
                vcl.data.contour_bounds(ds)
            )

            # Make meshgrid for contour plots
            preprocessed["Y1"], preprocessed["Z1"] = np.meshgrid(
                np.linspace(
                    preprocessed["plt_lims"][3],
                    preprocessed["plt_lims"][1],
                    preprocessed[f"ssp_{scenario}"]["conc"].shape[1],
                ),
                ds.z,
            )

            preprocessed["X2"], preprocessed["Y2"] = np.meshgrid(
                np.linspace(
                    preprocessed["plt_lims"][0],
                    preprocessed["plt_lims"][2],
                    preprocessed[f"ssp_{scenario}"]["conc"].shape[2],
                ),
                np.linspace(
                    preprocessed["plt_lims"][3],
                    preprocessed["plt_lims"][1],
                    preprocessed[f"ssp_{scenario}"]["conc"].shape[1],
                ),
            )

            # Set contour cmap
            cmap = ListedColormap(["royalblue", "coral"])
            contour_show = False

            # Initialize an array to store the intersections
            # Note that we choose an array filled with -2 instead of 0, since we have a contour level of 0. So in this case a value of -2
            # indicates a nan value
            preprocessed["nbpixels_x"] = 320
            preprocessed["nbpixels_y"] = preprocessed[f"ssp_{scenario}"]["conc"].shape[
                1
            ]

            # Compute filled contours
            preprocessed[f"ssp_{scenario}"]["conc_contour_top_view"] = (
                vcl.data.contourf_to_array_3d(
                    np.expand_dims(
                        preprocessed[f"ssp_{scenario}"]["conc"][-10, ...], axis=0
                    ),
                    preprocessed[f"ssp_{scenario}"]["conc"].shape[1],
                    preprocessed[f"ssp_{scenario}"]["conc"].shape[2],
                    1,
                    0,
                    preprocessed["X2"],
                    preprocessed["Y2"],
                    levels=[-1, 1.5, 16],
                )[..., 0]
            )

            preprocessed[f"ssp_{scenario}"]["conc_contours_x"] = (
                vcl.data.contourf_to_array_3d(
                    preprocessed[f"ssp_{scenario}"]["conc"],
                    preprocessed["nbpixels_x"],
                    preprocessed["nbpixels_y"],
                    preprocessed[f"ssp_{scenario}"]["conc"].shape[-1],
                    2,
                    preprocessed["Y1"],
                    preprocessed["Z1"],
                    levels=[-1, 1.5, 16],
                )
            )
            preprocessed[f"ssp_{scenario}"]["conc_contours_x"] = preprocessed[
                f"ssp_{scenario}"
            ]["conc_contours_x"].astype(np.float16)
            del preprocessed[f"ssp_{scenario}"]["conc"]
            np.save(
                data_dir / f"preprocessed-{year}-ssp-{scenario}.npy",
                preprocessed[f"ssp_{scenario}"],
            )
            del ds
            del preprocessed[f"ssp_{scenario}"]

        preprocessed["GXG"], preprocessed["GXG_extent"] = (
            vcl.data.prepare_rasterio_image(GXG)
        )

        preprocessed_datasets[year] = preprocessed

        # animation_data = {}
        # for i, frame in enumerate(datasets[year]["animation_files"]):
        #     animation_data[i] = vcl.data.get_frame_data(frame)

    # Close opened files
    ds_b0.close()
    GXG.close()

    return preprocessed_datasets


def preprocess(common_datasets, unique_datasets):
    preprocessed_common_datasets = preprocess_common(common_datasets)
    preprocessed_unique_datasets = preprocess_unique(unique_datasets)

    # Combine common and unique preprocessed datasets for each year
    preprocessed_datasets = {}
    for year in unique_datasets.keys():
        preprocessed_datasets[year] = {
            **preprocessed_common_datasets,
            **preprocessed_unique_datasets[year],
        }
        for scenario in unique_datasets[year]["ssp"].keys():
            preprocessed_datasets[year][f"ssp_{scenario}"] = np.load(
                data_dir / f"preprocessed-{year}-ssp-{scenario}.npy", allow_pickle=True
            ).item()
    return preprocessed_datasets
