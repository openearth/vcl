import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy
import shapely

from matplotlib.colors import LightSource, ListedColormap

matplotlib.use("qtagg")

import vcl.display
import vcl.data
import vcl.load_data


def preprocess_common(datasets, size):
    # Get bathymetry datasets
    ds_b0 = datasets["ds_b0"]
    ecotoop = datasets["ecotoop"]
    GSR = datasets["GSR"]
    GVG = datasets["GVG"]

    # Create dictionary to store processed data and values
    preprocessed = {}

    # Get the extent we want to show
    preprocessed["extent"] = datasets[f"extent_{size}"]

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

    sat.close()
    ecotoop.close()
    GSR.close()
    GVG.close()

    return preprocessed


def preprocess_unique(datasets, size):
    preprocessed_datasets = {}
    for scenario in datasets.keys():
        # Get salt concentration datasets
        ds = datasets[scenario]["ds"]

        # Replace negative concentrations (due to model errors) with 0
        ds = ds.where((ds.conc >= 0) | ds.isnull(), other=0)
        # ds_n = ds_n.where((ds_n.conc >= 0) | ds_n.isnull(), other=0)
        conc_bounds = (
            ds.conc.x.values[0],
            ds.conc.y.values[-1],
            ds.conc.x.values[-1],
            ds.conc.y.values[0],
        )

        # Get bathymetry datasets
        ds_b0 = datasets[scenario]["ds_b0"]

        # Replace -9999 with second lowest value
        bodem = ds_b0.read(1)
        bodem[np.where(bodem == -9999)] = -43.8

        # Create dictionary to store processed data and values
        preprocessed = {}

        # Get the extent we want to show
        preprocessed["extent"] = datasets[scenario][f"extent_{size}"]

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
        if scenario == "2023":
            conc = ds.conc.values
        else:
            conc = ds.conc.values[0, ...]

        dummy = vcl.data.rotate_and_crop(conc[0, :, :], -preprocessed["angle"])

        rot_ds = np.zeros((conc.shape[0], dummy.shape[0], dummy.shape[1]))
        preprocessed["conc"] = np.zeros((conc.shape[0], dummy.shape[0], dummy.shape[1]))

        for i in range(rot_ds.shape[0]):
            preprocessed["conc"][i, ...] = vcl.data.rotate_and_crop(
                conc[i, ...], -preprocessed["angle"]
            )

        (
            preprocessed["conc"],
            preprocessed["dx_conc"],
            preprocessed["dy_conc"],
        ) = vcl.data.fit_array_to_bounds(
            preprocessed["conc"],
            conc_bounds,
            preprocessed["mid_point"],
            preprocessed["plt_lims"],
            -preprocessed["angle"],
        )

        # Set new extent
        preprocessed["top_contour_extent"] = vcl.data.contour_bounds(ds)

        # Make meshgrid for contour plots
        preprocessed["Y1"], preprocessed["Z1"] = np.meshgrid(
            np.linspace(
                preprocessed["plt_lims"][3],
                preprocessed["plt_lims"][1],
                preprocessed["conc"].shape[1],
            ),
            ds.z,
        )

        preprocessed["X2"], preprocessed["Y2"] = np.meshgrid(
            np.linspace(
                preprocessed["plt_lims"][0],
                preprocessed["plt_lims"][2],
                preprocessed["conc"].shape[2],
            ),
            np.linspace(
                preprocessed["plt_lims"][3],
                preprocessed["plt_lims"][1],
                preprocessed["conc"].shape[1],
            ),
        )

        # Set contour cmap
        cmap = ListedColormap(["royalblue", "coral"])
        contour_show = False

        # Initialize an array to store the intersections
        # Note that we choose an array filled with -2 instead of 0, since we have a contour level of 0. So in this case a value of -2
        # indicates a nan value
        preprocessed["nbpixels_x"] = 320
        preprocessed["nbpixels_y"] = preprocessed["conc"].shape[1]

        # Compute filled contours
        preprocessed["conc_contour_top_view"] = vcl.data.contourf_to_array_3d(
            np.expand_dims(preprocessed["conc"][-10, ...], axis=0),
            preprocessed["conc"].shape[1],
            preprocessed["conc"].shape[2],
            1,
            0,
            preprocessed["X2"],
            preprocessed["Y2"],
            levels=[-1, 1.5, 16],
        )[..., 0]

        preprocessed["conc_contours_x"] = vcl.data.contourf_to_array_3d(
            preprocessed["conc"],
            preprocessed["nbpixels_x"],
            preprocessed["nbpixels_y"],
            preprocessed["conc"].shape[-1],
            2,
            preprocessed["Y1"],
            preprocessed["Z1"],
            levels=[-1, 1.5, 16],
        )

        preprocessed_datasets[scenario] = preprocessed

        # animation_data = {}
        # for i, frame in enumerate(datasets[scenario]["animation_files"]):
        #     animation_data[i] = vcl.data.get_frame_data(frame)

    # Close opened files
    ds_b0.close()

    return preprocessed_datasets


def preprocess(common_datasets, unique_datasets, size):
    preprocessed_common_datasets = preprocess_common(common_datasets, size)
    preprocessed_unique_datasets = preprocess_unique(unique_datasets, size)

    # Combine common and unique preprocessed datasets for each scenario
    preprocessed_datasets = {}
    for scenario in unique_datasets.keys():
        preprocessed_datasets[scenario] = {
            **preprocessed_common_datasets,
            **preprocessed_unique_datasets[scenario],
        }

    return preprocessed_datasets
