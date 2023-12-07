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


def preprocess(datasets):
    ds = datasets["ds"]
    ds_n = datasets["ds_n"]

    ds_b0 = datasets["ds_b0"]
    # ds_b0_n = datasets["ds_b0_n"]

    bodem = ds_b0.read(1)
    bodem[np.where(bodem == -9999)] = -43.8

    # Replace negative concentrations (due to model errors) with 0
    ds = ds.where((ds.conc >= 0) | ds.isnull(), other=0)
    # ds_n = ds_n.where((ds_n.conc >= 0) | ds_n.isnull(), other=0)

    klein = {}
    groot = {}

    klein["extent"] = datasets["extent_klein"]
    groot["extent"] = datasets["extent_klein"]

    klein["angle"] = vcl.data.compute_rotation_angle(klein["extent"])
    groot["angle"] = vcl.data.compute_rotation_angle(groot["extent"])

    klein["mid_point"] = klein["extent"].centroid.coords[0]
    groot["mid_point"] = groot["extent"].centroid.coords[0]

    # klein["sat_extent"] = datasets["sat"]
    # groot["sat_extent"] = datasets["sat"]
    sat = datasets["sat"]

    klein["plt_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)
    groot["plt_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)

    klein["plt_lims"] = vcl.data.get_plot_lims(
        shapely.affinity.rotate(klein["extent"], -klein["angle"])
    )
    groot["plt_lims"] = vcl.data.get_plot_lims(
        shapely.affinity.rotate(groot["extent"], -groot["angle"])
    )

    # bodem0_n = np.array(ds_b0_n[0, 1435:2466, 1201:2970])
    # bodem0_n[np.where(bodem0_n == -9999)] = -43.8

    # Ratio of 13.14 / 8.94 is that of width / height of satellite image, which is needed to make sure the ratio after rotation
    # is ~ 2 / 1
    klein["sat"] = vcl.data.create_shaded_image(sat, ds_b0)
    groot["sat"] = vcl.data.create_shaded_image(sat, ds_b0)

    # Each z layer of concentration dataset needs to be rotated and cropped seperately to account for the slices in x and y direction
    # Create dummy array of one layer and apply the function to it to obtain its shape
    # Note that the first dimension of ds_n is time instead of z

    dummy = vcl.data.rotate_and_crop(ds.conc.values[0, :, :], -klein["angle"])

    rot_ds = np.zeros((ds.conc.shape[0], dummy.shape[0], dummy.shape[1]))
    klein["conc"] = np.zeros((ds.conc.shape[0], dummy.shape[0], dummy.shape[1]))
    groot["conc"] = np.zeros((ds.conc.shape[0], dummy.shape[0], dummy.shape[1]))

    for i in range(rot_ds.shape[0]):
        klein["conc"][i, ...] = vcl.data.rotate_and_crop(
            ds.conc.values[i, ...], -klein["angle"]
        )
        groot["conc"][i, ...] = vcl.data.rotate_and_crop(
            ds.conc.values[i, ...], -groot["angle"]
        )

    print("Prima zo")
    # rot_img_shade = vcl.data.rotate_and_crop(img_shade, -15)
    # klein["sat"] = vcl.data.rotate_and_crop(klein["img_shade"], -15)
    # groot["sat"] = vcl.data.rotate_and_crop(groot["img_shade"], -15)
    # rot_bodem0 = vcl.data.rotate_and_crop(bodem0, -15)

    # yb0 = np.linspace(0, rot_img_shade.shape[0], rot_img_shade.shape[0])
    # smooth_bodem = np.zeros_like(rot_bodem0)
    # for i in range(smooth_bodem.shape[1]):
    #     smoother = scipy.interpolate.UnivariateSpline(yb0, rot_bodem0[:, i])
    #     smooth_bodem[:, i] = smoother(yb0)

    klein["conc"] = vcl.data.fit_rot_ds_to_bounds(
        ds, klein["conc"], klein["mid_point"], klein["plt_lims"], -klein["angle"]
    )

    groot["conc"] = vcl.data.fit_rot_ds_to_bounds(
        ds, groot["conc"], klein["mid_point"], groot["plt_lims"], -groot["angle"]
    )

    print("Prima zo")

    # Set new extent
    klein["img_shade_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)
    groot["img_shade_extent"] = vcl.data.sat_and_bodem_bounds(sat, ds_b0)

    klein["top_contour_extent"] = vcl.data.contour_bounds(ds)
    groot["top_contour_extent"] = vcl.data.contour_bounds(ds)

    klein["extent_n"] = 0, klein["sat"].shape[1], 0, klein["sat"].shape[0]
    groot["extent_n"] = 0, groot["sat"].shape[1], 0, groot["sat"].shape[0]

    klein["X2"], klein["Y2"] = np.meshgrid(
        np.linspace(klein["plt_lims"][0], klein["plt_lims"][2], klein["conc"].shape[2]),
        np.linspace(klein["plt_lims"][3], klein["plt_lims"][1], klein["conc"].shape[1]),
    )
    groot["X2"], groot["Y2"] = np.meshgrid(
        np.linspace(groot["plt_lims"][0], groot["plt_lims"][2], groot["conc"].shape[2]),
        np.linspace(groot["plt_lims"][3], groot["plt_lims"][1], groot["conc"].shape[1]),
    )

    Yk, Zk = np.meshgrid(
        np.linspace(klein["plt_lims"][3], klein["plt_lims"][1], klein["conc"].shape[1]),
        ds.z,
    )

    Yg, Zg = np.meshgrid(
        np.linspace(groot["plt_lims"][3], groot["plt_lims"][1], groot["conc"].shape[1]),
        ds.z,
    )

    # Set contour cmap
    cmap = ListedColormap(["royalblue", "coral"])
    contour_show = False

    # Initialize an array to store the intersections
    # Note that we choose an array filled with -2 instead of 0, since we have a contour level of 0. So in this case a value of -2
    # indicates a nan value
    klein["nbpixels_x"] = 160
    klein["nbpixels_y"] = klein["conc"].shape[1]
    groot["nbpixels_x"] = 160
    groot["nbpixels_y"] = groot["conc"].shape[1]

    print("Prima zo")
    klein["conc_contours_x"] = vcl.data.contourf_to_array_3d(
        klein["conc"],
        klein["nbpixels_x"],
        klein["nbpixels_y"],
        Yk,
        Zk,
        levels=[-3, 1.5, 16],
    )
    print("Prima zo")
    groot["conc_contours_x"] = vcl.data.contourf_to_array_3d(
        groot["conc"],
        groot["nbpixels_x"],
        groot["nbpixels_y"],
        Yg,
        Zg,
        levels=[-3, 1.5, 16],
    )

    ds_b0.close()
    sat.close()

    return {"klein": klein, "groot": groot}
