import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy

from matplotlib.colors import LightSource, ListedColormap

matplotlib.use("qtagg")

import vcl.display
import vcl.data
import vcl.load_data


def preprocess(datasets):
    ds = datasets["ds"]
    ds_n = datasets["ds_n"]

    ds_b0 = datasets["ds_b0"]
    ds_b0_n = datasets["ds_b0_n"]

    # Replace negative concentrations (due to model errors) with 0
    ds = ds.where((ds.conc >= 0) | ds.isnull(), other=0)
    ds_n = ds_n.where((ds_n.conc >= 0) | ds_n.isnull(), other = 0)

    klein = {}
    groot = {}

    klein["sat"] = datasets["sat_k"]
    groot["sat"] = datasets["sat_g"]

    sat = datasets["sat"]

    bounds_k, bounds_g = vcl.data.create_bounds()

    klein["x_b0"], klein["y_b0"], klein["bodem0"] = vcl.data.prep_bathymetry_data(ds_b0, vcl.data.get_bathymetry_extent(ds_b0, bounds_k))
    groot["x_b0"], groot["y_b0"], groot["bodem0"] = vcl.data.prep_bathymetry_data(ds_b0, vcl.data.get_bathymetry_extent(ds_b0, bounds_g))

    bodem0 = np.array(ds_b0[0, 1435:2466, 1201:2970])
    bodem0[np.where(bodem0 == -9999)] = -43.8
    bodem0_n = np.array(ds_b0_n[0,1435:2466,1201:2970])
    bodem0_n[np.where(bodem0_n == -9999)] = -43.8

    img_shade = vcl.data.create_shaded_image(sat, bodem0, (int(13.14 / 8.94 * 600), 600))
    # Ratio of 13.14 / 8.94 is that of width / height of satellite image, which is needed to make sure the ratio after rotation
    # is ~ 2 / 1
    klein["img_shade"] = vcl.data.create_shaded_image(klein["sat"], klein["bodem0"], (int(13.14 / 8.94 * 600), 600))
    groot["img_shade"] = vcl.data.create_shaded_image(groot["sat"], groot["bodem0"], (int(13.14 / 8.94 * 600), 600))
    
    conc_extent_k = vcl.data.get_conc_extent(ds, klein["x_b0"], klein["y_b0"])
    conc_extent_g = vcl.data.get_conc_extent(ds, groot["x_b0"], groot["y_b0"])
    print(conc_extent_g)

    klein["conc0"] = vcl.data.rescale_and_fit_ds(ds.conc.values, ((conc_extent_k[0], conc_extent_k[2]),
                                                          (conc_extent_k[1], conc_extent_k[3])), klein["bodem0"].shape, klein["img_shade"][..., 0].shape)
    groot["conc0"] = vcl.data.rescale_and_fit_ds(ds.conc.values, ((conc_extent_g[0], conc_extent_g[2]),
                                                          (conc_extent_g[1], conc_extent_g[3])), groot["bodem0"].shape, groot["img_shade"][..., 0].shape)
    
    # Each z layer of concentration dataset needs to be rotated and cropped seperately to account for the slices in x and y direction
    # Create dummy array of one layer and apply the function to it to obtain its shape
    # Note that the first dimension of ds_n is time instead of z
    dummy = vcl.data.rotate_and_crop(ds.conc.values[0, :, :], -15)
    dummy_k = vcl.data.rotate_and_crop(klein["conc0"][0, ...], -15)
    dummy_g = vcl.data.rotate_and_crop(groot["conc0"][0, ...], -15)

    rot_ds = np.zeros((ds.conc.shape[0], dummy.shape[0], dummy.shape[1]))
    rot_ds_n = np.zeros((ds_n.conc.shape[1], dummy.shape[0], dummy.shape[1]))
    klein["conc"] = np.zeros((klein["conc0"].shape[0], dummy_k.shape[0], dummy_k.shape[1]))
    groot["conc"] = np.zeros((groot["conc0"].shape[0], dummy_g.shape[0], dummy_g.shape[1]))
    for i in range(rot_ds.shape[0]):
        rot_ds[i, :, :] = vcl.data.rotate_and_crop(ds.conc.values[i, :, :], -15)
        rot_ds_n[i, :, :] = vcl.data.rotate_and_crop(ds_n.conc.values[0, i, :, :], -15)
        klein["conc"][i, ...] = vcl.data.rotate_and_crop(klein["conc0"][i, ...], -15)
        groot["conc"][i, ...] = vcl.data.rotate_and_crop(groot["conc0"][i, ...], -15)

    rot_img_shade = vcl.data.rotate_and_crop(img_shade, -15)
    klein["sat"] = vcl.data.rotate_and_crop(klein["img_shade"], -15)
    groot["sat"] = vcl.data.rotate_and_crop(groot["img_shade"], -15)
    rot_bodem0 = vcl.data.rotate_and_crop(bodem0, -15)

    # yb0 = np.linspace(0, rot_img_shade.shape[0], rot_img_shade.shape[0])
    # smooth_bodem = np.zeros_like(rot_bodem0)
    # for i in range(smooth_bodem.shape[1]):
    #     smoother = scipy.interpolate.UnivariateSpline(yb0, rot_bodem0[:, i])
    #     smooth_bodem[:, i] = smoother(yb0)

    # Set new extent
    extent_n = 0, rot_img_shade.shape[1], 0, rot_img_shade.shape[0]
    klein["extent_n"] = 0, klein["sat"].shape[1], 0, klein["sat"].shape[0]
    groot["extent_n"] = 0, groot["sat"].shape[1], 0, groot["sat"].shape[0]


    Y1, Z1 = np.meshgrid(np.linspace(0, rot_img_shade.shape[0], rot_ds.shape[1]), ds.z)
    Yk, Zk = np.meshgrid(np.linspace(0, klein["sat"].shape[0], klein["conc"].shape[1]), ds.z)
    Yg, Zg = np.meshgrid(np.linspace(0, groot["sat"].shape[0], groot["conc"].shape[1]), ds.z)

    X2, Y2 = np.meshgrid(
        np.linspace(0, rot_img_shade.shape[1], rot_ds.shape[2]),
        np.linspace(rot_img_shade.shape[0], 0, rot_ds.shape[1]),
    )
    klein["X2"], klein["Y2"] = np.meshgrid(
        np.linspace(0, klein["sat"].shape[1], klein["conc"].shape[2]),
        np.linspace(klein["sat"].shape[0], 0, klein["conc"].shape[1]),
    )
    groot["X2"], groot["Y2"] = np.meshgrid(
        np.linspace(0, groot["sat"].shape[1], groot["conc"].shape[2]),
        np.linspace(groot["sat"].shape[0], 0, groot["conc"].shape[1]),
    )

    # Set contour cmap
    cmap = ListedColormap(["royalblue", "coral"])
    contour_show = False

    # Initialize an array to store the intersections
    # Note that we choose an array filled with -2 instead of 0, since we have a contour level of 0. So in this case a value of -2
    # indicates a nan value
    nbpixels_x = 160
    nbpixels_y = rot_img_shade.shape[0]
    klein["nbpixels_x"] = 160
    klein["nbpixels_y"] = klein["sat"].shape[0]
    groot["nbpixels_x"] = 160
    groot["nbpixels_y"] = groot["sat"].shape[0]

    conc_contours_x = vcl.data.contourf_to_array_3d(rot_ds, nbpixels_x, nbpixels_y, Y1, Z1, levels=[0, 1.5, 16])
    conc_contours_x_n = vcl.data.contourf_to_array_3d(rot_ds_n, nbpixels_x, nbpixels_y, Y1, Z1, levels=[0, 1.5, 16])
    klein["conc_contours_x"] = vcl.data.contourf_to_array_3d(klein["conc"], klein["nbpixels_x"], klein["nbpixels_y"], Yk, Zk, levels=[-3, 1.5, 16])
    groot["conc_contours_x"] = vcl.data.contourf_to_array_3d(groot["conc"], groot["nbpixels_x"], groot["nbpixels_y"], Yg, Zg, levels=[-3, 1.5, 16])
    klein["conc_contours_x_n"] = vcl.data.contourf_to_array_3d(klein["conc"], klein["nbpixels_x"], klein["nbpixels_y"], Yk, Zk, levels=[-3, 1.5, 16])
    groot["conc_contours_x_n"] = vcl.data.contourf_to_array_3d(groot["conc"], groot["nbpixels_x"], groot["nbpixels_y"], Yg, Zg, levels=[-3, 1.5, 16])

    # updated_datasets = {
    #     "nbpixels_x": nbpixels_x,
    #     "nbpixels_y": nbpixels_y,
    #     "conc_contours_x": conc_contours_x,
    #     "conc_contours_x_n": conc_contours_x_n,
    #     "sat": rot_img_shade,
    #     "extent_n": extent_n,
    #     "X2": X2,
    #     "Y2": Y2,
    #     "conc": rot_ds,
    #     "conc_n": rot_ds_n,
    #     "bodem": smooth_bodem,
    # }
    return {"klein": klein, "groot": groot}
