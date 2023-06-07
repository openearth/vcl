import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.colors import LightSource, ListedColormap

matplotlib.use("qtagg")

import vcl.display
import vcl.data
import vcl.load_data


def preprocess(datasets):
    ds = datasets["ds"]
    ds_b0 = datasets["ds_b0"]
    sat = datasets["sat"]
    # Replace negative concentrations (due to model errors) with 0
    ds = ds.where((ds.conc >= 0) | ds.isnull(), other=0)
    bodem0 = np.array(ds_b0[0, 1435:2466, 1201:2970])
    bodem0[np.where(bodem0 == -9999)] = -43.8

    # Scale satellite image to bathymetry shapes
    sat_scaled = cv2.resize(
        sat,
        dsize=(bodem0.shape[1], bodem0.shape[0]),
        interpolation=cv2.INTER_CUBIC,
    )
    sat_scaled = sat_scaled.astype("float64")

    # Create shade using lightsource
    ls = LightSource(azdeg=315, altdeg=45)
    rgb = ls.hillshade(bodem0, vert_exag=5, dx=20, dy=20)

    # Add shade to scaled image
    img_shade = ls.shade_rgb(sat_scaled, bodem0, vert_exag=5, blend_mode="soft")

    # Each z layer of concentration dataset needs to be rotated and cropped seperately to account for the slices in x and y direction
    # Create dummy array of one layer and apply the function to it to obtain its shape
    dummy = vcl.data.rotate_and_crop(ds.conc.values[0, :, :], -15)
    rot_ds = np.zeros((ds.conc.shape[0], dummy.shape[0], dummy.shape[1]))
    for i in range(rot_ds.shape[0]):
        rot_ds[i, :, :] = vcl.data.rotate_and_crop(ds.conc.values[i, :, :], -15)

    rot_img_shade = vcl.data.rotate_and_crop(img_shade, -15)
    rot_bodem0 = vcl.data.rotate_and_crop(bodem0, -15)

    # Set new extent
    extent_n = 0, rot_img_shade.shape[1], 0, rot_img_shade.shape[0]

    Y1, Z1 = np.meshgrid(np.linspace(0, rot_img_shade.shape[0], rot_ds.shape[1]), ds.z)
    X2, Y2 = np.meshgrid(
        np.linspace(0, rot_img_shade.shape[1], rot_ds.shape[2]),
        np.linspace(rot_img_shade.shape[0], 0, rot_ds.shape[1]),
    )

    # Set contour cmap
    cmap = ListedColormap(["royalblue", "coral"])
    contour_show = False

    # Initialize an array to store the intersections
    # Note that we choose an array filled with -2 instead of 0, since we have a contour level of 0. So in this case a value of -2
    # indicates a nan value
    nbpixels_x = 160
    nbpixels_y = rot_img_shade.shape[0]

    conc_contours_x = np.zeros((nbpixels_x, nbpixels_y, rot_ds.shape[-1])) - 2
    for i in range(conc_contours_x.shape[-1]):
        cf = plt.contourf(Y1, Z1, rot_ds[:, :, i], levels=[0, 1.5, 16])
        conc_contours_x[:, :, i] = np.flip(
            vcl.data.contourf_to_array(cf, nbpixels_x, nbpixels_y, Y1, Z1), axis=0
        )
        # Change all values smaller than -1 to nan, since they were nan values before converting the contours to arrays
        conc_contours_x[:, :, i][np.where(conc_contours_x[:, :, i] < -1)] = np.nan
    plt.close("all")
    updated_datasets = {
        "nbpixels_x": nbpixels_x,
        "nbpixels_y": nbpixels_y,
        "conc_contours_x": conc_contours_x,
        "sat": rot_img_shade,
        "extent_n": extent_n,
        "X2": X2,
        "Y2": Y2,
    }
    return updated_datasets
