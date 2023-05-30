"""Console script for vcl."""
import concurrent.futures
import json
import sys
import time
from pathlib import Path

import click
import cv2
import matplotlib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import rioxarray as rxr
import scipy
import xarray as xr
import zmq
from matplotlib.colors import LightSource, ListedColormap
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable
from numpy import cos, mgrid, pi, sin

import vcl.display
import vcl.data

matplotlib.rcParams["toolbar"] = "None"

data_dir = Path("~/data/vcl/dataset").expanduser()

# Dataset for ground water model (concentrations)
# Dataset ds is ordered with (z,y,x) coordinates
ds = xr.open_dataset(data_dir.joinpath("concentratie_data_gw_model.nc"))
# Replace negative concentrations (due to model errors) with 0
ds = ds.where((ds.conc >= 0) | ds.isnull(), other=0)

# Dataset of bathymetry
ds_b0 = rxr.open_rasterio(data_dir.joinpath("originele_bodem.tif"))
ds_b0_n = rxr.open_rasterio(data_dir.joinpath("nieuwe_bodem_v2.tif"))

# Read satellite image with surrounding sea
sat = mpimg.imread(data_dir.joinpath("terschelling-sat2.png"))

extent = ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()

# Grid for chosen area of Terschelling (for the bathymetry)
x_b0, y_b0 = np.array(ds_b0.x[1201:2970]), np.array(ds_b0.y[1435:2466])
X_b0, Y_b0 = np.meshgrid(ds_b0.x[1201:2969], ds_b0.y[1435:2466])
bodem0 = np.array(ds_b0[0, 1435:2466, 1201:2970])
bodem0[np.where(bodem0 == -9999)] = -43.8

ls = LightSource(azdeg=315, altdeg=45)
# Create shade using lightsource
rgb = ls.hillshade(bodem0, vert_exag=5, dx=20, dy=20)
# Scale satellite image to bathymetry shapes
sat_scaled = cv2.resize(
    sat, dsize=(bodem0.shape[1], bodem0.shape[0]), interpolation=cv2.INTER_CUBIC
)
sat_scaled = sat_scaled.astype("float64")

# Add shade to scaled image
img_shade = ls.shade_rgb(sat_scaled, bodem0, vert_exag=5, blend_mode="soft")

cmap = ListedColormap(["royalblue", "coral"])
contour_show = False


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


@click.command()
def main(args=None):
    """Console script for vcl."""

    import ipdb

    ipdb.set_trace()
    contextr = zmq.Context()
    socketr = contextr.socket(zmq.SUB)
    socketr.connect("tcp://localhost:5556")
    socketr.setsockopt(zmq.SUBSCRIBE, b"")

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)
    executor.submit(vcl.display.satellite_window)
    # executor.submit(mayavi_window)
    executor.submit(vcl.display.opencv_window)
    # executor.submit(slider_window)
    executor.submit(vcl.display.contour_slice_window)

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    print("socket", socket)

    i = 0
    while True:
        update = socketr.recv()
        socket.send(update)
        time.sleep(1)
        i += 1
        if i == 20:
            break
    # while True:
    #     #  Wait for next request from client
    #     message = "yo give me a scenario"
    #     time.sleep(1)
    #     socket.send(message.encode())
    #     time.sleep(1)
    #     if i%2 == 0:
    #         socket.send(json.dumps([1,2]).encode())
    #     if i%2 == 1:
    #         socket.send(json.dumps([2,2]).encode())
    #     print(f'sent message {message}')
    #     i += 1
    #     if i == 10:
    #         break

    # return exit status 0
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
