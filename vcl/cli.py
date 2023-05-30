"""Console script for vcl."""
import concurrent.futures
import sys
import time

import json
import click
import cv2
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.colors import LightSource, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.widgets import Slider, Button

import numpy as np
import xarray as xr
import rioxarray as rxr
import scipy

import zmq
from numpy import cos, mgrid, pi, sin

from pathlib import Path

matplotlib.rcParams['toolbar'] = 'None'

data_dir = Path("~/data/vcl/dataset").expanduser()

# Dataset for ground water model (concentrations)
# Dataset ds is ordered with (z,y,x) coordinates
ds = xr.open_dataset(data_dir.joinpath('concentratie_data_gw_model.nc'))
# Replace negative concentrations (due to model errors) with 0
ds = ds.where((ds.conc >= 0) | ds.isnull(), other = 0)

# Dataset of bathymetry
ds_b0 = rxr.open_rasterio(data_dir.joinpath('originele_bodem.tif'))
ds_b0_n = rxr.open_rasterio(data_dir.joinpath('nieuwe_bodem_v2.tif'))

# Read satellite image with surrounding sea
sat = mpimg.imread(data_dir.joinpath('terschelling-sat2.png'))

extent = ds.x.values.min(), ds.x.values.max(), ds.y.values.min(), ds.y.values.max()

# Grid for chosen area of Terschelling (for the bathymetry)
x_b0, y_b0 = np.array(ds_b0.x[1201:2970]), np.array(ds_b0.y[1435:2466])
X_b0, Y_b0 = np.meshgrid(ds_b0.x[1201:2969], ds_b0.y[1435:2466])
bodem0 = np.array(ds_b0[0,1435:2466,1201:2970])
bodem0[np.where(bodem0 == -9999)] = -43.8

ls = LightSource(azdeg=315, altdeg=45)
# Create shade using lightsource
rgb = ls.hillshade(bodem0,
            vert_exag=5, dx=20, dy=20)
# Scale satellite image to bathymetry shapes
sat_scaled = cv2.resize(sat, dsize=(bodem0.shape[1], bodem0.shape[0]), interpolation=cv2.INTER_CUBIC)
sat_scaled = sat_scaled.astype('float64')

# Add shade to scaled image
img_shade = ls.shade_rgb(sat_scaled, bodem0, vert_exag=5, blend_mode='soft')

cmap = ListedColormap(['royalblue', 'coral'])
contour_show = False


def rotate_and_crop(arr, ang):
    """Array arr to be rotated by ang degrees and cropped afterwards"""
    arr_rot = scipy.ndimage.rotate(arr, ang, reshape=True, order=0)
    
    shift_up = np.ceil(np.arcsin(abs(ang)/360 * 2*np.pi) * arr.shape[1])
    shift_right = np.ceil(np.arcsin(abs(ang)/360 * 2*np.pi) * arr.shape[0])
    
    arr_crop = arr_rot[int(shift_up):arr_rot.shape[0] - int(shift_up), int(shift_right):arr_rot.shape[1] - int(shift_right)]

    return arr_crop

def contourf_to_array(cs, nbpixels_x, nbpixels_y, scale_x, scale_y):
    """Draws filled contours from contourf or tricontourf cs on output array of size (nbpixels_x, nbpixels_y)"""
    image = np.zeros((nbpixels_x,nbpixels_y)) - 2

    for i,collection in enumerate(cs.collections):
        z = cs.levels[i] # get contour levels from cs
        for path in collection.get_paths():
            verts = path.to_polygons() # get vertices of current contour level (is a list of arrays)
            for v in verts:
                # rescale vertices to image size
                v[:,0] = (v[:,0] - np.min(scale_x)) / (np.max(scale_x) - np.min(scale_x)) * nbpixels_y
                v[:,1] = (v[:,1] - np.min(scale_y)) / (np.max(scale_y) - np.min(scale_y)) * nbpixels_x
                poly = np.array([v], dtype=np.int32) # dtype integer is necessary for the next instruction
                cv2.fillPoly( image, poly, z )
    return image

# Each z layer of concentration dataset needs to be rotated and cropped seperately to account for the slices in x and y direction
# Create dummy array of one layer and apply the function to it to obtain its shape
dummy = rotate_and_crop(ds.conc.values[0,:,:], -15)
rot_ds = np.zeros((ds.conc.shape[0], dummy.shape[0], dummy.shape[1]))
for i in range(rot_ds.shape[0]):
    rot_ds[i,:,:] = rotate_and_crop(ds.conc.values[i,:,:], -15)

rot_img_shade = rotate_and_crop(img_shade, -15)
rot_bodem0 = rotate_and_crop(bodem0, -15)

# Set new extent
extent_n = 0, rot_img_shade.shape[1], 0, rot_img_shade.shape[0]

Y1, Z1 = np.meshgrid(np.linspace(0,rot_img_shade.shape[0],rot_ds.shape[1]), ds.z)
X2, Y2 = np.meshgrid(np.linspace(0,rot_img_shade.shape[1],rot_ds.shape[2]), np.linspace(rot_img_shade.shape[0],0,rot_ds.shape[1]))
# Initialize an array to store the intersections
# Note that we choose an array filled with -2 instead of 0, since we have a contour level of 0. So in this case a value of -2 
# indicates a nan value
nbpixels_x = 160
nbpixels_y = rot_img_shade.shape[0]
conc_contours_x = np.zeros((nbpixels_x, nbpixels_y, rot_ds.shape[-1])) - 2
for i in range(conc_contours_x.shape[-1]):
    cf = plt.contourf(Y1, Z1, rot_ds[:,:,i], levels=[0,1.5,16])
    conc_contours_x[:,:,i] = np.flip(contourf_to_array(cf, nbpixels_x, nbpixels_y, Y1, Z1), axis=0)
    # Change all values smaller than -1 to nan, since they were nan values before converting the contours to arrays
    conc_contours_x[:,:,i][np.where(conc_contours_x[:,:,i] < -1)] = np.nan
plt.close('all')

def opencv_window():
    img = np.zeros([100, 100, 3])
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.imshow("window", img)

    while True:
        k = cv2.waitKey(0)
        if k == ord('f'):
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)           
        elif k == ord('n'):
            cv2.setWindowProperty('window', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif k == ord('q'):
            cv2.destroyWindow("window")
            break
        else:
            break



def satellite_window():

    #def key_press(event):
    #    if event.key == 'x':
    #        line.set_ydata()

    print("starting matplotlib")


    #print("matplotlib socket", socket)
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    """

    context = zmq.Context()
    socket1 = context.socket(zmq.SUB)
    socket1.connect("tcp://localhost:5556")
    socket1.subscribe("x_slice")
    
    socket2 = context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:5556")
    socket2.subscribe("top_view")

    poller = zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)

    init_x = 100
    matplotlib.use('qtagg')
    fig, ax = plt.subplots()
    im_sat = ax.imshow(rot_img_shade, extent=extent_n)    # keep window open
    im_c = ax.contourf(X2, Y2, rot_ds[-10,:,:], levels=[0,1.5,16], vmin=0, vmax=15, extent=extent_n, alpha=0, cmap=cmap)
    line, = ax.plot([2.5*init_x, 2.5*init_x], [extent_n[2], extent_n[3]],
         color='blue', linewidth=2)
    
    nm, lbl = im_c.legend_elements()
    lbl[0]= 'Zoet water'
    lbl[1] = 'Zout water'
    legend = ax.legend(nm, lbl, fontsize= 8, loc='upper left', framealpha=1) 

    plt.axis('off')
    manager = plt.get_current_fig_manager()
    #manager.full_screen_toggle()
    plt.show(block=False)
    plt.pause(0.1)
    while True:
        socks = dict(poller.poll(10))
        if socket1 in socks and socks[socket1] == zmq.POLLIN:
            #message = json.loads(socket1.recv().decode())
            topic, message = socket1.recv(zmq.DONTWAIT).split()
            #if message[0] == 'x_slice':
            slider_val = int(message)
            line.set_xdata([2.5*slider_val, 2.5*slider_val])
            #if message[0] == 'top_view':
            plt.pause(0.04)

        if socket2 in socks and socks[socket2] == zmq.POLLIN:
            #message = json.loads(socket2.recv().decode())
            topic, message2 = socket2.recv(zmq.DONTWAIT).split()
            alpha = float(message2)
            #alpha = 0
            if alpha == 0:
                for c in im_c.collections:
                    c.set_alpha(0)
                for i in range(2):
                    legend.get_patches()[i].set(alpha=0)
                    legend.get_texts()[i].set(alpha=0)
                legend.draw_frame(False)
            else:
                for c in im_c.collections:
                    c.set_alpha(0.3)
                for i in range(2):
                    legend.get_patches()[i].set_alpha(0.3)
                    legend.get_texts()[i].set_alpha(1)
                legend.draw_frame(True) 
            plt.pause(0.04)


def contour_slice_window():

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    #def key_press(event):
    #    if event.key == 'x':
    #        line.set_ydata()

    print("starting matplotlib")


    # Define initial parameters (index instead of x value)
    init_x = 100
    extent_x = (0, nbpixels_y, -140, 25.5)

    matplotlib.use('qtagg')
    fig, axes = plt.subplots()

    # adjust the main plot to make room for the slider
    fig.subplots_adjust(left=0.05, bottom=0.25)
        
    # pcolormesh is faster, but not as smooth as contourf
    im_x = axes.imshow(conc_contours_x[:,:,init_x], vmin=0, vmax=1.5, extent=extent_x, aspect='auto', cmap=cmap)

    # Make a horizontal slider to control the position on the x-axis.
    x_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    x_slider = Slider(
        ax=x_ax,
        label='x',
        valmin=0,
        valmax=rot_ds.shape[2]-1,
        valinit=init_x,
        valstep=1,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        #socket.send(json.dumps(['x_slice', val]).encode())
        socket.send_string("x_slice %d" % val)
        im_x.set_data(conc_contours_x[:,:,val])
        fig.canvas.draw_idle()
        #plt.draw()

    # Change the transparency for each individual element, especially the legend had to be made transparent partswise
    # Contour plot does not have Artist, hence call Artist of im_c.collections
    def contour(event):
        global contour_show
        if not contour_show:
            #socket.send(json.dumps(['top_view', 0.3]).encode())
            socket.send_string("top_view %d" % 1)
            contour_show = True
        else:
            #socket.send(json.dumps(['top_view', 0]).encode())
            socket.send_string("top_view %d" % 0)
            contour_show = False
    contourax = fig.add_axes([0.6, 0.025, 0.1, 0.04])
    contour_button = Button(contourax, 'Contour', hovercolor='0.900')

    # register the update function with each slider
    x_slider.on_changed(update)
    contour_button.on_clicked(contour)


    # register the update function with each slider
    x_slider.on_changed(update)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes('right', size='5%', pad=0.10)
    cbar = plt.colorbar(im_x, cax=cax)
    cbar.ax.get_yaxis().set_ticks([])
    for i, label in enumerate(['Zoet water', 'Zout water']):
        cbar.ax.text(3.5, (3 + i * 6) / 8, label, ha='center', va='center')


    plt.show()
    #print("matplotlib socket", socket)



def slider_window():

    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    fig, axes = plt.subplots()

    x_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    x_slider = Slider(
        ax=x_ax,
        label='x',
        valmin=0,
        valmax=6,
        valinit=1,
        valstep=1,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        socket.send(str(val).encode())
        fig.canvas.draw_idle()
        #plt.draw()
    
    plt.show()



@click.command()
def main(args=None):
    """Console script for vcl."""

    contextr = zmq.Context()
    socketr = contextr.socket(zmq.SUB)
    socketr.connect("tcp://localhost:5556")
    socketr.setsockopt(zmq.SUBSCRIBE, b'')

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)
    executor.submit(satellite_window)
    # executor.submit(mayavi_window)
    executor.submit(opencv_window)
    #executor.submit(slider_window)
    executor.submit(contour_slice_window)


    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    print('socket', socket)

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
