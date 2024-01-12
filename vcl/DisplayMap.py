import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zmq
import cmocean
from matplotlib.colors import LightSource, ListedColormap, LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.transforms as mtransforms
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button
import scipy

import vcl.prep_data


class DisplayMap:
    def __init__(self, sat_img, sat_extent, plt_lims, transform=None) -> None:
        super(DisplayMap, self).__init__()
        self.fig, self.ax = plt.subplots()
        # Set background color
        self.fig.patch.set_facecolor("black")
        # No margins
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # No axis
        self.ax.set_axis_off()
        self.ax.set_frame_on(False)
        self.transform = self.ax.transData

        im = self.ax.imshow(sat_img, extent=sat_extent)

        if transform is not None:
            im.set_transform(transform + self.transform)

        xmin, ymin, xmax, ymax = plt_lims
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        self.current_layer = None
        plt.show(block=False)

    def imshow(self, data, transform=None, **kwargs):
        im = self.ax.imshow(data, **kwargs)

        if transform is not None:
            im.set_transform(transform + self.transform)

        self.current_layer = im

        return im

    def line_plot(self, x_data, y_data, **kwargs):
        # self.ax.plot(x_data, y_data, **kwargs)
        return self.ax.plot(x_data, y_data, **kwargs)

    def contourf(self, X, Y, data, transform=None, **kwargs):
        im = self.ax.contourf(X, Y, data, **kwargs)

        if transform is not None:
            im.set_transform(transform + self.transform)

        self.current_layer = im

        return im

    def change_layer(self, data=None, transform=None, **kwargs):
        if self.current_layer is not None:
            self.current_layer.remove()

        if data is not None:
            self.imshow(data, transform=transform, **kwargs)
        else:
            self.current_layer = None
