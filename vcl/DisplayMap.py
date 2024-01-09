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
    def __init__(self) -> None:
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

        self.current_layer = None

    def imshow(self, data, transform=None, **kwargs):
        im = self.ax.imshow(data, **kwargs)

        if transform is not None:
            im.set_transform(transform)

        self.current_layer = im

        return im

    def contourf(self, X, Y, data, transform=None, **kwargs):
        im = self.ax.contourf(X, Y, data, **kwargs)

        if transform is not None:
            im.set_transform(transform)

        self.current_layer = im

        return im

    def get_transform(self):
        return self.ax.transData

    def change_layer(self, data, transform=None, **kwargs):
        self.current_layer.remove()

        im = self.imshow(data, **kwargs)

        if transform is not None:
            im.set_transform(transform)

        self.current_layer = im
