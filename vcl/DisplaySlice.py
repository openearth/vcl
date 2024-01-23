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


class DisplaySlice:
    def __init__(
        self, contour_data, slice_extent, plt_lims, cmap, init_x=0, transform=None
    ) -> None:
        super(DisplaySlice, self).__init__()

        self.fig, self.ax = plt.subplots()
        self.ax.fill_between(
            [slice_extent[0], slice_extent[1]],
            y1=slice_extent[2],
            y2=0,
            facecolor="#255070",
            zorder=0,
        )
        im = self.ax.imshow(
            contour_data[..., init_x], extent=slice_extent, aspect="auto", cmap=cmap
        )

        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        cbar = plt.colorbar(im, cax=cax)
        cbar.ax.get_yaxis().set_ticks([])
        for i, label in enumerate(["Zoet water", "Zout water"]):
            cbar.ax.text(3.5, (3 + i * 6) / 8, label, ha="center", va="center")

        self.ax.set_ylim(-100, 25)
        self.ax.set_xlim(plt_lims[1], plt_lims[3])

        self.fig.tight_layout()
        plt.axis("off")
        plt.show(block=False)

        self.current_x_contour = init_x
        self.current_x_line = init_x
        self.current_contour = im
        self.current_data = contour_data
        self.current_line = None
        self.current_y = None

    def imshow(self, data, transform=None, **kwargs):
        im = self.ax.imshow(data[..., self.current_x], **kwargs)

        if transform is not None:
            im.set_transform(transform + self.transform)

        self.current_contour = im
        self.current_data = data

        return im

    def line_plot(self, x_data, y_data, **kwargs):
        # import ipdb

        # ipdb.set_trace()
        (line,) = self.ax.plot(x_data, y_data[..., self.current_x_line], **kwargs)
        self.current_line = line
        self.current_y = y_data
        return (line,)

    def change_layer(self, data=None, transform=None, **kwargs):
        if self.current_layer is not None:
            self.current_layer.remove()

        if data is not None:
            self.imshow(data, transform=transform, **kwargs)
        else:
            self.current_layer = None

    def change_contour_data(self, data, **kwargs):
        if self.current_contour is not None:
            self.current_contour.remove()

        if data is not None:
            self.imshow(data, **kwargs)
        else:
            self.current_contour = None

    def change_line_data(self, x_data=None, y_data=None, **kwargs):
        if self.current_line is not None:
            self.current_line.remove()

        if y_data is not None:
            self.line_plot(x_data, y_data, **kwargs)
        else:
            self.current_line = None

    def change_slice(self, x_contour, x_line):
        if self.current_contour is not None:
            self.current_contour.set_data(self.current_data[..., x_contour])
        if self.current_line is not None:
            self.current_line.set_ydata(self.current_y[..., x_line])

        self.current_x_contour = x_contour
        self.current_x_line = x_line
