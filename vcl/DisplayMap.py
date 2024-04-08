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
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import scipy

import vcl.prep_data
import vcl.data


class DisplayMap:
    def __init__(
        self,
        dataset,
        kwargs_dict,
        start_layer,
        plt_lims,
        start_scenario,
        transform=None,
    ) -> None:
        super(DisplayMap, self).__init__()
        self.dataset = dataset
        self.kwargs_dict = kwargs_dict
        self.current_scenario = start_scenario

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

        im = self.ax.imshow(
            self.dataset[self.current_scenario][start_layer], **kwargs_dict[start_layer]
        )

        if transform is not None:
            im.set_transform(transform + self.transform)

        xmin, ymin, xmax, ymax = plt_lims
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        self.current_layer = None
        self.current_layer_text = None
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

    def change_layer(self, layer=None, transform=None, **kwargs):
        try:
            self.ani.pause()
            self.ax_text.remove()
        except:
            pass
        if self.current_layer is not None:
            self.current_layer.remove()
        if layer is not None:
            if layer == "animation_data":
                self.show_animation(layer)
            elif layer.split(":")[0] == "tidal_flows":
                self.show_tidal_flows(
                    layer.split(":")[0],
                    layer.split(":")[1],
                    **self.kwargs_dict[layer.split(":")[0]]
                )
            else:
                self.imshow(
                    self.dataset[self.current_scenario][layer],
                    **self.kwargs_dict[layer]
                )
                self.current_layer_text = layer
        else:
            self.current_layer = None
            self.current_layer_text = None

    def change_scenario(self, scenario):
        self.current_scenario = scenario
        self.change_layer(self.current_layer_text)

    def show_animation(self, layer):
        animation_data = self.dataset[self.current_scenario][layer]

        frames = [animation_data[frame]["image"] for frame in animation_data.keys()]
        extents = [animation_data[frame]["extent"] for frame in animation_data.keys()]
        texts = [animation_data[frame]["text"] for frame in animation_data.keys()]

        kwargs_dict = {"extent": extents[0]} | self.kwargs_dict[layer]

        img = self.imshow(frames[0], **kwargs_dict)
        self.ax_text = self.ax.text(
            1,
            0,
            texts[0],
            transform=self.ax.transAxes,
            fontsize=60,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="white",
            bbox=dict(facecolor="none", edgecolor="none"),
        )

        def update(frame):
            img.set_data(frames[frame])
            img.set_extent(extents[frame])

            self.ax_text.set_text(texts[frame])
            return img

        import ipdb

        ipdb.set_trace()
        self.ani = animation.FuncAnimation(
            fig=self.fig, func=update, frames=len(frames), interval=150
        )

        # Pause, otherwise only first frame is displayed
        plt.pause(1)

        self.current_layer = img
        self.current_layer_text = layer
        # plt.show(block=False)

    def show_tidal_flows(self, layer, tide, transform=None, **kwargs):
        dataset = self.dataset[self.current_scenario][layer]
        u = dataset["ucx"][int(tide), :][::4]
        v = dataset["ucy"][int(tide), :][::4]
        c = np.sqrt(u**2 + v**2)
        im = self.ax.quiver(
            dataset["face_x"][::4], dataset["face_y"][::4], u, v, c, **kwargs
        )

        if transform is not None:
            im.set_transform(transform + self.transform)

        self.current_layer = im
