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

colors = [(0, "firebrick"), (0.5, "white"), (1, "royalblue")]
mycmap = LinearSegmentedColormap.from_list("my_colormap", colors)

colors = [
    (0, "firebrick"),
    (0.33, "royalblue"),
    (0.66, cmocean.cm.haline(0)),
    (1, cmocean.cm.haline(0.99)),
]
cbar_cmap = LinearSegmentedColormap.from_list("cbar_colormap", colors)


class DisplaySlice:
    def __init__(
        self,
        dataset,
        kwargs_dict,
        start_layer,
        start_year,
        start_scenario,
        slice_extent,
        plt_lims,
        scenario_layers=[],
        init_x=0,
        transform=None,
    ) -> None:
        super(DisplaySlice, self).__init__()

        self.dataset = dataset
        self.kwargs_dict = kwargs_dict
        self.current_year = start_year
        self.current_scenario = start_scenario
        self.scenario_layers = scenario_layers

        self.ref_year = start_year
        self.ref_scenario = start_scenario

        self.fig, self.ax = plt.subplots()
        self.ax.fill_between(
            [slice_extent[0], slice_extent[1]],
            y1=slice_extent[2],
            y2=0,
            facecolor="#255070",
            zorder=0,
        )

        if start_layer in scenario_layers:
            im = self.ax.imshow(
                self.dataset[self.current_year][self.current_scenario][start_layer][
                    ..., init_x
                ],
                **self.kwargs_dict[start_layer],
            )
        else:
            im = self.ax.imshow(
                self.dataset[self.current_year][start_layer][..., init_x],
                **self.kwargs_dict[start_layer],
            )

        divider = make_axes_locatable(self.ax)
        cax = divider.append_axes("right", size="5%", pad=0.10)
        # cbar = plt.colorbar(im, cax=cax)

        cbar = matplotlib.colorbar.ColorbarBase(
            ax=cax,
            cmap=cbar_cmap,
            orientation="vertical",
            boundaries=[-0.25, 0.25, 0.5, 0.75, 1.25],
        )
        cbar.ax.get_yaxis().set_ticks([])
        ymin, ymax = cbar.ax.get_ylim()
        labels = ["Zoet naar zout", "Zout naar zoet", "Zoet water", "Zout water"]
        for i, label in enumerate(labels):
            pos = (1 / 2 + i) / len(labels)
            cbar.ax.text(
                3.5, ymin + pos * (ymax - ymin), label, ha="center", va="center"
            )

        self.ax.set_ylim(-120, 30)
        self.ax.set_xlim(plt_lims[1] + 2000, plt_lims[3])

        self.fig.tight_layout()
        plt.axis("off")
        plt.show(block=False)

        self.current_x_contour = init_x
        self.current_x_line = init_x
        self.current_contour = im
        self.current_contour_text = start_layer
        if start_layer in scenario_layers:
            self.current_data = self.dataset[self.current_year][self.current_scenario][
                start_layer
            ]
        else:
            self.current_data = self.dataset[self.current_year][start_layer]
        self.current_line = None
        self.current_line_text = None
        self.current_y = None
        self.current_overlay = None

    def imshow(self, data, transform=None, **kwargs):
        self.current_x_contour = np.clip(
            self.current_x_contour, 0, data.shape[-1] - 1
        ).astype(np.int32)
        im = self.ax.imshow(data[..., self.current_x_contour], **kwargs)

        if transform is not None:
            im.set_transform(transform + self.transform)

        return im

    def line_plot(self, x_data, y_data, **kwargs):
        self.current_x_line = np.clip(
            self.current_x_line, 0, y_data.shape[-1] - 1
        ).astype(np.int32)

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

    def change_overlay(self, layer=None, **kwargs):
        if self.current_overlay is not None:
            self.current_overlay.remove()
        if layer is not None:
            im = self.imshow(layer, **kwargs)
            self.current_overlay = im
            self.current_overlay_text = layer
        else:
            self.current_overlay = None
            self.current_overlay_text = None

    def show_difference(self, **kwargs):
        diff = (
            self.dataset[self.ref_year][self.ref_scenario][self.current_contour_text]
            - self.dataset[self.current_year][self.current_scenario][
                self.current_contour_text
            ]
        )

        diff = diff.astype(np.float32)
        self.current_data_overlay = diff
        self.overlay_alpha = np.where(diff == 0, 0, 1).astype(np.float32)

        kwargs = self.kwargs_dict[self.current_contour_text]

        kwargs_extra = {
            "alpha": self.overlay_alpha[..., self.current_x_contour],
            "vmin": -2.5,
            "vmax": 2.5,
            "zorder": 2,
            "cmap": mycmap,
        }
        kwargs = kwargs | kwargs_extra
        self.change_overlay(diff, **kwargs)
        # im = self.imshow(diff, **kwargs)

        # return im
        # TODO: Incorporate change_overlay in here, to be able to deal with slider updates, maybe not needed actually

    def change_contour_data(self, layer, **kwargs):
        if layer is not None:
            if layer in self.scenario_layers:
                if (
                    self.current_year == self.ref_year
                    or self.current_scenario == self.ref_scenario
                ):
                    if self.current_contour is not None:
                        self.current_contour.remove()

                    im = self.imshow(
                        self.dataset[self.current_year][self.current_scenario][layer],
                        **self.kwargs_dict[layer],
                    )

                    self.current_contour = im
                    self.current_contour_text = layer
                    self.current_data = self.dataset[self.current_year][
                        self.current_scenario
                    ][layer]

                    self.change_overlay()

                else:
                    self.show_difference()

                # self.current_data = data
            else:
                self.imshow(
                    self.dataset[self.current_year][layer], **self.kwargs_dict[layer]
                )
                self.current_contour_text = layer
        else:
            self.current_contour = None
            self.current_contour_text = None

    def change_line_data(self, y_data=None):
        if self.current_line is not None:
            self.current_line.remove()

        if y_data is not None:
            if y_data in self.scenario_layers:
                self.line_plot(
                    self.kwargs_dict[y_data]["x"],
                    self.dataset[self.current_year][self.current_scenario][y_data],
                    **{
                        key: val
                        for key, val in self.kwargs_dict[y_data].items()
                        if key != "x"
                    },
                )
                self.current_line_text = y_data
            else:
                self.line_plot(
                    self.kwargs_dict[y_data]["x"],
                    self.dataset[self.current_year][y_data],
                    **{
                        key: val
                        for key, val in self.kwargs_dict[y_data].items()
                        if key != "x"
                    },
                )
                self.current_line_text = y_data
        else:
            self.current_line = None
            self.current_line_text = None

    def change_slice(self, x_contour, x_line):
        if self.current_contour is not None:
            x_contour = np.clip(x_contour, 0, self.current_data.shape[-1] - 1).astype(
                np.int32
            )
            self.current_contour.set_data(self.current_data[..., x_contour])
        if self.current_line is not None:
            x_line = np.clip(x_line, 0, self.current_y.shape[-1] - 1).astype(np.int32)
            self.current_line.set_ydata(self.current_y[..., x_line])
        if self.current_overlay is not None:
            x_contour_overlay = np.clip(
                x_contour, 0, self.current_data.shape[-1] - 1
            ).astype(np.int32)
            self.current_overlay.set_data(
                self.current_data_overlay[..., x_contour_overlay]
            )
            self.current_overlay.set_alpha(self.overlay_alpha[..., x_contour_overlay])
        # TODO: add overlay and alpha in here

        self.current_x_contour = x_contour
        self.current_x_line = x_line

    def change_scenario(self, scenario):
        self.current_scenario = scenario
        self.change_line_data(self.current_line_text)
        self.change_contour_data(self.current_contour_text)

    def change_year(self, year):
        # import ipdb

        # ipdb.set_trace()
        try:
            self.current_year = year
            self.change_line_data(self.current_line_text)
            self.change_contour_data(self.current_contour_text)
        except Exception as e:
            # import ipdb

            # ipdb.set_trace()
            # import sys

            # print(f"Error on line {sys.exc_info()[-1].tb_lineno}: {str(e)}")
            print(e)
