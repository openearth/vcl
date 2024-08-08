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
        start_year,
        start_scenario,
        scenario_layers=[],
        transform=None,
    ) -> None:
        super(DisplayMap, self).__init__()
        # Datasets with all the data to be visualized, is a dictionary of the form
        # 'layer': 'data'
        self.dataset = dataset
        # Dictionary containing kwargs for plotting, is a dictionary of the form
        # 'layer': {kwargs}
        self.kwargs_dict = kwargs_dict
        # Set current year to starting year
        self.current_year = start_year
        # Set scenario to starting scenario
        self.current_scenario = start_scenario
        # List with layer names which have different scenarios
        self.scenario_layers = scenario_layers

        # Initialize figure and axis
        self.fig, self.ax = plt.subplots()
        self.cbar = None
        # Set background color
        self.fig.patch.set_facecolor("black")
        # No margins
        self.fig.tight_layout()
        self.fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

        # No axis
        self.ax.set_axis_off()
        self.ax.set_frame_on(False)

        # Store transform of the axis
        self.transform = self.ax.transData

        # Plot start layer, this layer does not get changed
        im = self.ax.imshow(
            self.dataset[self.current_year][start_layer],
            **kwargs_dict[self.current_year][start_layer],
        )

        # Split scenario text
        scen = self.current_scenario.split("_")
        scen = f"{scen[0]} {scen[1]}"
        # Plot text on figure
        self.ax_text = self.ax.text(
            1,
            0,
            f"{scen}, {self.current_year}",
            transform=self.ax.transAxes,
            fontsize=60,
            verticalalignment="bottom",
            horizontalalignment="right",
            color="white",
            bbox=dict(facecolor="none", edgecolor="none"),
        )

        # Plot layer name on figure
        self.title = self.ax.text(
            0,
            1,
            kwargs_dict[self.current_year][start_layer]["label"],
            transform=self.ax.transAxes,
            fontsize=60,
            verticalalignment="top",
            horizontalalignment="left",
            color="black",
            bbox=dict(facecolor="none", edgecolor="none"),
        )

        # Transform the plot if needed
        if transform is not None:
            im.set_transform(transform + self.transform)

        # Set plot limits
        xmin, ymin, xmax, ymax = plt_lims
        self.ax.set_xlim(xmin, xmax)
        self.ax.set_ylim(ymin, ymax)

        # Current layer (plot) and current layer name
        self.current_layer = None
        self.current_layer_text = None

        # Current overlay (plot) and current overlay name
        self.current_overlay = None
        self.current_overlay_text = None
        plt.show(block=False)

    def imshow(self, data, transform=None, **kwargs):
        # Plot data on the axis with kwargs
        im = self.ax.imshow(data, **kwargs)

        # Transform plot if needed
        if transform is not None:
            im.set_transform(transform + self.transform)

        return im

    def line_plot(self, x_data, y_data, **kwargs):
        # Plot a line with kwargs
        return self.ax.plot(x_data, y_data, **kwargs)

    def contourf(self, X, Y, data, transform=None, **kwargs):
        # Plot data as a filled contour with grid (X, Y) and with kwargs
        im = self.ax.contourf(X, Y, data, **kwargs)

        # Transform plot if needed
        if transform is not None:
            im.set_transform(transform + self.transform)

        return im

    def change_layer(self, layer=None, transform=None, **kwargs):
        # If animation is running, pause the animation and remove the displayed text
        try:
            self.ani.pause()
            scen = self.current_scenario.split("_")
            scen = f"{scen[0]} {scen[1]}"
            self.ax_text.set_text(f"{scen}, {self.current_year}")
            # self.ax_text.remove()
        except:
            pass
        # If a layer is already plotted on the axis, remove it
        if self.current_layer is not None:
            self.current_layer.remove()
            try:
                self.cbar.remove()
            except:
                pass
        # If we pass a new layer name, plot it on the axis
        if layer is not None:
            # If layer is "animation_data", create and show animation
            if layer == "animation_data":
                self.show_animation(layer)
            # Layers which require a colorbar
            elif layer == "GXG" or layer == "floodmap":
                im = self.imshow(
                    self.dataset[self.current_year][layer],
                    **self.kwargs_dict[self.current_year][layer],
                )
                self.current_layer = im
                self.current_layer_text = layer

                self.cbar = self.fig.colorbar(
                    im,
                    cax=self.fig.add_axes([0.6, 0.20, 0.38, 0.05]),
                    orientation="horizontal",
                    shrink=1,
                )
            else:
                # If layer has scenarios, select layer corresponding to the current scenario and current year
                if layer in self.scenario_layers:
                    im = self.imshow(
                        self.dataset[self.current_year][self.current_scenario][layer],
                        **self.kwargs_dict[self.current_year][layer],
                    )
                # Else select layer only corresponding to current year
                else:
                    im = self.imshow(
                        self.dataset[self.current_year][layer],
                        **self.kwargs_dict[self.current_year][layer],
                    )
                # Set current layer and current layer name
                self.current_layer = im
                self.current_layer_text = layer
                self.title.set_text(
                    self.kwargs_dict[self.current_year][self.current_layer_text][
                        "label"
                    ]
                )
        # If we pass no new layer, set current layer and current layer name to None
        else:
            self.current_layer = None
            self.current_layer_text = None
            self.title.set_text("")

    def change_year(self, year):
        # Set current year to year
        self.current_year = year
        # Change layer and overlay with new year
        self.change_layer(self.current_layer_text)
        self.change_overlay(self.current_overlay_text)
        # Update displayed text
        scen = self.current_scenario.split("_")
        scen = f"{scen[0]} {scen[1]}"
        self.ax_text.set_text(f"{scen}, {self.current_year}")

    def change_scenario(self, scenario):
        # Set current scenario to scenario
        self.current_scenario = scenario
        # Change layer and overlay with new scenario
        self.change_layer(self.current_layer_text)
        self.change_overlay(self.current_overlay_text)
        # Update displayed text
        scen = self.current_scenario.split("_")
        scen = f"{scen[0]} {scen[1]}"
        self.ax_text.set_text(f"{scen}, {self.current_year}")

    def change_overlay(self, layer=None, **kwargs):
        # Separate function for 'overlay', which can be plotted on top of layer
        # If an overlay is already plotted on the axis, remove it
        if self.current_overlay is not None:
            self.current_overlay.remove()
        # If we pass a new layer name, plot it on the axis
        if layer is not None:
            # If the layer is 'tidal_flows', use separate function to plot quivers
            if layer.split(":")[0] == "tidal_flows":
                im = self.show_tidal_flows(
                    layer.split(":")[0],
                    layer.split(":")[1],
                    **self.kwargs_dict[self.current_year][layer.split(":")[0]],
                )
                self.current_overlay = im
                self.current_overlay_text = layer
            else:
                # If layer has scenarios, select layer corresponding to the current scenario and current year
                if layer in self.scenario_layers:
                    im = self.imshow(
                        self.dataset[self.current_year][self.current_scenario][layer],
                        **self.kwargs_dict[self.current_year][layer],
                    )
                # Else select layer only corresponding to current year
                else:
                    im = self.imshow(
                        self.dataset[self.current_year][layer],
                        **self.kwargs_dict[self.current_year][layer],
                    )
                # Set current overlay and current overlay name
                self.current_overlay = im
                self.current_overlay_text = layer
        # If we pass no new layer, set current layer and current layer name to None
        else:
            self.current_overlay = None
            self.current_overlay_text = None

    def show_animation(self, layer):
        # Data necessary for creating animation
        animation_data = self.dataset[self.current_year][layer]

        # Frames for the animation
        frames = [animation_data[frame]["image"] for frame in animation_data.keys()]
        # Extent for each frame
        extents = [animation_data[frame]["extent"] for frame in animation_data.keys()]
        # Text (year) for each frame
        texts = [animation_data[frame]["text"] for frame in animation_data.keys()]

        # Kwargs for starting frame
        kwargs_dict = {"extent": extents[0]} | self.kwargs_dict[self.current_year][
            layer
        ]

        # Plot starting frame
        img = self.imshow(frames[0], **kwargs_dict)
        # If no text yet, plot text
        if self.ax_text is None:
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
        # Else, change plottetd text
        else:
            self.ax_text.set_text(texts[0])

        # Function to update the frames in the animation
        def update(frame):
            img.set_data(frames[frame])
            img.set_extent(extents[frame])

            self.ax_text.set_text(texts[frame])
            return img

        # Create animation
        self.ani = animation.FuncAnimation(
            fig=self.fig, func=update, frames=len(frames), interval=150
        )

        # Pause, otherwise only first frame is displayed
        plt.pause(1)

        # Change current layer and current layer text
        self.current_layer = img
        self.current_layer_text = layer
        # plt.show(block=False)

    def show_tidal_flows(self, layer, tide, transform=None, **kwargs):
        dataset = self.dataset[self.current_year][layer]
        # Get subset of u and v vectors for direction of tide
        u = dataset["ucx"][int(tide), :][::4]
        v = dataset["ucy"][int(tide), :][::4]
        # Color depending on norm of the vector
        c = np.sqrt(u**2 + v**2)
        # Create quiver plot
        im = self.ax.quiver(
            dataset["face_x"][::4], dataset["face_y"][::4], u, v, c, **kwargs
        )

        if transform is not None:
            im.set_transform(transform + self.transform)

        return im
