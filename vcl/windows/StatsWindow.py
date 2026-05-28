"""
Statistics window module for the Virtual Climate Lab.

This module provides the StatsWindow class for displaying statistical visualizations
using matplotlib, including pie charts, histograms, and images in a multi-panel layout.

Classes:
    StatsWindow: Window for displaying statistics and plots using matplotlib.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation
import numpy as np
import seaborn as sns

matplotlib.rcParams["toolbar"] = "None"


class StatsWindow:
    """
    Window for displaying statistical visualizations using matplotlib.

    Provides a flexible plotting interface that can display combinations of
    pie charts, histograms, and images for different data layers. Supports
    overlay layers that can be toggled on/off and dynamic layer switching.

    The window uses matplotlib's pyplot interface to create figures with
    multiple subplots arranged horizontally based on the number of plots
    for the current layer.

    Attributes:
        datasets (dict): Dictionary of datasets where each layer contains
                        plot type keys ('piechart', 'histogram', 'image') with data.
        dataset_kwargs (dict): Configuration for each layer's plot types.
        overlay_layers (list): List of layer names that are overlays in map plots.
        layers_to_ignore (list): List of layer names to skip when changing layers.
        current_layer (str): Currently displayed layer name, or None for start screen.
        fig (matplotlib.figure.Figure): The matplotlib figure object.
        axes (list or matplotlib.axes.Axes): List of subplot axes or single axis.
    """

    def __init__(
        self,
        datasets: dict,
        dataset_kwargs: dict,
        overlay_layers: list = [],
        layers_to_ignore: list = [],
    ):
        """
        Initialize a StatsWindow.

        Creates a matplotlib figure with a start screen showing the
        "Virtual Climate Lab" title text.

        Args:
            datasets: Dictionary of datasets where each layer contains plot data.
                     Keys are layer names, values are dicts with plot type keys.
            dataset_kwargs: Configuration for plot rendering (colors, labels, etc.).
            overlay_layers: List of layers that should be handled as overlays.
            layers_to_ignore: List of layers to skip when changing layers.
        """
        self.datasets = datasets
        self.dataset_kwargs = dataset_kwargs
        self.dataset_kwargs = self.supplement_dataset_kwargs(dataset_kwargs)
        self.overlay_layers = overlay_layers
        self.layers_to_ignore = layers_to_ignore
        self._anims = {}

        self.fig, self.axes = plt.subplots(num="Virtual Climate Lab - Info screen")
        self.fig.tight_layout()

        self.current_layer = None

        self.fig.set_facecolor((9 / 255, 11 / 255, 128 / 255))

        # 3. Set the subplot's background to be transparent
        # This ensures the figure's background color shows through
        self.axes.set_facecolor("none")  # Or 'transparent'

        # Or simply:
        # ax.axis('off')

        # 5. Add text to the center of the subplot
        # transform=ax.transAxes means the coordinates are relative to the Axes (0,0 to 1,1)
        self.axes.text(
            0.5,
            0.5,
            "Virtual Climate Lab",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=48,
            color="white",  # Choose a contrasting color for the text
            transform=self.axes.transAxes,
        )  # Important for relative positioning

        self.fig.tight_layout()
        # 7. Display the plot
        plt.axis("off")
        plt.show(block=False)

    def supplement_dataset_kwargs(self, dataset_kwargs):
        """
        Supplement dataset kwargs with default plotting parameters.

        Adds default values for pie charts (autopct, startangle) and
        histograms (bins, color) that can be overridden by user kwargs.

        Args:
            dataset_kwargs: User-provided plotting configuration.

        Returns:
            dict: Complete configuration with defaults merged in.
        """
        default_kwargs = {}
        for layer in self.datasets.keys():
            if layer not in dataset_kwargs:
                dataset_kwargs[layer] = {}
        for layer, kwargs_dict_group in dataset_kwargs.items():
            for plot_type, kwargs_dict in kwargs_dict_group.items():
                if plot_type == "piechart":
                    default_kwargs = {"autopct": "%1.1f%%", "startangle": 90}
                elif plot_type == "histogram":
                    default_kwargs = {"bins": 20, "color": "lightgray"}
                elif plot_type == "image":
                    default_kwargs = {"aspect": "auto"}

                dataset_kwargs[layer][plot_type] = default_kwargs | kwargs_dict

        return dataset_kwargs

    def plot_start_screen(self):
        """
        Display the start screen with "Virtual Climate Lab" title.

        Clears all axes and creates a single subplot with blue background
        and centered white text.
        """
        for ax in self.fig.get_axes():
            ax.remove()

        gs = plt.GridSpec(1, 1, figure=self.fig, hspace=0.3, wspace=0.3)

        self.axes = self.fig.add_subplot(gs[0, 0])

        self.fig.set_facecolor((9 / 255, 11 / 255, 128 / 255))

        # 3. Set the subplot's background to be transparent
        # This ensures the figure's background color shows through
        self.axes.set_facecolor("none")  # Or 'transparent'

        # Or simply:
        # ax.axis('off')

        # 5. Add text to the center of the subplot
        # transform=ax.transAxes means the coordinates are relative to the Axes (0,0 to 1,1)
        self.axes.text(
            0.5,
            0.5,
            "Virtual Climate Lab",
            horizontalalignment="center",
            verticalalignment="center",
            fontsize=48,
            color="white",  # Choose a contrasting color for the text
            transform=self.axes.transAxes,
        )  # Important for relative positioning

        self.fig.tight_layout()
        # 7. Display the plot
        plt.axis("off")
        plt.show(block=False)

    def plot_piechart(self, layer, ax):
        """
        Plot a pie chart for the specified layer on the given axis.

        Args:
            layer: Layer name containing piechart data.
            ax: Matplotlib axis to draw on.
        """
        data = self.datasets[layer]["piechart"]
        kwargs = self.dataset_kwargs[layer].get("piechart", {})

        title_text = kwargs.pop("title", "Pie Chart")

        ax.pie(data, **kwargs)
        ax.set_title(title_text)

    def plot_histogram(self, layer, ax):
        """
        Plot a histogram for the specified layer on the given axis.

        Args:
            layer: Layer name containing histogram data.
            ax: Matplotlib axis to draw on.
        """
        data = self.datasets[layer]["histogram"]
        kwargs = self.dataset_kwargs[layer].get("histogram", {})

        title_text = kwargs.pop("title", "Histogram")

        sns.histplot(data, ax=ax, **kwargs)
        ax.set_title(title_text)

    def plot_image(self, img, ax):
        """
        Display an image on the given axis.

        Args:
            img: Image array to display.
            ax: Matplotlib axis to draw on.
        """
        # img = self.datasets[layer]["image"]
        kwargs = self.dataset_kwargs[self.current_layer].get("image", {})

        title_text = kwargs.pop("title", "")
        kwargs.pop("multiple", None)  # handled in plot_layer for multi-image cases
        kwargs.pop("interval_s", None)

        ax.imshow(img, **kwargs)
        ax.set_title(title_text)
        ax.set_axis_off()
        ax.margins(0)

    def plot_image_sequence(self, frames, ax, *, interval_s=5.0, mode="sequence"):
        """
        Animate a list/array of images on a single axis.

        mode:
        - "sequence": play once, then freeze on last frame
        - "loop": keep looping
        """
        # Normalize frames to a list of arrays
        if isinstance(frames, np.ndarray) and frames.ndim >= 3:
            # could be (N,H,W) or (N,H,W,C)
            if frames.ndim in (3, 4):
                frames_list = [frames[i] for i in range(frames.shape[0])]
            else:
                raise ValueError(f"Unsupported frames array shape: {frames.shape}")
        else:
            frames_list = list(frames)

        if len(frames_list) == 0:
            return

        kwargs = self.dataset_kwargs[self.current_layer].get("image", {}).copy()
        title_text = kwargs.pop("title", "")
        kwargs.pop("multiple", None)  # handled here
        interval_s = float(kwargs.pop("interval_s", interval_s))
        interval_ms = int(interval_s * 1000)

        # Draw first frame
        im = ax.imshow(frames_list[0], **kwargs)
        ax.set_title(title_text)
        ax.set_axis_off()
        ax.margins(0)

        n = len(frames_list)
        loop = mode == "loop"

        def update(i):
            im.set_data(frames_list[i])

            # If playing once: stop right after last frame is drawn
            if (not loop) and i == (n - 1):
                # Stop the timer so it freezes on the last frame
                anim.event_source.stop()
            return (im,)

        anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=range(n),
            interval=interval_ms,
            blit=True,  # safer across backends
            repeat=loop,
        )

        # Keep reference alive (otherwise matplotlib may GC it)
        self._anims[ax] = anim

    def plot_layer(self):
        """
        Plot all visualizations for the current layer.

        If multiple images exist and dataset_kwargs[layer]["image"]["multiple"] is:
        - "panel": show as multiple subplots (current behavior)
        - "sequence": animate once on one subplot, then freeze
        - "loop": animate continuously on one subplot
        """
        # Remove old axes
        for ax in self.fig.get_axes():
            ax.remove()

        # Stop/forget old animations (optional but clean)
        if hasattr(self, "_anims"):
            self._anims.clear()

        layer_items = list(
            self.datasets[self.current_layer]
        )  # list of (plot_type, data)

        # Determine multiple-image behavior for this layer
        image_kwargs = self.dataset_kwargs[self.current_layer].get("image", {})
        multiple_mode = image_kwargs.get(
            "multiple", "panel"
        )  # "panel" | "sequence" | "loop"

        # Collect all image entries
        image_indices = [
            i for i, (plot_type, _) in enumerate(layer_items) if plot_type == "image"
        ]

        # If >1 image and mode is sequence/loop, collapse them into one animated entry
        if len(image_indices) > 1 and multiple_mode in ("sequence", "loop"):
            frames = [layer_items[i][1] for i in image_indices]

            # Build a new list where the first image becomes an animated sequence, others removed
            new_items = []
            first_image_idx = image_indices[0]
            for i, item in enumerate(layer_items):
                if i == first_image_idx:
                    new_items.append(("image_sequence", frames))
                elif i in image_indices:
                    continue
                else:
                    new_items.append(item)
            layer_items = new_items

        n_plots = len(layer_items)

        # Create grid
        gs = plt.GridSpec(1, n_plots, figure=self.fig, wspace=0.0, hspace=0.0)

        self.axes = []
        for i in range(n_plots):
            ax = self.fig.add_subplot(gs[0, i])
            self.axes.append(ax)

        # Plot
        for ax, (plot_type, data) in zip(self.axes, layer_items):
            if plot_type == "piechart":
                self.plot_piechart(self.current_layer, ax)

            elif plot_type == "histogram":
                self.plot_histogram(self.current_layer, ax)

            elif plot_type == "image":
                self.plot_image(data, ax)

            elif plot_type == "image_sequence":
                interval_s = image_kwargs.get("interval_s", 5.0)
                self.plot_image_sequence(
                    data, ax, interval_s=interval_s, mode=multiple_mode
                )

        # Layout
        self.fig.set_facecolor("white")
        self.fig.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0, hspace=0)
        self.fig.canvas.draw()

    def change_layer(self, layer):
        """
        Change the displayed layer and update plots accordingly.

        Implements overlay logic where overlay layers can be toggled on/off.
        Non-overlay layers replace the current display.

        Args:
            layer: Name of the layer to switch to or toggle.
        """
        try:
            if (
                layer in self.dataset_kwargs
                and (self.current_layer is None)
                and layer in self.overlay_layers
            ):
                self.current_layer = layer
                self.plot_layer()
            elif layer in self.dataset_kwargs and layer not in self.overlay_layers:
                self.current_layer = layer
                self.plot_layer()
            elif (
                layer in self.dataset_kwargs
                and layer in self.overlay_layers
                and layer == self.current_layer
            ):
                self.current_layer = None
                self.plot_start_screen()
            else:
                if layer in self.layers_to_ignore or layer in self.overlay_layers:
                    return
                self.current_layer = None
                self.plot_start_screen()
        except Exception as e:
            print(e)


# data = {
#     "layer1": {
#         "piechart": [10, 20, 30],
#         "histogram": np.random.randn(1000),
#     },
#     "layer2": {
#         "piechart": [5, 15, 80],
#         "histogram": np.random.randn(500),
#     },
# }

# dataset_kwargs = {
#     "layer1": {
#         "piechart": {"labels": ["A", "B", "C"], "title": "Verdeling van ABC"},
#         "histogram": {"bins": 30, "color": "skyblue"},
#     },
#     "layer2": {
#         "piechart": {"labels": ["X", "Y", "Z"]},
#         "histogram": {"bins": 20, "color": "salmon"},
#     },
# }


# sw = StatsWindow(data, dataset_kwargs)
# # sw.change_layer("layer1")
# i = 0
# while True:
#     plt.pause(0.01)
#     i += 1
#     if i % 1000 == 0:
#         sw.change_layer("layer2" if sw.current_layer == "layer1" else "layer1")
