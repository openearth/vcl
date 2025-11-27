import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

matplotlib.rcParams["toolbar"] = "None"


class StatsWindow:
    def __init__(
        self,
        datasets,
        dataset_kwargs,
        overlay_layers: list = [],
        layers_to_ignore: list = [],
    ):
        self.datasets = datasets
        self.dataset_kwargs = dataset_kwargs
        self.dataset_kwargs = self.supplement_dataset_kwargs(dataset_kwargs)
        self.overlay_layers = overlay_layers
        self.layers_to_ignore = layers_to_ignore

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

                dataset_kwargs[layer][plot_type] = default_kwargs | kwargs_dict

        return dataset_kwargs

    def plot_start_screen(self):
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
        data = self.datasets[layer]["piechart"]
        kwargs = self.dataset_kwargs[layer].get("piechart", {})

        title_text = kwargs.pop("title", "Pie Chart")

        ax.pie(data, **kwargs)
        ax.set_title(title_text)

    def plot_histogram(self, layer, ax):
        data = self.datasets[layer]["histogram"]
        kwargs = self.dataset_kwargs[layer].get("histogram", {})

        title_text = kwargs.pop("title", "Histogram")

        sns.histplot(data, ax=ax, **kwargs)
        ax.set_title(title_text)

    def plot_image(self, img, ax):
        # img = self.datasets[layer]["image"]
        kwargs = self.dataset_kwargs[self.current_layer].get("image", {})

        title_text = kwargs.pop("title", "")

        ax.imshow(img, **kwargs)
        ax.set_title(title_text)
        ax.set_axis_off()

    def plot_layer(self):
        for ax in self.fig.get_axes():
            ax.remove()
        n_plots = len(self.datasets[self.current_layer])

        gs = plt.GridSpec(1, n_plots, figure=self.fig, hspace=0.3, wspace=0.3)

        self.axes = []
        for i in range(n_plots):
            ax = self.fig.add_subplot(gs[0, i])
            self.axes.append(ax)

        for ax, (plot_type, data) in zip(self.axes, self.datasets[self.current_layer]):
            if plot_type == "piechart":
                self.plot_piechart(self.current_layer, ax)
            elif plot_type == "histogram":
                self.plot_histogram(self.current_layer, ax)
            elif plot_type == "image":
                self.plot_image(data, ax)

        self.fig.set_facecolor("white")
        self.fig.tight_layout()
        self.fig.canvas.draw()

    def change_layer(self, layer):
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
