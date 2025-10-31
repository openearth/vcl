from typing import Optional, Tuple, Union

import matplotlib as mpl
import matplotlib.colors
import numpy as np
import pygame
import pywinctl as gw

# import pygetwindow as gw
from matplotlib.colors import Normalize
from screeninfo import get_monitors


class PygameWindow:
    def __init__(
        self,
        datasets: dict,
        dataset_kwargs: dict,
        bg_layer: str = None,
        start_year="",
        scenarios=["Ref"],
        screen_size: Optional[Tuple[int, int]] = (800, 600),
        aspect_ratio: Optional[float] = 2 / 1,
    ):
        pygame.init()
        self.datasets = datasets
        self.dataset_kwargs = self.supplement_dataset_kwargs(dataset_kwargs)
        self.current_year = start_year
        self.current_scenario = scenarios[0]
        self.scenarios = scenarios

        self.screen_width, self.screen_height = screen_size
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.surfaces = self.prepare_surface_dict()
        self.convert_dataset_to_surfaces()

        self.current_year = next(iter(self.datasets.keys()), None)
        if not bg_layer:
            bg_layer = next(iter(self.dataset_kwargs.keys()))
        self.bg_layer = bg_layer
        self.aspect_ratio = aspect_ratio
        self.current_layer = None
        self.show_flows = False
        self.show_animation = False

        self.adjust_aspect_ratio()
        self.font = pygame.font.Font(None, 96)

        pygame.mixer.init()

    def prepare_surface_dict(self):
        surfaces = {}

        # Check if the structure is nested (keyed by year/group)
        first_key = next(iter(self.datasets.keys()), None)
        is_nested = first_key is not None and isinstance(self.datasets[first_key], dict)

        if is_nested:
            # Nested structure: Use actual group keys (e.g., '2023', '2100')
            years = self.datasets.keys()
        else:
            # Flat structure: Use the defined constant placeholder key
            years = [self.current_year]

        for group_key in years:
            surfaces[group_key] = {}
            for scenario in self.scenarios:
                surfaces[group_key][scenario] = {}
        return surfaces

    def supplement_dataset_kwargs(self, dataset_kwargs: dict):
        default_kwargs = {
            "type": "RGB",
            "variation": "year",
            "text": "",
            "text_color": (0, 0, 0),
        }
        for layer, kwargs_dict in dataset_kwargs.items():
            dataset_kwargs[layer] = default_kwargs | kwargs_dict

        return dataset_kwargs

    def convert_dataset_to_surfaces(self):
        for layer in self.dataset_kwargs.keys():
            for year, dataset_dict in self.datasets.items():
                if self.dataset_kwargs[layer]["variation"] == "scenario":
                    for scenario in self.scenarios:
                        if self.dataset_kwargs[layer]["type"] == "RGB":
                            array = dataset_dict[layer]
                            self.surfaces[year][scenario][layer] = (
                                self.create_pygame_surface_from_rgb(array)
                            )
                else:
                    if self.dataset_kwargs[layer]["type"] == "RGB":
                        array = dataset_dict[layer]
                        self.surfaces[year][layer] = (
                            self.create_pygame_surface_from_rgb(array)
                        )
                    elif self.dataset_kwargs[layer]["type"] == "CMAP":
                        array = dataset_dict[layer]
                        norm = self.dataset_kwargs[layer].get("norm", None)
                        cmap = self.dataset_kwargs[layer]["cmap"]
                        self.surfaces[year][layer] = (
                            self.create_pygame_surface_from_cmap(
                                array=array, cmap=cmap, norm=norm
                            )
                        )

    def create_pygame_surface_from_rgb(self, array: np.ndarray):
        array_height, array_width = array.shape[:2]

        # Detect NaNs (assumes NaNs are consistent across channels)
        nan_mask = np.isnan(array[..., 0])

        # Replace NaNs in RGB with 0
        array = np.nan_to_num(array, nan=0)

        # Ensure array is uint8
        array = array.astype(np.uint8)

        # Add alpha channel if missing
        if array.shape[-1] == 3:
            alpha_channel = np.where(nan_mask, 0, 255).astype(np.uint8)
            array = np.concatenate([array, alpha_channel[..., None]], axis=-1)

        # Create surface and assign RGB
        surface_native = pygame.Surface((array_width, array_height), pygame.SRCALPHA)
        pixels = pygame.surfarray.pixels3d(surface_native)
        pixels[:, :, :] = np.transpose(array[..., :3], (1, 0, 2))

        # Assign alpha
        pixels_alpha = pygame.surfarray.pixels_alpha(surface_native)
        pixels_alpha[:, :] = np.transpose(array[..., 3], (1, 0))

        del pixels
        del pixels_alpha

        return surface_native

    def create_pygame_surface_from_cmap(
        self,
        array: np.ndarray,
        cmap: Union[matplotlib.colors.Colormap, str] = None,
        norm=None,
    ):
        if cmap is None:
            cmap = mpl.colormaps.get_cmap("viridis")
        if isinstance(cmap, str):
            cmap = mpl.colormaps.get_cmap(cmap)
        if norm is None:
            # vmin, vmax = np.nanpercentile(array, [0, 1])
            vmin, vmax = np.nanmin(array), np.nanmax(array)
            vmax = 0
            vmin = -1000
            norm = Normalize(vmin=vmin, vmax=vmax)

        rgba_array = np.zeros((array.shape[0], array.shape[1], 4), dtype=np.uint8)

        nan_mask = np.isnan(array)
        normalized_array = norm(array)
        normalized_array_no_nan = normalized_array[~nan_mask]
        rgba_array[~nan_mask] = (cmap(normalized_array_no_nan) * 255).astype(np.uint8)
        rgba_array[nan_mask] = [0, 0, 0, 0]

        surface = pygame.image.frombuffer(
            rgba_array.tobytes(), rgba_array.shape[1::-1], "RGBA"
        )
        surface = surface.convert_alpha()

        return surface

    def adjust_aspect_ratio(self):
        self.screen_width, self.screen_height = self.screen.get_size()
        # Calculate the new image dimensions based on screen size
        if self.screen_width / self.screen_height > self.aspect_ratio:
            # Screen is wider than 2:1, so the height will be limited by the screen height.
            # The image width will be twice the screen height to maintain the 2:1 ratio.
            new_image_height = self.screen_height
            new_image_width = int(new_image_height * self.aspect_ratio)
        else:
            # Screen is taller than 2:1, so the width will be limited by the screen width.
            # The image height will be half the screen width to maintain the 2:1 ratio.
            new_image_width = self.screen_width
            new_image_height = int(new_image_width / self.aspect_ratio)

        self.img_width = new_image_width
        self.img_height = new_image_height

        self.x_pos = (self.screen_width - new_image_width) // 2
        self.y_pos = (self.screen_height - new_image_height) // 2

    def go_fullscreen(self):
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_f:
                    self.is_fullscreen = not self.is_fullscreen
                    window = gw.getActiveWindow()
                    if self.is_fullscreen:
                        monitors = get_monitors()
                        current_monitor = None
                        for monitor in monitors:
                            # Check if the window's center is within this monitor's bounds
                            if (
                                monitor.x <= window.center.x < monitor.x + monitor.width
                                and monitor.y
                                <= window.center.y
                                < monitor.y + monitor.height
                            ):
                                current_monitor = monitor
                                break

                        if current_monitor:
                            # Go to borderless mode with the new full-screen size
                            self.screen = pygame.display.set_mode(
                                (current_monitor.width, current_monitor.height),
                                pygame.NOFRAME,
                            )

                            # Manually reposition the new borderless window to the top-left of the current monitor
                            window.moveTo(current_monitor.x, current_monitor.y)

                    else:
                        # Go back to a regular windowed mode
                        # Define your desired windowed size
                        windowed_width, windowed_height = (
                            self.screen_width,
                            self.screen_height,
                        )

                        self.screen = pygame.display.set_mode(
                            (windowed_width, windowed_height), pygame.RESIZABLE
                        )

                        # Reposition the window to the center of the current monitor
                        monitors = get_monitors()
                        current_monitor = None
                        for monitor in monitors:
                            if (
                                monitor.x <= window.center.x < monitor.x + monitor.width
                                and monitor.y
                                <= window.center.y
                                < monitor.y + monitor.height
                            ):
                                current_monitor = monitor
                                break

                        if current_monitor:
                            # Calculate the new position to center the window on the monitor
                            new_x = (
                                current_monitor.x
                                + (current_monitor.width - windowed_width) // 2
                            )
                            new_y = (
                                current_monitor.y
                                + (current_monitor.height - windowed_height) // 2
                            )
                            window.moveTo(new_x, new_y)

            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self.zoomed_in:
                        self.pan_start_pos = pygame.mouse.get_pos()
                    else:
                        self.zoom_start_pos = event.pos
                        self.zoom_rect = None  # Reset the zoom rectangle

            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    if not self.zoomed_in and self.zoom_start_pos:
                        self.zoom_end_pos = event.pos
                        # Create the rectangle from the start and end points
                        left = min(self.zoom_start_pos[0], self.zoom_end_pos[0])
                        top = min(self.zoom_start_pos[1], self.zoom_end_pos[1])
                        width = abs(self.zoom_start_pos[0] - self.zoom_end_pos[0])
                        height = abs(self.zoom_start_pos[1] - self.zoom_end_pos[1])
                        self.zoom_rect = pygame.Rect(left, top, width, height)
                        self.zoomed_in = True
                        self.zoom_start_pos = None

                elif event.button == 3:
                    self.zoomed_in = False
                    self.zoom_rect = None
                    self.total_pan_offset = pygame.Vector2(0, 0)
