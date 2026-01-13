"""
Base window module for pygame-based visualization in the Virtual Climate Lab.

This module provides the foundational PygameWindow class that handles common
window management tasks such as screen initialization, fullscreen toggling,
dataset-to-surface conversion, aspect ratio management, and zoom/pan interactions.

Classes:
    PygameWindow: Base class for creating pygame windows with dataset visualization.
"""

from typing import Optional, Tuple, Union, List

import matplotlib as mpl
import matplotlib.colors
import numpy as np
import pygame
import pywinctl as gw

# import pygetwindow as gw
from matplotlib.colors import Normalize
from screeninfo import get_monitors


class PygameWindow:
    """
    Base pygame window class for dataset visualization.

    This class provides core functionality for creating pygame-based windows including
    screen management, dataset loading and conversion to pygame surfaces, aspect ratio
    handling, zoom/pan interactions, and fullscreen toggle capabilities.

    Attributes:
        datasets (dict): Dictionary of datasets keyed by year/group.
        dataset_kwargs (dict): Configuration for each dataset layer.
        current_year (str): Currently displayed year/time period.
        current_scenario (str): Currently selected scenario.
        scenarios (list): List of available scenarios.
        screen_width (int): Current screen width in pixels.
        screen_height (int): Current screen height in pixels.
        is_fullscreen (bool): Whether window is in fullscreen mode.
        screen (pygame.Surface): Main pygame display surface.
        clock (pygame.time.Clock): Clock for frame rate control.
        surfaces (dict): Pre-rendered pygame surfaces from datasets.
        bg_layer (str): Name of the background layer.
        aspect_ratio (float): Target aspect ratio for image display.
        current_layer (str): Currently active display layer.
        show_flows (bool): Whether to show flow arrows.
        show_animation (bool): Whether animation is playing.
        font (pygame.font.Font): Font for text rendering.
        img_width (int): Calculated image width based on aspect ratio.
        img_height (int): Calculated image height based on aspect ratio.
        x_pos (int): X position where image should be drawn.
        y_pos (int): Y position where image should be drawn.
        zoomed_in (bool): Whether currently in zoomed view.
        zoom_rect (pygame.Rect): Rectangle defining zoom region.
        zoom_start_pos (tuple): Starting position of zoom selection.
        pan_start_pos (tuple): Starting position for pan operation.
    """

    def __init__(
        self,
        datasets: dict,
        dataset_kwargs: dict,
        bg_layer: Optional[str] = None,
        start_year: Optional[str] = "",
        scenarios: Optional[List[str]] = ["Ref"],
        screen_size: Optional[Tuple[int, int]] = (800, 600),
        aspect_ratio: Optional[float] = 2 / 1,
    ):
        """
        Initialize a new PygameWindow instance.

        Args:
            datasets: Dictionary of datasets, typically keyed by year/time period.
                     Each dataset contains layer data as numpy arrays.
            dataset_kwargs: Configuration dictionary for each dataset layer.
                           Specifies rendering type (RGB/CMAP), colormap, text labels, etc.
            bg_layer: Name of the background layer to display. If None, uses first layer.
            start_year: Initial year/time period to display. Defaults to first available.
            scenarios: List of scenario names (e.g., ['Ref', 'High', 'Low']).
            screen_size: Initial window dimensions (width, height) in pixels.
            aspect_ratio: Target aspect ratio for displayed images.
        """
        pygame.init()
        pygame.display.set_caption("Virtual Climate Lab - Map Screen")
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
        """
        Prepare the surface dictionary structure based on dataset organization.

        Creates a nested dictionary structure for storing pygame surfaces.
        Handles both flat (single year) and nested (multiple years) dataset structures.

        Returns:
            dict: Nested dictionary with structure {year: {scenario: {layer: surface}}}.
        """
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
        """
        Supplement dataset kwargs with default values.

        Merges user-provided layer configuration with default values for
        rendering type, text labels, colormaps, and other display parameters.

        Args:
            dataset_kwargs: User-provided configuration for dataset layers.

        Returns:
            dict: Complete configuration dict with defaults applied.
        """
        default_kwargs = {
            "type": "RGB",
            "variation": "year",
            "text": "",
            "text_color": (0, 0, 0),
            "cmap": None,
            "cbar": False,
        }
        for layer, kwargs_dict in dataset_kwargs.items():
            dataset_kwargs[layer] = default_kwargs | kwargs_dict

        return dataset_kwargs

    def convert_dataset_to_surfaces(self):
        """
        Convert all dataset arrays to pygame surfaces for efficient rendering.

        Iterates through all datasets and layers, converting numpy arrays to
        pre-rendered pygame surfaces using either RGB or colormap-based methods.
        Stores results in self.surfaces for quick blitting during rendering.
        """
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
        """
        Create a pygame surface from an RGB(A) numpy array.

        Handles NaN values by making them transparent. Converts the array to
        uint8 format and creates a pygame surface with proper alpha channel.

        Args:
            array: RGB or RGBA array with shape (height, width, 3 or 4).
                  Values should be in 0-255 range. NaN values become transparent.

        Returns:
            pygame.Surface: Surface with SRCALPHA flag, ready for blitting.
        """
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
        """
        Create a pygame surface from a 2D array using a matplotlib colormap.

        Applies a colormap to a 2D data array, handling normalization and NaN values.
        NaN values are rendered as transparent pixels.

        Args:
            array: 2D numpy array with data values to visualize.
            cmap: Matplotlib colormap object or name string. Defaults to 'viridis'.
            norm: Matplotlib Normalize object for value mapping. If None, uses
                 array's min/max values (with hardcoded override for specific use case).

        Returns:
            pygame.Surface: RGBA surface with colormap applied.
        """
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
        if isinstance(cmap, matplotlib.colors.ListedColormap):
            normalized_array = array.astype(np.int16)
        else:
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
        """
        Adjust image dimensions to maintain target aspect ratio within screen bounds.

        Calculates optimal image dimensions and position to fit the target aspect
        ratio within the current screen size. Updates img_width, img_height, x_pos,
        and y_pos attributes for centered display.
        """
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

    def draw_textbox(self, point, text, font):
        """
        Draw a text box with leader line at the specified point.

        Creates a white box with black border containing text, positioned to avoid
        overlapping with the point. The box is placed in one of four quadrants
        relative to the point, with a connecting line.

        Args:
            point: Tuple (x, y) specifying the point to annotate.
            text: String to display in the text box.
            font: pygame.font.Font object for text rendering.
        """
        text_surface = font.render(text, True, (0, 0, 0))
        text_width, text_height = text_surface.get_size()
        padding = 6
        box_width = text_width + padding * 2
        box_height = text_height + padding * 2

        x, y = point

        if x < self.img_width / 2 and y < self.img_height / 2:
            offset = (10, 10)
            box_x = x + offset[0]
            box_y = y + offset[1]
        elif x > self.img_width / 2 and y < self.img_height / 2:
            offset = (-10, 10)
            box_x = x + offset[0] - box_width
            box_y = y + offset[1]
        elif x < self.img_width / 2 and y > self.img_height / 2:
            offset = (10, -10)
            box_x = x + offset[0]
            box_y = y + offset[1] - box_height
        else:
            offset = (-10, -10)
            box_x = x + offset[0] - box_width
            box_y = y + offset[1] - box_height

        # Draw line from point to box corner
        pygame.draw.line(
            self.screen, (0, 0, 0), point, (x + offset[0], y + offset[1]), 2
        )

        # Draw box
        pygame.draw.rect(
            self.screen, (255, 255, 255), (box_x, box_y, box_width, box_height)
        )
        pygame.draw.rect(
            self.screen, (0, 0, 0), (box_x, box_y, box_width, box_height), 2
        )

        # Draw text
        self.screen.blit(text_surface, (box_x + padding, box_y + padding))

    def go_fullscreen(self):
        """
        Handle fullscreen toggle and mouse interaction events.

        Processes pygame events including:
        - F key: Toggle between fullscreen and windowed mode
        - Left mouse button: Initiate zoom selection or pan (when zoomed)
        - Right mouse button: Reset zoom

        Manages multi-monitor support by detecting the current monitor and
        adjusting window size and position accordingly.
        """
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
