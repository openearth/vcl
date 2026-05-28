"""
Map display window module for the Virtual Climate Lab.

This module provides the DisplayMap class for interactive map visualization with
features including multiple layers, zoom/pan, flow arrows, animations, hand tracking,
and slider navigation.

Classes:
    DisplayMap: Interactive map display window with advanced visualization features.
"""

from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pygame

from vcl.utils import ArrowManager, pygame_utils
from vcl.windows import PygameWindow


def wrap_text(text, font, max_width):
    """Return a list of lines where each line fits within max_width."""
    words = text.split(" ")
    lines = []
    current = ""

    for w in words:
        test = w if current == "" else current + " " + w
        if font.size(test)[0] <= max_width:
            current = test
        else:
            if current:  # push current line
                lines.append(current)
            current = w  # start new line with this word

            # If a single word is longer than max_width, hard-split it:
            while font.size(current)[0] > max_width:
                # find largest prefix that fits
                for i in range(1, len(current) + 1):
                    if font.size(current[:i])[0] > max_width:
                        lines.append(current[: i - 1] + "-")
                        current = current[i - 1 :]
                        break

    if current:
        lines.append(current)
    return lines


def draw_textbox(
    surface,
    text,
    font,
    box_rect,
    color=(240, 240, 240),
    bg=(30, 30, 30),
    padding=10,
    line_spacing=4,
):
    # background
    pygame.draw.rect(surface, bg, box_rect, border_radius=8)

    inner = box_rect.inflate(-2 * padding, -2 * padding)
    lines = wrap_text(text, font, inner.width)

    line_h = font.get_linesize()
    y = inner.top

    # clip so text doesn't draw outside the box
    old_clip = surface.get_clip()
    surface.set_clip(box_rect)

    for line in lines:
        if y + line_h > inner.bottom:
            break
        img = font.render(line, True, color)
        surface.blit(img, (inner.left, y))
        y += line_h + line_spacing

    surface.set_clip(old_clip)


class DisplayMap(PygameWindow.PygameWindow):
    """
    Interactive map display window with zoom, pan, and animation capabilities.

    Extends PygameWindow to provide a feature-rich map visualization interface including:
    - Multiple data layers with transparency control
    - Slider navigation with visual indicator
    - Flow field visualization using arrows
    - Zoom and pan capabilities
    - Animation playback
    - Hand tracking interaction
    - Mask layer overlay
    - Scenario and year selection

    Attributes:
        flow_data (dict): Flow field data with velocity components (ucx, ucy).
        animation_data (dict): Animation frames and associated text for each layer.
        i (int): Current index position on the timeline.
        i_max (int): Maximum timeline index.
        overlays (list): List of active overlay layer names.
        mask_layer (str): Name of the mask layer.
        show_mask (bool): Whether mask layer is currently visible.
        zoom_start_pos (tuple): Starting position for zoom rectangle selection.
        zoom_end_pos (tuple): Ending position for zoom rectangle selection.
        zoom_rect (pygame.Rect): Current zoom region.
        zoomed_in (bool): Whether currently in zoomed view.
        pan_start_pos (tuple): Starting position for pan operation.
        pan_offset (pygame.Vector2): Current pan offset.
        total_pan_offset (pygame.Vector2): Accumulated pan offset.
        bottom_text (str): Text to display at bottom of screen.
        hand_tracking (bool): Whether hand tracking is active.
        hand_tracking_coords (tuple): Normalized coordinates (0-1) of tracked hand.
        show_flows (bool): Whether flow arrows are displayed.
        arrow_manager (ArrowManager): Manager for flow arrow visualization.
        show_animation (bool): Whether animation is playing.
        animation_frame (int): Current animation frame index.
    """

    def __init__(
        self,
        datasets: dict,
        dataset_kwargs: dict,
        flow_data: Optional[dict] = None,
        bg_layer: Optional[str] = None,
        animations_data: Optional[dict] = None,
        mask_layer: Optional[str] = None,
        start_year: Optional[str] = "",
        scenarios: Optional[List[str]] = ["Ref"],
        i_max: Optional[int] = 300,
        screen_size: Optional[Tuple[int, int]] = (800, 600),
        aspect_ratio: Optional[float] = 2 / 1,
    ):
        """
        Initialize a DisplayMap window.

        Args:
            datasets: Dictionary of datasets keyed by year/time period.
            dataset_kwargs: Configuration for each dataset layer.
            flow_data: Flow field data with face_x, face_y, ucx, ucy arrays.
            bg_layer: Name of the background layer to display.
            animations_data: Optional animation data with frames and text.
            mask_layer: Optional name of mask layer for overlay.
            start_year: Initial year to display.
            scenarios: List of scenario names.
            i_max: Maximum slider index (300 default).
            screen_size: Initial window dimensions.
            aspect_ratio: Target aspect ratio for map display.
        """
        super().__init__(
            datasets=datasets,
            dataset_kwargs=dataset_kwargs,
            bg_layer=bg_layer,
            start_year=start_year,
            scenarios=scenarios,
            screen_size=screen_size,
            aspect_ratio=aspect_ratio,
        )

        self.flow_data = flow_data
        self.animation_data = animations_data
        self.animation_data["1983"] = [
            {
                "frame": self.datasets[self.current_year]["1983_on"],
            },
            {
                "frame": self.datasets[self.current_year]["1983_off"],
            },
        ]
        self.animation_frame = 0
        self.animation_update_time = pygame.time.get_ticks()

        self.i = 0
        self.i_max = i_max
        self.overlays = []
        self.mask_layer = mask_layer
        self.show_mask = False

        self.zoom_start_pos = None
        self.zoom_end_pos = None
        self.zoom_rect = None
        self.zoomed_in = False
        self.pan_start_pos = None
        self.pan_offset = pygame.Vector2(0, 0)
        self.total_pan_offset = pygame.Vector2(0, 0)
        self.current_scenario = "none"
        self.current_measure = "contamination"

        self.bottom_text = None
        self.hand_tracking = None

        self.panel_current = self.panels["contamination"]

    def draw_line(self):
        """
        Draw the slider indicator line at current position.

        Renders a vertical line across the map at position determined by
        self.i / self.i_max, providing visual feedback of slider position.
        """

        pygame.draw.line(
            self.screen,
            (37, 80, 112, 200),
            (self.i / self.i_max * self.screen_width, self.y_pos),
            (self.i / self.i_max * self.screen_width, self.img_height + self.y_pos),
            4,
        )
        pygame.draw.line(
            self.screen,
            (255, 255, 255),
            (int(self.i / self.i_max * self.screen_width), self.y_pos),
            (
                int(self.i / self.i_max * self.screen_width),
                self.img_height + self.y_pos,
            ),
            2,
        )

    def init_arrowmanager(self, t):
        """
        Initialize the arrow manager for flow visualization at time t.

        Sets up the ArrowManager with flow field data at the specified time index.

        Args:
            t: Time index for flow data selection.
        """
        try:
            self.t = int(t)
            current_flow_data = {
                "face_x": self.flow_data["face_x"],
                "face_y": self.flow_data["face_y"],
                "ucx": self.flow_data["ucx"][self.t],
                "ucy": self.flow_data["ucy"][self.t],
                "points": np.vstack(
                    (self.flow_data["face_x"], self.flow_data["face_y"])
                ).T,
            }
            self.arrow_manager, self.min_max_data = (
                ArrowManager.initialize_arrow_manager(
                    current_flow_data, (self.img_width, self.img_height)
                )
            )
            self.show_flows = True

        except (ValueError, TypeError) as e:
            print(e)
            self.show_flows = False

    def zoom_surface(self):
        """
        Handle zoom and pan operations on the map surface.

        Manages three interaction states:
        1. Zoomed in with panning: Allows dragging to pan the zoomed view
        2. Drawing zoom rectangle: Shows blue outline while selecting region
        3. Zoomed in: Displays scaled version of selected region

        Clamps panning to prevent viewing outside the map boundaries.
        """
        # Panning logic while mouse is held down and zoomed in
        if self.zoomed_in and pygame.mouse.get_pressed()[0] and self.pan_start_pos:
            current_pos = pygame.mouse.get_pos()

            # Calculate the delta movement of the mouse
            delta_x = self.pan_start_pos[0] - current_pos[0]
            delta_y = self.pan_start_pos[1] - current_pos[1]

            # Calculate the potential new position of the zoom_rect
            new_x = self.zoom_rect.x + delta_x
            new_y = self.zoom_rect.y + delta_y

            # Define the content boundaries on the screen (where the map is drawn)
            # Assuming self.img_width/height are the dimensions of the map image drawn to self.screen
            map_end_x = self.x_pos + self.img_width
            map_end_y = self.y_pos + self.img_height

            # --- CORRECTED CLAMPING LOGIC ---

            # Upper Bound: zoom_rect.y must be >= self.y_pos
            # Lower Bound: zoom_rect.y + zoom_rect.height must be <= map_end_y
            max_y = map_end_y - self.zoom_rect.height
            new_y = max(self.y_pos, min(new_y, max_y))

            # Left Bound: zoom_rect.x must be >= self.x_pos
            # Right Bound: zoom_rect.x + zoom_rect.width must be <= map_end_x
            max_x = map_end_x - self.zoom_rect.width
            new_x = max(self.x_pos, min(new_x, max_x))

            # --- End of Corrected Clamping Logic ---

            # Update the zoom_rect with the clamped position
            self.zoom_rect.x = new_x
            self.zoom_rect.y = new_y

            # Update the pan_start_pos for the next frame
            self.pan_start_pos = current_pos

        # If a zoom rectangle has been defined, perform the zoom (rest of the method is fine)
        if self.zoomed_in and self.zoom_rect:
            # Get the sub-surface from the main screen
            # Use screen.copy() to prevent issues with blitting to a part of itself
            zoomed_surface = self.screen.subsurface(self.zoom_rect).copy()

            # Scale the sub-surface to the full screen size
            scaled_surface = pygame.transform.scale(
                zoomed_surface, (self.img_width, self.img_height)
            )

            # Blit the scaled surface to the main screen
            self.screen.blit(
                scaled_surface,
                (self.x_pos, self.y_pos),
            )

        # If the user is currently drawing the rectangle, draw it
        if self.zoom_start_pos and not self.zoom_rect:
            current_pos = pygame.mouse.get_pos()
            draw_rect = pygame.Rect(
                min(self.zoom_start_pos[0], current_pos[0]),
                min(self.zoom_start_pos[1], current_pos[1]),
                abs(current_pos[0] - self.zoom_start_pos[0]),
                abs(current_pos[1] - self.zoom_start_pos[1]),
            )
            pygame.draw.rect(self.screen, (0, 0, 255), draw_rect, 2)  # Blue outline

    def draw_layer(self, layer):
        """
        Render a specific data layer to the screen.

        Implements surface caching to avoid redundant scaling operations.
        Handles both scenario-based and year-based layer variations.
        Renders colorbar for CMAP-type layers if enabled.

        Args:
            layer: Name of the layer to render.
        """
        if layer in self.dataset_kwargs:

            layer_data = self.dataset_kwargs[layer]
            # ... (determine source_surface) ...
            if self.dataset_kwargs[layer]["variation"] == "scenario":
                source_surface = self.surfaces[self.current_year][
                    self.current_scenario
                ][layer]
            else:
                source_surface = self.surfaces[self.current_year][layer]

            current_dims = (self.img_width, self.img_height)
            # 1. Scaling Check
            cached_scaled_surface = layer_data.get("_cached_scaled_surface")
            cached_scale_dims = layer_data.get("_cached_scale_dims")
            cached_source_surface = layer_data.get("_cached_source_surface")

            if (
                cached_scaled_surface is None
                or cached_source_surface is not source_surface
                or cached_scale_dims != current_dims
            ):
                scaled_surface = pygame.transform.scale(source_surface, current_dims)
                # Store cache
                layer_data["_cached_scaled_surface"] = scaled_surface
                layer_data["_cached_scale_dims"] = current_dims
                layer_data["_cached_source_surface"] = source_surface

                colorbar_rect = pygame.Rect(
                    self.x_pos + 0.65 * self.img_width,
                    self.y_pos + 0.8 * self.img_height,
                    0.3 * self.img_width,
                    0.05 * self.img_height,
                )
                layer_data["_cached_colorbar_rect"] = colorbar_rect
            else:
                scaled_surface = cached_scaled_surface
                colorbar_rect = layer_data.get("_cached_colorbar_rect")

            self.current_surface = scaled_surface
            alpha = self.dataset_kwargs[layer].get("alpha", 1)
            alpha = int(alpha * 255)
            scaled_surface.set_alpha(alpha)
            self.screen.blit(scaled_surface, (self.x_pos, self.y_pos))

            if (
                self.dataset_kwargs[layer]["type"] == "CMAP"
                and self.dataset_kwargs[layer]["cbar"] == True
            ):
                cmap = self.dataset_kwargs[layer]["cmap"]
                norm = self.dataset_kwargs[layer].get("norm", None)
                if norm:
                    value_range = (norm.vmin, norm.vmax)
                else:
                    value_range = np.nanpercentile(
                        self.datasets[self.current_year][layer], [0.05, 0.95]
                    )
                cbar = self.dataset_kwargs[layer].get("cbar", None)
                if cbar is None:
                    cbar = pygame_utils.ContinuousColorBar(
                        rect=colorbar_rect,
                        widget_size=(self.img_width, self.img_height),
                        cmap=cmap,
                        value_range=value_range,
                        orientation="horizontal",
                        font=pygame.font.Font(None, 24),
                    )
                    self.dataset_kwargs[layer]["cbar"] = cbar

                cbar.set_rect(colorbar_rect, (self.img_width, self.img_height))
                cbar.draw(self.screen, (self.x_pos, self.y_pos))

    def draw_text(self):
        """
        Draw scenario, year, and layer text overlays on the map.

        Renders text at bottom-right showing current scenario and year.
        Displays layer-specific text at top-left if current layer is defined.
        """
        if len(self.scenarios) > 1:
            scenario_text = f"Scenario {self.current_scenario}"
        else:
            scenario_text = ""
        if self.current_year != "" and len(self.scenarios) > 1:
            year_text = f", {self.current_year}"
        else:
            year_text = self.current_year

        if self.bottom_text is None:
            text = scenario_text + year_text

        else:
            text = self.bottom_text

        text = str(self.index_to_year(min_year=1911, max_year=2250))

        text_bottom = self.font.render(
            text,
            True,
            (0, 0, 0),
        )
        text_bottom_rect = text_bottom.get_rect(
            bottomleft=(
                self.x_pos + 10,
                self.y_pos + self.img_height - 10,
            )
        )

        self.screen.blit(text_bottom, text_bottom_rect)

        if self.current_layer in self.dataset_kwargs:
            text_top = self.font.render(
                self.dataset_kwargs[self.current_layer]["text"],
                True,
                self.dataset_kwargs[self.current_layer]["text_color"],
            )
            self.screen.blit(text_top, (self.x_pos + 10, self.y_pos + 10))

    def display_mask(self):
        """
        Toggle the visibility of the mask layer.
        """
        if self.mask_layer in self.dataset_kwargs:
            self.show_mask = not self.show_mask

    def change_layer(self, layer):
        """
        Change the currently displayed layer.

        Special handling for "animation" layer to trigger animation playback.
        Stops animation if switching to a different layer.

        Args:
            layer: Name of the layer to display, or "animation" to play animation.
        """
        if layer == "animation":
            # self.current_layer = ""
            if self.current_layer in self.animation_data:
                self.play_animation()
            # self.current_layer = ""
        else:
            if self.show_animation:
                self.play_animation()
            self.current_layer = layer

    def display_overlay(self, overlay):
        """
        Toggle an overlay layer on or off.

        Args:
            overlay: Name of the overlay layer to toggle.
        """
        if overlay in self.overlays:
            self.overlays.remove(overlay)
        else:
            self.overlays.append(overlay)

    def change_line_index(self, i):
        """
        Change the slider position indicator.

        Accepts both absolute indices and normalized values (0-1).
        Clamps the result to valid range [0, i_max].

        Args:
            i: New index value. If >= 1, treated as absolute index.
               If 0 <= i < 1, treated as normalized position.
        """
        if isinstance(i, float):
            if i >= 1:
                i = int(i)
            elif i >= 0 and i < 1:
                i = int(i * self.i_max)
        self.i = np.clip(i, 0, self.i_max)

    def draw_animation_frame(self, alpha=255):
        """
        Update and render the current animation frame.

        Advances animation frames at 1 second intervals, with a 3 second pause
        on the final frame. Loops back to start after completion.
        Renders the frame image and associated text overlay.
        """
        frame_surface = self.create_pygame_surface_from_rgb(
            self.animation_data[self.current_measure][self.i - 72]["frame"]
        )
        frame_surface = pygame.transform.scale(
            frame_surface, (self.img_width, self.img_height)
        )
        frame_surface.set_alpha(alpha)
        self.screen.blit(frame_surface, (self.x_pos, self.y_pos))

        frame_text = self.animation_data[self.current_measure][self.i - 72]["text"]
        self.bottom_text = frame_text

    def change_year(self, year):
        """
        Change the displayed year/time period.

        Args:
            year: Year key that exists in the datasets dictionary.
        """
        if year in self.datasets.keys():
            self.current_year = year

    def change_scenario(self, scenario):
        self.current_scenario = scenario
        new_panel_name = f"{self.current_scenario}_{self.current_measure}"
        self.start_panel_transition(new_panel_name)

    def change_measure(self, measure):
        self.current_measure = measure
        new_panel_name = f"{self.current_scenario}_{self.current_measure}"
        self.start_panel_transition(new_panel_name)

    def index_to_year(self, min_year, max_year):
        year = min_year + int((max_year - min_year) * self.i / self.i_max)
        return year

    def change_alpha(self, layer, alpha):
        """
        Change the transparency of a layer.

        Args:
            layer: Name of the layer to modify.
            alpha: Alpha value in range [0, 1], where 0 is fully transparent.
        """
        alpha = np.clip(alpha, a_min=0, a_max=1)
        if layer in self.dataset_kwargs:
            self.dataset_kwargs[layer]["alpha"] = alpha

    def play_animation(self):
        """
        Toggle animation playback on or off.

        Initializes animation from frame 0 if starting, or stops if already playing.
        """
        self.animation_update_time = pygame.time.get_ticks()
        if not self.show_animation:
            self.show_animation = True
            self.animation_frame = 0
        else:
            self.show_animation = False
            self.bottom_text = None

    def update_animation(
        self, frame_time=50, final_frame_time=3000, show_text=False, animation_data=None
    ):
        """
        Update and render the current animation frame.

        Advances animation frames at 1 second intervals, with a 3 second pause
        on the final frame. Loops back to start after completion.
        Renders the frame image and associated text overlay.
        """
        now = pygame.time.get_ticks()
        if animation_data is None:
            animation_data = self.animation_data[self.current_layer]

        if self.animation_frame == len(animation_data) - 1:
            frame_time = final_frame_time
        if now - self.animation_update_time > frame_time:
            self.animation_frame = (self.animation_frame + 1) % len(animation_data)
            self.animation_update_time = now

        frame_surface = self.create_pygame_surface_from_rgb(
            animation_data[self.animation_frame]["frame"]
        )
        frame_surface = pygame.transform.scale(
            frame_surface, (self.img_width, self.img_height)
        )
        self.screen.blit(frame_surface, (self.x_pos, self.y_pos))

        if show_text:
            frame_text = animation_data[self.animation_frame]["text"]
            self.bottom_text = frame_text

    def start_panel_transition(self, new_panel_name):
        if new_panel_name in self.panels:
            new_panel = self.panels[new_panel_name]
            super().start_panel_transition(new_panel)

    def start_hand_tracking(self, coords):
        """
        Enable hand tracking visualization at specified coordinates.

        Args:
            coords: Tuple (x, y) with normalized coordinates in range [0, 1].
                   Disables tracking if coordinates are out of bounds.
        """
        self.hand_tracking = True
        self.hand_tracking_coords = coords
        if coords[0] < 0 or coords[0] > 1 or coords[1] < 0 or coords[1] > 1:
            self.hand_tracking = False

    def draw_hand_tracking(self):
        """
        Draw hand tracking visualization showing depth at tracked position.

        Samples bathymetry data at the tracked hand position and displays
        depth value in a text box with leader line.
        """
        xpos, ypos = self.hand_tracking_coords

        xpos_depth = int(xpos * self.datasets[self.current_year]["bathymetry"].shape[1])
        ypos_depth = int(ypos * self.datasets[self.current_year]["bathymetry"].shape[0])
        depth = self.datasets[self.current_year]["bathymetry"][ypos_depth, xpos_depth]

        xpos = int(xpos * self.img_width + self.x_pos)
        ypos = int(ypos * self.img_height + self.y_pos)

        font = pygame.font.Font(None, 48)
        self.draw_textbox(point=(xpos, ypos), text=f"Depth: {depth:.2f}m", font=font)

    def draw_layers(self):
        """
        Main rendering method - orchestrates all drawing operations.

        Handles fullscreen mode, clears screen, renders all active layers,
        draws flow arrows, mask layer, slider line, text overlays,
        zoom/pan effects, and hand tracking. Updates display at 60 FPS.
        """
        self.go_fullscreen()
        self.adjust_aspect_ratio()
        self.screen.fill((0, 0, 0))

        year = self.index_to_year(min_year=1911, max_year=2250)

        if self.bg_layer in self.dataset_kwargs:
            self.draw_layer(self.bg_layer)
        if self.current_layer in self.dataset_kwargs and not self.show_animation:
            self.draw_layer(self.current_layer)
        if self.show_animation:
            self.update_animation()
        # if self.current_layer in self.animation_data:
        #     self.draw_animation_frame()
        if year >= 1983 and year < 2006:
            rect_surface = pygame.Surface(
                (self.img_width, self.img_height), pygame.SRCALPHA
            )
            pygame.draw.rect(
                rect_surface,
                (255, 255, 255, 180),  # 128 = 50% transparency
                (0, 0, self.img_width, self.img_height),
            )
            self.screen.blit(rect_surface, (self.x_pos, self.y_pos))
            self.update_animation(
                frame_time=1000,
                final_frame_time=1000,
                animation_data=self.animation_data["1983"],
            )
            self.draw_info_panel(
                colour=(255, 255, 255),
                border_colour=(0, 0, 0),
                screen_ratio=1 / 8,
                position="right",
            )
            self.draw_info_panel(
                colour=(255, 255, 255),
                border_colour=(0, 0, 0),
                screen_ratio=1 / 8,
                position="right",
                image=self.panels["1983 - 2005"],
            )

        elif year >= 2022 and self.current_scenario == "none":
            # self.current_scenario = "contamination"
            rect_surface = pygame.Surface(
                (self.img_width, self.img_height), pygame.SRCALPHA
            )
            pygame.draw.rect(
                rect_surface,
                (255, 255, 255, 180),  # 128 = 50% transparency
                (0, 0, self.img_width, self.img_height),
            )
            self.screen.blit(rect_surface, (self.x_pos, self.y_pos))
            self.draw_layer("2022_lines")
            self.draw_animation_frame()

            self.draw_info_panel(
                colour=(255, 255, 255),
                border_colour=(0, 0, 0),
                screen_ratio=1 / 8,
                position="right",
            )

            self.draw_info_panel_r(
                self.clock.tick(60) / 1000,
                screen_ratio=1 / 8,
                position="right",
            )
        elif year >= 2022 and self.current_scenario != "none":
            self.draw_animation_frame(alpha=100)

            self.draw_layer(self.current_layer)
            self.draw_layer(self.current_scenario)

            self.draw_info_panel_r(
                self.clock.tick(60) / 1000,
                screen_ratio=1 / 8,
                position="right",
            )
        elif year < 1983 or (year >= 2006 and year < 2022):
            self.draw_info_panel(
                colour=(255, 255, 255),
                border_colour=(0, 0, 0),
                screen_ratio=1 / 8,
                position="right",
            )
            if self.current_layer in self.panels:
                self.draw_info_panel(
                    colour=(255, 255, 255),
                    border_colour=(0, 0, 0),
                    screen_ratio=1 / 8,
                    position="right",
                    image=self.panels[self.current_layer],
                )

        for overlay in self.overlays:
            self.draw_layer(overlay)

        if self.show_flows:
            self.arrow_manager.update_and_draw(
                self.screen, 0.1, (self.x_pos, self.y_pos)
            )

        if self.show_mask:
            self.draw_layer(self.mask_layer)

        self.draw_info_panel(
            colour=(255, 255, 255),
            border_colour=(0, 0, 0),
            screen_ratio=1 / 10,
            position="bottom",
        )
        # self.draw_info_panel(
        #     colour=(255, 255, 255),
        #     border_colour=(0, 0, 0),
        #     screen_ratio=1 / 8,
        #     position="right",
        #     image=self.panels["none_treat_water"],
        # )

        # self.draw_line()
        # pygame.draw.line(
        #     self.screen,
        #     (0, 0, 0),
        #     (self.x_pos, 0.80 * self.img_height + self.y_pos),
        #     (self.img_width, 0.80 * self.img_height + self.y_pos),
        #     2,
        # )

        self.draw_text()

        self.zoom_surface()

        if self.hand_tracking:
            self.draw_hand_tracking()

        pygame.display.flip()
        self.clock.tick(60)

    def go_fullscreen(self):
        """
        Handle fullscreen toggle with arrow manager reinitialization.

        Extends parent fullscreen behavior to reinitialize arrow manager
        when screen dimensions change, ensuring arrows scale correctly.
        """
        super().go_fullscreen()

        if (
            self.show_flows
            and (self.screen_width, self.screen_height) != self.screen.get_size()
        ):
            self.adjust_aspect_ratio()
            if self.flow_data:
                self.init_arrowmanager(self.t)
