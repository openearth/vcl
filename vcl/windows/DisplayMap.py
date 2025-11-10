from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pygame

from vcl.utils import ArrowManager, pygame_utils
from vcl.windows import PygameWindow


class DisplayMap(PygameWindow.PygameWindow):
    def __init__(
        self,
        datasets: dict,
        flow_data,
        dataset_kwargs: dict,
        bg_layer: str,
        animations_data: dict = None,
        mask_layer: str = None,
        start_year="",
        scenarios=["Ref"],
        i_max: int = 300,
        screen_size: Optional[Tuple[int, int]] = (800, 600),
        aspect_ratio: Optional[float] = 2 / 1,
    ):
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
        self.i = 0
        self.i_max = i_max
        self.mask_layer = mask_layer
        self.show_mask = False

        self.zoom_start_pos = None
        self.zoom_end_pos = None
        self.zoom_rect = None
        self.zoomed_in = False
        self.pan_start_pos = None
        self.pan_offset = pygame.Vector2(0, 0)
        self.total_pan_offset = pygame.Vector2(0, 0)

    def draw_line(self):

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
            sound_path = Path(__file__).parent.parent / "sounds"
            if self.t == 170:
                pygame.mixer.music.load(sound_path / "high-tide.mp3")
                pygame.mixer.music.set_volume(0.1)
            else:
                pygame.mixer.music.load(sound_path / "low-tide.mp3")
                pygame.mixer.music.set_volume(0.3)
            pygame.mixer.music.play(-1)

        except (ValueError, TypeError):
            self.show_flows = False
            pygame.mixer.music.stop()

    def zoom_surface(self):
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

    def draw_text(self, text: str = None):
        if len(self.scenarios) > 1:
            scenario_text = f"Scenario {self.current_scenario}"
        else:
            scenario_text = ""
        if self.current_year != "" and len(self.scenarios) > 1:
            year_text = f", {self.current_year}"
        else:
            year_text = self.current_year

        if text is None:
            text = scenario_text + year_text
        text_bottom = self.font.render(
            text,
            True,
            (0, 0, 0),
        )
        text_bottom_rect = text_bottom.get_rect(
            bottomright=(
                self.img_width + self.x_pos - 10,
                self.img_height + self.y_pos - 10,
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
        if self.mask_layer in self.dataset_kwargs:
            self.show_mask = not self.show_mask

    def change_layer(self, layer):
        if layer == "animation":
            # self.current_layer = ""
            if self.current_layer in self.animation_data:
                self.play_animation()
            # self.current_layer = ""
        else:
            if self.show_animation:
                self.play_animation()
            self.current_layer = layer

    def change_line_index(self, i):
        self.i = np.clip(i, 0, self.i_max)

    def change_year(self, year):
        if year in self.datasets.keys():
            self.current_year = year

    def play_animation(self):
        self.animation_update_time = pygame.time.get_ticks()
        if not self.show_animation:
            self.show_animation = True
            self.animation_frame = 0
        else:
            self.show_animation = False

    def update_animation(self):
        now = pygame.time.get_ticks()
        if now - self.animation_update_time > 1000:
            self.animation_frame = (self.animation_frame + 1) % len(
                self.animation_data[self.current_layer]
            )
            self.animation_update_time = now

        frame_surface = self.create_pygame_surface_from_rgb(
            self.animation_data[self.current_layer][self.animation_frame]["frame"]
        )
        frame_surface = pygame.transform.scale(
            frame_surface, (self.img_width, self.img_height)
        )
        self.screen.blit(frame_surface, (self.x_pos, self.y_pos))

        frame_text = self.animation_data[self.current_layer][self.animation_frame][
            "text"
        ]
        self.draw_text(frame_text)

    def draw_layers(self):
        self.go_fullscreen()
        self.adjust_aspect_ratio()
        self.screen.fill((0, 0, 0))
        if self.bg_layer in self.dataset_kwargs:
            self.draw_layer(self.bg_layer)
        if self.current_layer in self.dataset_kwargs and not self.show_animation:
            self.draw_layer(self.current_layer)
        if self.show_animation:
            self.update_animation()

        if self.show_flows:
            self.arrow_manager.update_and_draw(
                self.screen, 0.1, (self.x_pos, self.y_pos)
            )

        if self.show_mask:
            self.draw_layer(self.mask_layer)

        self.draw_line()

        if not self.show_animation:
            self.draw_text()

        self.zoom_surface()

        pygame.display.flip()
        self.clock.tick(60)

    def go_fullscreen(self):
        super().go_fullscreen()

        if (
            self.show_flows
            and (self.screen_width, self.screen_height) != self.screen.get_size()
        ):
            self.adjust_aspect_ratio()
            if self.flow_data:
                self.init_arrowmanager(self.t)
