import pygame
import numpy as np

import cmocean
from typing import Tuple, Optional
from vcl.windows import PygameWindow


def draw_colorbar(screen, rect, colors, labels, label_x_pos):
    """
    Draws a segmented color bar with distinct color blocks.
    :param screen: The Pygame surface to draw on.
    :param rect: The pygame.Rect object for the color bar's position and size.
    :param colors: A list of (threshold, color) tuples.
    :param labels: A list of labels for each segment.
    :param label_x_pos: The x-coordinate for the labels.
    """
    bar_width, bar_height = rect.width, rect.height

    # Sort colors by threshold to ensure correct drawing order
    colors.sort(key=lambda x: x[0])

    # Draw each color segment as a solid rectangle
    # The segments are drawn from bottom to top, corresponding to the thresholds
    for i in range(len(colors)):
        threshold_start = i / len(colors)
        threshold_end = (i + 1) / len(colors)
        color = colors[i][1]

        # Calculate the height of this segment in pixels
        segment_height = (threshold_end - threshold_start) * bar_height

        # Calculate the y position of the top of this segment
        segment_y = rect.y + bar_height - (threshold_end * bar_height)

        # Convert the color from matplotlib's 0-1 range to Pygame's 0-255
        pygame_color = (np.array(color[:3]) * 255).astype(np.uint8)

        # Draw the rectangle for the segment
        segment_rect = pygame.Rect(rect.x, segment_y, bar_width, segment_height)
        pygame.draw.rect(screen, pygame_color, segment_rect)

    # Draw the labels next to the color bar
    # The label logic from your original function is still valid here
    screen_width, screen_height = screen.get_size()
    font = pygame.font.Font(None, int(24 / 800 * screen_width))
    label_spacing = bar_height / len(labels)

    for i, label in enumerate(labels):
        text_surface = font.render(label, True, (0, 0, 0))
        # Position the label correctly relative to the segment transitions
        text_rect = text_surface.get_rect(
            midleft=(label_x_pos, rect.y + i * label_spacing + label_spacing / 2)
        )
        screen.blit(text_surface, text_rect)


class DisplaySlice(PygameWindow.PygameWindow):
    def __init__(
        self,
        datasets,
        dataset_kwargs,
        starting_layer="conc",
        screen_size: Optional[Tuple[int, int]] = (800, 600),
    ):
        pygame.init()
        self.datasets = datasets
        self.dataset_kwargs = dataset_kwargs
        self.screen_width, self.screen_height = screen_size
        self.is_fullscreen = False
        self.screen = pygame.display.set_mode(
            (self.screen_width, self.screen_height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.surfaces = {}
        self.current_layer = None
        self.show_flows = False

        self.current_layer = starting_layer
        self.i = 0

    def show_slice(self, screen, data, slice_index, position, size):
        slice_index = np.clip(slice_index, 0, data.shape[-1] - 1)
        slice_2d = data[..., slice_index]
        surface = self.create_pygame_surface_from_cmap(
            array=slice_2d,
            cmap=cmocean.cm.haline,
            norm=self.dataset_kwargs[self.current_layer].get("norm", None),
        )

        scaled_surface = pygame.transform.scale(surface, size)
        screen.blit(scaled_surface, position)

    def initialize_ratios(self):
        self.padding_left = 0.1 * self.screen_width
        self.padding_bottom = 0.05 * self.screen_height
        self.padding_top = 0.05 * self.screen_height

        self.cbar_width = 0.05 * self.screen_width
        self.cbar_label_padding_left = self.screen_width / 800 * 10
        self.cbar_label_padding_right = self.screen_width / 800 * 20

        self.cbar_font = pygame.font.Font(None, int(self.screen_width / 800 * 24))
        self.label_width = max(
            self.cbar_font.size(label)[0]
            for label in self.dataset_kwargs[self.current_layer]["cbar_labels"]
        )

        self.cbar_section_width = (
            self.cbar_width
            + self.cbar_label_padding_left
            + self.label_width
            + self.cbar_label_padding_right
        )

        self.plot_width = (
            self.screen_width - self.cbar_section_width - self.padding_left
        )
        self.plot_height = self.screen_height - self.padding_bottom - self.padding_top
        self.plot_rect = pygame.Rect(
            self.padding_left, self.padding_top, self.plot_width, self.plot_height
        )

    def draw_axes(self):
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (self.padding_left, self.padding_top),
            (self.padding_left, self.padding_top + self.plot_height),
            4,
        )
        pygame.draw.line(
            self.screen,
            (0, 0, 0),
            (self.padding_left, self.padding_top + self.plot_height),
            (self.padding_left + self.plot_width, self.padding_top + self.plot_height),
            4,
        )

        self.draw_xticks()
        self.draw_yticks()

    def draw_yticks(self):
        font = pygame.font.Font(None, int(16 / 800 * self.screen_width))
        ticks = self.dataset_kwargs[self.current_layer]["yticks"]

        if type(ticks[0]) in [int, float]:
            max_tick = max(ticks)
            min_tick = min(ticks)
            for tick in ticks:
                tick_ypos = (
                    self.padding_top
                    + (max_tick - tick) / (max_tick - min_tick) * self.plot_height
                )
                text_surface = font.render(str(tick), True, (0, 0, 0))
                # Position the label correctly relative to the segment transitions
                text_rect = text_surface.get_rect(
                    midright=(self.padding_left - 10 * self.plot_width / 800, tick_ypos)
                )
                self.screen.blit(text_surface, text_rect)
        elif type(ticks[0]) in [tuple, list]:
            for tick in ticks:

                text_surface = font.render(str(tick[1]), True, (0, 0, 0))
                # Position the label correctly relative to the segment transitions
                text_rect = text_surface.get_rect(
                    midright=(
                        self.padding_left + 10 / 800 * self.screen_width,
                        self.padding_top + self.plot_height * tick[0],
                    )
                )
                self.screen.blit(text_surface, text_rect)

    def draw_xticks(self):
        font = pygame.font.Font(None, int(16 / 800 * self.screen_width))
        ticks = self.dataset_kwargs[self.current_layer]["xticks"]

        if type(ticks[0]) in [int, float]:
            max_tick = max(ticks)
            min_tick = min(ticks)
            for tick in ticks:
                tick_xpos = (
                    self.padding_left
                    - (min_tick - tick) / (max_tick - min_tick) * self.plot_width
                )
                text_surface = font.render(str(tick), True, (0, 0, 0))
                # Position the label correctly relative to the segment transitions
                text_rect = text_surface.get_rect(
                    center=(
                        tick_xpos,
                        self.padding_top
                        + self.plot_height
                        + 20 * self.plot_width / 800,
                    )
                )
                self.screen.blit(text_surface, text_rect)
        elif type(ticks[0]) in [tuple, list]:
            for tick in ticks:

                text_surface = font.render(str(tick[1]), True, (0, 0, 0))
                # Position the label correctly relative to the segment transitions
                text_rect = text_surface.get_rect(
                    center=(
                        self.padding_left + self.plot_width * tick[0],
                        self.padding_top
                        + self.plot_height
                        + 20 / 800 * self.plot_width,
                    )
                )
                self.screen.blit(text_surface, text_rect)

    def draw_colorbar(self):
        # Define colorbar position based on calculated widths
        colorbar_x = self.padding_left + self.plot_rect.width + 10
        colorbar_rect = pygame.Rect(
            colorbar_x, self.padding_top, self.cbar_width, self.plot_height
        )

        # Calculate the label's starting x-position
        label_x_pos = colorbar_x + self.cbar_width + self.cbar_label_padding_left

        draw_colorbar(
            self.screen,
            colorbar_rect,
            self.dataset_kwargs[self.current_layer]["cmap"],
            self.dataset_kwargs[self.current_layer]["cbar_labels"],
            label_x_pos,
        )

    def draw_fill_rect(self):
        fill_color = (37, 80, 112)

        fill_rect_x = self.plot_rect.x
        fill_rect_width = self.plot_rect.width
        fill_rect_y = self.padding_top + (25.0 - 0) / (25.5 - (-140)) * self.plot_height
        fill_rect_height = (
            self.screen_height - self.padding_bottom
        ) - fill_rect_y  # * (0 - ymin) / original_plot_height

        pygame.draw.rect(
            self.screen,
            fill_color,
            (fill_rect_x, fill_rect_y, fill_rect_width, fill_rect_height),
        )

    def draw_slice(self):
        self.show_slice(
            self.screen,
            self.datasets[self.current_layer],
            slice_index=self.i,
            position=(self.padding_left, self.padding_top),
            size=(int(self.plot_width), int(self.plot_height)),
        )

    def change_index(self, i):
        self.i = i

    def draw_layers(self):
        self.go_fullscreen()
        self.initialize_ratios()
        self.screen.fill((255, 255, 255))

        self.draw_fill_rect()
        self.draw_axes()
        self.draw_colorbar()
        self.draw_slice()

        pygame.display.flip()

    def go_fullscreen(self):
        super().go_fullscreen()

        self.screen_width, self.screen_height = self.screen.get_size()
