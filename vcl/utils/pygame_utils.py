import numpy as np
import pygame
from matplotlib.colors import LinearSegmentedColormap, ListedColormap


def draw_continuous_colorbar(
    screen, rect, cmap, value_range, orientation="vertical", num_ticks=5, font=None
):
    """
    Draws a continuous color bar in either vertical or horizontal orientation
    using NumPy and surfarray for performance.

    :param screen: The Pygame surface to draw on.
    :param rect: The pygame.Rect object for the color bar's position and size.
    :param cmap: A Matplotlib LinearSegmentedColormap object.
    :param value_range: A tuple (min_val, max_val) for the data represented.
    :param orientation: "vertical" or "horizontal".
    :param num_ticks: The number of tick labels to generate (including min/max).
    """
    if font is None:
        pygame.font.init()
        font = pygame.font.Font(None, 24)

    bar_width, bar_height = rect.width, rect.height
    min_val, max_val = value_range

    # Determine the primary dimension (along which the gradient flows)
    if orientation == "vertical":
        primary_dim = bar_height
        secondary_dim = bar_width
    elif orientation == "horizontal":
        primary_dim = bar_width
        secondary_dim = bar_height
    else:
        raise ValueError("Orientation must be 'vertical' or 'horizontal'")

    # --- 1. OPTIMIZED GRADIENT DRAWING ---

    # 1. Create a 1D array of indices (0 to 1) for sampling
    # The length of the data array equals the length of the gradient path
    data_indices = np.linspace(0, 1, primary_dim)

    # 2. Apply the colormap (Hx4 float array)
    # [::-1] reverses the array so min_val is at the bottom/left (start of gradient)
    color_data_float = cmap(data_indices)[::-1]

    # 3. Convert to 8-bit RGB integers (Hx3 array)
    color_data_int = (color_data_float[:, :3] * 255).astype(np.uint8)

    # 4. Reshape the array based on orientation
    if orientation == "vertical":
        # Shape: (1, Height, 3) -> Transpose to (Width, Height, 3) for Pygame
        # The color is along the second axis (height)
        gradient_array = color_data_int[:, np.newaxis, :]
        gradient_array_transposed = gradient_array.transpose((1, 0, 2))

    elif orientation == "horizontal":
        # Shape: (Width, 1, 3) -> Transpose to (Width, Height, 3) for Pygame
        # The color is along the first axis (width)

        # Invert the order to ensure 0.0 starts at the left side of the bar
        color_data_int = color_data_int[::-1]

        # Reshape to (Width, 1, 3)
        gradient_array = color_data_int.reshape(primary_dim, 1, 3)

        # Transpose to (Width, Height, 3) where Height is 1 pixel
        # The dimensions are (Width, 1, Color) -> (Width, Height, Color)
        gradient_array_transposed = gradient_array.transpose((0, 1, 2))

        # Need to tile the 1-pixel high data vertically across the bar_height
        # This is the most reliable way to fill the required dimensions:
        # Array shape is (Width, Height=1, 3). Tile it secondary_dim (height) times.
        gradient_array_transposed = np.tile(
            gradient_array_transposed, (1, secondary_dim, 1)
        )

    # 5. Create the surface and blit
    gradient_surface = pygame.surfarray.make_surface(gradient_array_transposed)

    # Note: We don't need pygame.transform.scale if the NumPy array is tiled/shaped correctly,
    # but the shape calculation above ensures the array matches the rect dimensions (bar_width, bar_height)
    screen.blit(gradient_surface, rect.topleft)

    # Draw a border around the bar
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    # --- 2. LABEL GENERATION AND DRAWING ---

    # Generate tick values
    tick_values = np.linspace(min_val, max_val, num_ticks)
    auto_labels = [f"{val:.1f}" for val in tick_values]

    # Relative positions (0.0 is top/left, 1.0 is bottom/right)
    relative_positions = np.linspace(0.0, 1.0, num_ticks)

    # Determine label alignment based on orientation
    if orientation == "vertical":
        label_x_pos = rect.right + 5
        # The gradient was reversed above, so 0.0 must align with the top (0.0 y-offset)
        # and 1.0 must align with the bottom (bar_height y-offset).
        # We need to reverse relative positions again to match the physical screen space.
        positions_y = [rect.y + bar_height * (1.0 - p) for p in relative_positions]
        label_alignments = [(label_x_pos, pos) for pos in positions_y]

    elif orientation == "horizontal":
        label_y_pos = rect.bottom + 5
        # The gradient flows left (0.0) to right (1.0)
        positions_x = [rect.x + bar_width * p for p in relative_positions]
        label_alignments = [(pos, label_y_pos) for pos in positions_x]

    # Draw the calculated labels
    for i, label in enumerate(auto_labels):
        x, y = label_alignments[i]
        text_surface = font.render(label, True, (255, 255, 255))

        if orientation == "vertical":
            # Align midleft for vertical bar labels
            text_rect = text_surface.get_rect(midleft=(x, y))
        elif orientation == "horizontal":
            # Align midtop for horizontal bar labels
            text_rect = text_surface.get_rect(midtop=(x, y))

        screen.blit(text_surface, text_rect)


# Initialize Pygame font once outside the draw function for efficiency
# You should initialize pygame.font.init() in your main program
# We'll define a default font object here for the function's scope
try:
    pygame.font.init()
    GLOBAL_FONT = pygame.font.Font(None, 24)
except pygame.error:
    # Fallback if pygame hasn't been initialized yet
    print("Pygame font not initialized. Using a placeholder.")
    GLOBAL_FONT = None


def draw_colorbar(screen, rect, cmap, value_range=None, labels=None, label_x_pos=None):
    """
    Draws a color bar, supporting both discrete and continuous colormaps.

    :param screen: The Pygame surface to draw on.
    :param rect: The pygame.Rect object for the color bar's position and size.
    :param cmap: A Matplotlib Colormap object (LinearSegmentedColormap/ListedColormap)
                 OR a list of (threshold, color) tuples (legacy discrete).
    :param value_range: A tuple (min_val, max_val) for the data represented by the cmap. Required for continuous auto-labeling.
    :param labels: A list of specific labels (strings) to use for discrete or custom continuous ticks. Overrides auto-labeling.
    :param label_x_pos: The x-coordinate for the labels.
    """
    global GLOBAL_FONT
    if GLOBAL_FONT is None:
        # Re-initialize font if needed (though it should be done in main program)
        try:
            GLOBAL_FONT = pygame.font.Font(None, 24)
        except:
            return  # Cannot proceed without a font

    bar_width, bar_height = rect.width, rect.height

    # --- 1. Draw the Color Bar Body ---

    # Check if the colormap is continuous (LinearSegmentedColormap or similar)
    if isinstance(cmap, LinearSegmentedColormap) or (
        hasattr(cmap, "colorbar_extent") and not isinstance(cmap, ListedColormap)
    ):
        # **CONTINUOUS COLOR BAR LOGIC**
        NUM_STEPS = bar_height  # Use one step per pixel for a smooth gradient

        for i in range(NUM_STEPS):
            # Sample the colormap from 0.0 (bottom) to 1.0 (top)
            t = i / (NUM_STEPS - 1)

            # Get the color (returns an RGBA tuple in 0-1 range)
            color_rgba = cmap(t)

            # Convert to Pygame's (R, G, B) 0-255 format
            pygame_color = (np.array(color_rgba[:3]) * 255).astype(np.uint8)

            # Draw from bottom (i=0) upwards.
            segment_height = bar_height / NUM_STEPS
            segment_y = rect.y + bar_height - (i * segment_height) - segment_height

            segment_rect = pygame.Rect(
                rect.x, segment_y, bar_width, segment_height + 1
            )  # +1 to prevent gaps
            pygame.draw.rect(screen, pygame_color, segment_rect)

        is_continuous = True
    # CONDITION 2: DISCRETE (LISTEDCOLORMAP)
    # This block handles maps that are explicitly discrete, such as those used for categories.
    elif isinstance(cmap, ListedColormap):
        colors = cmap.colors
        num_segments = len(colors)

        for i in range(num_segments):
            threshold_end = (i + 1) / num_segments
            color = colors[i]

            segment_height = bar_height / num_segments
            segment_y = rect.y + bar_height - (threshold_end * bar_height)

            pygame_color = (np.array(color[:3]) * 255).astype(np.uint8)

            segment_rect = pygame.Rect(rect.x, segment_y, bar_width, segment_height)
            pygame.draw.rect(screen, pygame_color, segment_rect)

        is_continuous = False  # Must be False for discrete maps

    elif isinstance(cmap, list):
        # **DISCRETE COLOR BAR LOGIC (Legacy List of tuples)**

        # Sort colors by threshold to ensure correct drawing order
        cmap.sort(key=lambda x: x[0])

        for i in range(len(cmap)):
            # Assuming equal spacing if thresholds aren't explicitly used for height
            threshold_start = i / len(cmap)
            threshold_end = (i + 1) / len(cmap)
            color = cmap[i][1]

            segment_height = (threshold_end - threshold_start) * bar_height
            segment_y = rect.y + bar_height - (threshold_end * bar_height)

            # Convert color from 0-1 range to Pygame's 0-255
            pygame_color = (np.array(color[:3]) * 255).astype(np.uint8)

            segment_rect = pygame.Rect(rect.x, segment_y, bar_width, segment_height)
            pygame.draw.rect(screen, pygame_color, segment_rect)

        is_continuous = False

    else:
        print(f"Unsupported colormap type: {type(cmap)}")
        return

    # Draw a border around the bar
    pygame.draw.rect(screen, (0, 0, 0), rect, 1)

    # --- 2. Label Generation and Drawing ---

    if label_x_pos is None:
        # Default label position
        label_x_pos = rect.right + 5

    if labels is None and is_continuous and value_range:
        # **Auto-Generate Labels for Continuous Maps**
        min_val, max_val = value_range
        num_ticks = 5  # Example: Generate 5 labels (bottom, top, and 3 in between)

        # Create a list of tick values
        tick_values = np.linspace(min_val, max_val, num_ticks)

        # Convert values to strings (formatted to one decimal place)
        auto_labels = [f"{val:.1f}" for val in tick_values]

        # Create a list of corresponding y-positions (0.0 for bottom, 1.0 for top)
        # Note: linspace already gives us evenly spaced values from 0 to 1,
        # so we can use its index to determine the relative position.
        relative_positions = np.linspace(
            1.0, 0.0, num_ticks
        )  # 1.0 is bottom, 0.0 is top

        # Combine labels and positions for drawing
        labels_to_draw = list(zip(auto_labels, relative_positions))

    elif labels is not None:
        # **Use Explicit Labels**

        if is_continuous:
            # For continuous, assume labels correspond to evenly spaced ticks (0.0 to 1.0)
            num_labels = len(labels)
            # Positions run from 1.0 (bottom) to 0.0 (top)
            relative_positions = np.linspace(1.0, 0.0, num_labels)
            labels_to_draw = list(zip(labels, relative_positions))

        else:  # Discrete map
            # For discrete, labels correspond to the center of each segment
            num_labels = len(labels)
            # Positions run from center of bottom segment (1/2) to center of top segment (1 - 1/2*n)
            # We use (i + 0.5) / num_labels to get the center of the i-th block (from bottom)
            relative_positions = [(i + 0.5) / num_labels for i in range(num_labels)]
            # We need to reverse the positions because we calculate from bottom up
            relative_positions.reverse()
            labels_to_draw = list(zip(labels, relative_positions))

    else:
        # No labels requested or necessary data missing
        return

    # Draw the calculated labels
    for label, relative_pos in labels_to_draw:
        text_surface = GLOBAL_FONT.render(label, True, (0, 0, 0))

        # Calculate y-position based on relative position (0.0=top, 1.0=bottom)
        label_y = rect.y + bar_height - (relative_pos * bar_height)

        # Adjust y-position slightly for discrete center labels
        if not is_continuous and num_labels == len(labels_to_draw):
            # Discrete labels should be centered on the segment
            text_rect = text_surface.get_rect(midleft=(label_x_pos, label_y))
        else:
            # Continuous labels should align with the tick mark
            text_rect = text_surface.get_rect(midleft=(label_x_pos, label_y))

        screen.blit(text_surface, text_rect)


# from matplotlib import cm

# # Assuming you have imported pygame and numpy

# # Get a continuous colormap object
# viridis_cmap = cm.get_cmap("viridis")
# data_range = (25.0, 100.0)  # Data goes from 25 to 100

# colorbar_rect = pygame.Rect(50, 50, 30, 500)

# pygame.init()
# screen = pygame.display.set_mode((800, 600), pygame.RESIZABLE)

# while True:
#     draw_colorbar(
#         screen,
#         colorbar_rect,
#         cmap=viridis_cmap,
#         value_range=data_range,
#         label_x_pos=100,  # Position the labels at x=100
#         # labels=None is default, triggering auto-labeling
#     )
#     pygame.display.flip()

import numpy as np
import pygame
from matplotlib.colors import LinearSegmentedColormap

# Initialize Pygame once (assuming this is done outside the function/class)
# pygame.init()


class ContinuousColorBar:
    def __init__(
        self,
        widget_size,
        rect,
        cmap,
        value_range,
        orientation="vertical",
        num_ticks=5,
        font=None,
    ):
        """
        Initializes and caches the static elements of the color bar.
        """
        if font is None:
            pygame.font.init()
            self.font = pygame.font.Font(None, 24)
        else:
            self.font = font

        self.rect = rect
        self.widget_width, self.widget_height = widget_size
        self.cmap = cmap
        self.value_range = value_range
        self.orientation = orientation
        self.num_ticks = num_ticks

        # Caching variables
        self._gradient_surface = None
        self._label_data = []  # Stores (text_surface, text_rect_alignment_data)

        # Pre-calculate the expensive elements immediately
        self._generate_gradient_surface()
        self._generate_labels()
        self._generate_widget_surface()

    def _generate_widget_surface(self):
        widget_surface = pygame.Surface(
            (self.widget_width, self.widget_height), pygame.SRCALPHA
        )
        widget_surface.fill((0, 0, 0, 0))  # Fill with transparent black

        self._generate_gradient_surface()
        self._generate_labels()

        widget_surface.blit(self._gradient_surface, self.rect.topleft)

        # 2. Draw border
        pygame.draw.rect(widget_surface, (0, 0, 0), self.rect, 1)

        # 3. Blit the cached labels
        for text_surface, text_rect in self._label_data:
            widget_surface.blit(text_surface, text_rect)

        self._cached_widget_surface = widget_surface

    def _generate_gradient_surface(self):
        """
        PERFORMANCE CRITICAL: Generates the gradient surface ONCE and caches it.
        This method should only be called if rect, cmap, or orientation changes.
        """
        bar_width, bar_height = self.rect.width, self.rect.height

        if self.orientation == "vertical":
            primary_dim = bar_height
            secondary_dim = bar_width
        elif self.orientation == "horizontal":
            primary_dim = bar_width
            secondary_dim = bar_height
        else:
            raise ValueError("Orientation must be 'vertical' or 'horizontal'")

        # 1. Create 1D array of indices (0 to 1)
        data_indices = np.linspace(0, 1, primary_dim)

        # 2. Apply the colormap (Hx4 float array)
        # Note: Vertical gradient requires reversal for screen coordinates (0,0 is top)
        if self.orientation == "vertical":
            color_data_float = self.cmap(data_indices)[::-1]
        else:  # Horizontal gradient
            color_data_float = self.cmap(data_indices)

        # 3. Convert to 8-bit RGB integers
        color_data_int = (color_data_float[:, :3] * 255).astype(np.uint8)

        # 4. Reshape/Tile the array
        if self.orientation == "vertical":
            # The color is along the height axis. Shape: (1, H, 3) -> (W, H, 3)
            gradient_array = color_data_int[:, np.newaxis, :]
            # Transpose to (Width, Height, 3) for Pygame
            gradient_array_transposed = np.tile(
                gradient_array.transpose((1, 0, 2)), (secondary_dim, 1, 1)
            )

        elif self.orientation == "horizontal":
            # The color is along the width axis. Shape: (W, 1, 3) -> (W, H, 3)
            gradient_array = color_data_int.reshape(primary_dim, 1, 3)
            # Tile vertically across the bar_height
            gradient_array_transposed = np.tile(gradient_array, (1, secondary_dim, 1))

        # 5. Create the surface and cache it
        self._gradient_surface = pygame.surfarray.make_surface(
            gradient_array_transposed
        )

    def _generate_labels(self):
        """
        Generates and caches the text surfaces and their alignment points.
        This should only be called if value_range or num_ticks changes.
        """
        min_val, max_val = self.value_range
        bar_width, bar_height = self.rect.width, self.rect.height

        tick_values = np.linspace(min_val, max_val, self.num_ticks)
        relative_positions = np.linspace(0.0, 1.0, self.num_ticks)
        self._label_data = []  # Reset cache

        for i, val in enumerate(tick_values):
            label = f"{val:.1f}"

            # --- Text Surface Caching ---
            text_surface = self.font.render(label, True, (255, 255, 255))

            # --- Position Calculation ---
            p = relative_positions[i]

            if self.orientation == "vertical":
                # Align midleft. Gradient was reversed, so 0.0 is at the bottom (1.0 screen pos)
                label_x_pos = self.rect.right + 5
                # Screen Y: 0.0 at top (max_val) -> 1.0 at bottom (min_val)
                # Since we want min_val label at the bottom and max_val at top,
                # we map relative position 'p' (0 to 1) to screen position (bottom to top)
                screen_y_pos = self.rect.y + bar_height * (1.0 - p)

                # Get rect for alignment
                text_rect = text_surface.get_rect(midleft=(label_x_pos, screen_y_pos))

            elif self.orientation == "horizontal":
                # Align midtop. Gradient flows left (min_val) to right (max_val)
                label_y_pos = self.rect.bottom + 5
                screen_x_pos = self.rect.x + bar_width * p

                # Get rect for alignment
                text_rect = text_surface.get_rect(midtop=(screen_x_pos, label_y_pos))

            self._label_data.append((text_surface, text_rect))

    def draw(self, screen, pos):
        """
        PERFORMANCE OPTIMIZED: Draws the cached surface and labels.5
        This is the method to call every frame in the Pygame loop.
        """
        if self._cached_widget_surface:
            # Only ONE blit operation in the loop!
            screen.blit(self._cached_widget_surface, pos)

    def set_value_range(self, new_range):
        """
        Updates the value range and only regenerates the labels (cheap part).
        """
        if new_range != self.value_range:
            self.value_range = new_range
            self._generate_labels()  # Labels must be regenerated with new values

    def set_rect(self, new_rect, widget_size):
        """
        Updates the external rect and forces a full widget surface regeneration
        since the size/scale has changed.
        """
        # Compare dimensions for scale change (fast)
        if new_rect.size != self.rect.size:
            self.rect.size = new_rect.size
            self.widget_width, self.widget_height = widget_size
            self._generate_widget_surface()  # Rerun all generation logic


# --- USAGE EXAMPLE ---
# In your setup/initialization phase:
# my_cmap = LinearSegmentedColormap.from_list("custom", ["blue", "red"])
# colorbar_rect = pygame.Rect(50, 50, 20, 300)
# my_colorbar = ContinuousColorBar(
#     rect=colorbar_rect,
#     cmap=my_cmap,
#     value_range=(0.0, 100.0),
#     orientation="vertical"
# )

# In your main game loop:
# screen.fill((50, 50, 50))
# # This is now extremely fast!
# my_colorbar.draw(screen)
# # pygame.display.flip()
