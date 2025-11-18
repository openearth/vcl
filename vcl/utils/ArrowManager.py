import numpy as np
import pygame
from matplotlib import colormaps
from scipy.spatial import KDTree

# --- Constants ---
SCREEN_WIDTH, SCREEN_HEIGHT = 800, 600
BACKGROUND_COLOR = (20, 20, 30)
FLOW_SPEED = 50
MAX_ARROWS = 1500  # The maximum number of arrows to maintain
ARROW_SCALE = 35
ARROWHEAD_SIZE = 10
SPAWN_THRESHOLD_MAG = 0.02
STAGNANT_MAGNITUDE_THRESHOLD = 0.02
OFFSCREEN_BUFFER = 50
ARROW_LIFETIME_RANGE = (5, 10)  # Min and Max lifetime in seconds


# --- Sample Data Generation ---
def generate_sample_data(num_points=1000):
    np.random.seed(42)
    x_grid = np.linspace(-10, 10, int(np.sqrt(num_points)))
    y_grid = np.linspace(-10, 10, int(np.sqrt(num_points)))
    face_x, face_y = np.meshgrid(x_grid, y_grid)
    face_x = face_x.flatten()
    face_y = face_y.flatten()
    ucx = np.cos(face_x) * np.sin(face_y)
    ucy = -np.sin(face_x) * np.cos(face_y)

    return {"face_x": face_x, "face_y": face_y, "ucx": ucx, "ucy": ucy}


def get_vector_at_position(pos_x, pos_y, dataset, tree):
    dist, idx = tree.query([pos_x, pos_y])
    u, v = dataset["ucx"][idx], dataset["ucy"][idx]
    return u, v


# --- Particle/Arrow Class ---
class Arrow:
    def __init__(
        self, pos, u, v, vector_scale, scale_x, scale_y, color, color_func=None
    ):
        self.pos = np.array(pos, dtype=float)
        self.u = u
        self.v = v
        self.vector_scale = vector_scale
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.color = color
        self.color_func = color_func
        self.image = self.create_arrow_surface()
        self.rect = self.image.get_rect(center=self.pos)
        self.ttl = np.random.uniform(ARROW_LIFETIME_RANGE[0], ARROW_LIFETIME_RANGE[1])
        self.plasma_cmap = colormaps.get_cmap("plasma")

    def create_arrow_surface(self):
        arrow_size = int(self.vector_scale) + 20
        surface = pygame.Surface((arrow_size, arrow_size), pygame.SRCALPHA)
        start_x, start_y = arrow_size // 2, arrow_size // 2

        end_x = float(start_x + self.u * self.vector_scale)
        end_y = float(start_y + self.v * self.vector_scale)

        pygame.draw.line(surface, self.color, (start_x, start_y), (end_x, end_y), 3)

        arrowhead_size = ARROWHEAD_SIZE
        angle = np.arctan2(end_y - start_y, end_x - start_x)
        p1 = (
            end_x - arrowhead_size * np.cos(angle - np.pi / 6),
            end_y - arrowhead_size * np.sin(angle - np.pi / 6),
        )
        p2 = (
            end_x - arrowhead_size * np.cos(angle + np.pi / 6),
            end_y - arrowhead_size * np.sin(angle + np.pi / 6),
        )
        pygame.draw.polygon(surface, self.color, [(end_x, end_y), p1, p2])

        return surface

    def get_color_from_magnitude(self, normalized_mag):
        # if self.max_mag <= self.min_mag:
        #     return (0, 0, 0)  # Handle the edge case to prevent division by zero

        # Get the 'plasma' colormap object

        # Get the RGBA color from the colormap (it returns a tuple of floats 0-1)
        rgba_color = self.plasma_cmap(normalized_mag)

        return rgba_color

    def update(
        self,
        dt,
        dataset,
        tree,
        data_min_x,
        data_min_y,
        scale_x,
        scale_y,
        min_mag,
        max_mag,
    ):
        self.pos[0] += self.u * FLOW_SPEED * dt
        self.pos[1] += self.v * FLOW_SPEED * dt
        self.ttl -= dt  # Decrease TTL over time

        data_x = (self.pos[0] / scale_x) + data_min_x
        data_y = -(self.pos[1] / scale_y) - data_min_y
        u_new, v_new = get_vector_at_position(data_x, data_y, dataset, tree)

        magnitude_new = np.sqrt(u_new**2 + v_new**2)
        normalized_mag_new = (
            (magnitude_new - min_mag) / (max_mag - min_mag) if max_mag > min_mag else 0
        )
        color_new = self.color_func(magnitude_new)
        # r = int(0 + (255 - 0) * normalized_mag_new)
        # g = int(255 + (0 - 255) * normalized_mag_new)
        # c_b = int(255 + (0 - 255) * normalized_mag_new)
        # color_new = (r, g, c_b)
        # color_new = self.get_color_from_magnitude(normalized_mag_new)

        self.u = u_new
        self.v = v_new
        self.color = color_new

        self.image = self.create_arrow_surface()
        self.rect.center = self.pos

        return True


# --- Arrow Manager Class ---
class ArrowManager:
    def __init__(
        self,
        dataset,
        data_min_x,
        data_min_y,
        scale_x,
        scale_y,
        min_mag,
        max_mag,
        screen_size,
    ):
        self.arrows = []
        self.dataset = dataset
        self.tree = KDTree(np.column_stack((dataset["face_x"], dataset["face_y"])))
        self.data_min_x = data_min_x
        self.data_min_y = data_min_y
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.min_mag = min_mag
        self.max_mag = max_mag
        self.screen_size = screen_size

        self.spawn_points = []
        x_coords = dataset["face_x"]
        y_coords = -dataset["face_y"]
        u = dataset["ucx"]
        v = dataset["ucy"]
        magnitudes = np.sqrt(u**2 + v**2)
        self.plasma_cmap = colormaps.get_cmap("plasma")

        self.plasma_lut = [
            (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255))
            for c in self.plasma_cmap(np.linspace(0, 1, 256))
        ]

        for i in range(len(u)):
            if magnitudes[i] > SPAWN_THRESHOLD_MAG:
                self.spawn_points.append(
                    {
                        "pos": (x_coords[i], y_coords[i]),
                        "u": u[i],
                        "v": v[i],
                        "color": self.get_color_from_magnitude_plasma(magnitudes[i]),
                        "color_func": self.get_color_from_magnitude_plasma,
                    }
                )

        np.random.shuffle(self.spawn_points)

        for _ in range(MAX_ARROWS):
            self.spawn_arrow()

        # Create a single surface to blit all arrows onto
        self.arrow_surface = pygame.Surface(self.screen_size, pygame.SRCALPHA)

    def get_color_from_magnitude(self, magnitude):

        normalized_mag = (
            (magnitude - self.min_mag) / (self.max_mag - self.min_mag)
            if self.max_mag > self.min_mag
            else 0
        )

        r = int(0 + (255 - 0) * normalized_mag)

        g = int(255 + (0 - 255) * normalized_mag)

        c_b = int(255 + (0 - 255) * normalized_mag)

        return (r, g, c_b)

    def get_color_from_magnitude_plasma(self, magnitude):
        """
        Returns an RGB color from the 'plasma' colormap based on a given magnitude.
        """
        if self.max_mag <= self.min_mag:
            return (0, 0, 0)  # Handle the edge case to prevent division by zero

        # Normalize the magnitude to a value between 0 and 1
        normalized_mag = (magnitude - self.min_mag) / (self.max_mag - self.min_mag)

        # Get the 'plasma' colormap object

        # Get the RGBA color from the colormap (it returns a tuple of floats 0-1)
        rgba_color = self.plasma_cmap(normalized_mag)
        index = int(normalized_mag * 255)

        # Convert the RGBA tuple to an RGB tuple of integers (0-255)

        return self.plasma_lut[index]

    def spawn_arrow(self):
        if not self.spawn_points:
            return

        point = self.spawn_points.pop(0)

        start_x = (point["pos"][0] - self.data_min_x) * self.scale_x
        start_y = (point["pos"][1] - self.data_min_y) * self.scale_y

        new_arrow = Arrow(
            (start_x, start_y),
            point["u"],
            point["v"],
            ARROW_SCALE,
            self.scale_x,
            self.scale_y,
            point["color"],
            point["color_func"],
        )
        self.arrows.append(new_arrow)
        self.spawn_points.append(point)

    def update_and_draw(self, screen, dt, pos_offset=(0, 0)):
        culled_arrows = []

        for arrow in self.arrows:
            arrow.update(
                dt,
                self.dataset,
                self.tree,
                self.data_min_x,
                self.data_min_y,
                self.scale_x,
                self.scale_y,
                self.min_mag,
                self.max_mag,
            )

            is_off_screen = (
                arrow.pos[0] < -OFFSCREEN_BUFFER
                or arrow.pos[0] > self.screen_size[0] + OFFSCREEN_BUFFER
                or arrow.pos[1] < -OFFSCREEN_BUFFER
                or arrow.pos[1] > self.screen_size[1] + OFFSCREEN_BUFFER
            )

            is_stagnant = (
                np.sqrt(arrow.u**2 + arrow.v**2) < STAGNANT_MAGNITUDE_THRESHOLD
            )

            is_expired = arrow.ttl <= 0

            if is_off_screen or is_stagnant or is_expired:
                culled_arrows.append(arrow)

        for arrow_to_remove in culled_arrows:
            self.arrows.remove(arrow_to_remove)
            self.spawn_arrow()

        self.arrow_surface.fill((0, 0, 0, 0))
        blit_list = [(arrow.image, arrow.rect) for arrow in self.arrows]
        self.arrow_surface.blits(blit_list)

        screen.blit(self.arrow_surface, pos_offset)


def initialize_arrow_manager(tidal_data, screen_size):
    screen_width, screen_height = screen_size

    original_dataset = {k: v.copy() for k, v in tidal_data.items()}

    x_coords_o = original_dataset["face_x"]
    y_coords_o = -original_dataset["face_y"]
    u_o = original_dataset["ucx"]
    v_o = original_dataset["ucy"]

    # Calculate magnitude for coloring
    magnitudes = np.sqrt(u_o**2 + v_o**2)
    max_magnitude = np.max(magnitudes)
    min_magnitude = np.min(magnitudes)

    # Define a scaling factor to fit the data to the screen
    data_min_x, data_max_x = np.min(x_coords_o), np.max(x_coords_o)
    data_min_y, data_max_y = np.min(y_coords_o), np.max(y_coords_o)

    scale_x = screen_width / (data_max_x - data_min_x)
    scale_y = screen_height / (data_max_y - data_min_y)

    arrow_manager = ArrowManager(
        tidal_data,
        data_min_x,
        data_min_y,
        scale_x,
        scale_y,
        min_magnitude,
        max_magnitude,
        (screen_width, screen_height),
    )

    return arrow_manager, (data_min_x, data_max_x, data_min_y, data_max_y)


def update_arrow_manager(arrow_manager, event, min_max_data):
    data_min_x, data_max_x, data_min_y, data_max_y = min_max_data

    if event.type == pygame.QUIT:
        running = False

    if event.type == pygame.VIDEORESIZE:
        # Recalculate everything on resize
        screen_width, screen_height = event.size
        # screen = pygame.display.set_mode(
        #     (screen_width, screen_height), pygame.RESIZABLE
        # )

        scale_x = screen_width / (data_max_x - data_min_x)
        scale_y = screen_height / (data_max_y - data_min_y)

        # Update the manager with the new values
        arrow_manager.scale_x = scale_x
        arrow_manager.scale_y = scale_y
        arrow_manager.screen_size = (screen_width, screen_height)
        arrow_manager.arrow_surface = pygame.Surface(
            (screen_width, screen_height), pygame.SRCALPHA
        )

        # Rescale existing arrows
        for arrow in arrow_manager.arrows:
            data_x = (arrow.pos[0] / arrow.scale_x) + data_min_x
            data_y = (arrow.pos[1] / arrow.scale_y) - data_min_y
            arrow.pos[0] = (data_x - data_min_x) * scale_x
            arrow.pos[1] = (data_y - data_min_y) * scale_y
            arrow.rect.center = arrow.pos

    # screen.fill(BACKGROUND_COLOR)

    # arrow_manager.update_and_draw(screen, dt)

    # pygame.display.flip()
