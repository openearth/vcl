"""Pygame-based visualization and control module for VCL (Virtual Climate Lab).

This module provides the main display interface for the VCL system using Pygame and
ZeroMQ for inter-process communication. It manages multiple visualization windows,
handles user input from keyboards and MIDI controllers, and coordinates real-time
data updates across different display components.

Key Components:
    - DisplayMap: Main map visualization showing geospatial layers
    - DisplaySlice: Cross-section/slice visualization
    - StatsWindow: Statistics and information panels
    - Hand tracking: MediaPipe-based gesture control
    - UID detection: AprilTag/QR code/ArUco marker detection for interactivity

Communication:
    The module uses ZeroMQ publish-subscribe pattern across multiple ports:
    - 5556: Keyboard/MIDI commands (maps, year changes)
    - 5557: Hand tracking coordinates
    - 5558: UID detection results
"""

import bisect
import collections
import concurrent.futures
import os
import sys
import threading
import time
from pathlib import Path
import geopandas as gpd

import matplotlib as mpl
import matplotlib.pyplot as plt
import mido
import numpy as np

# import pywinctl as gw
import pygame
import zmq
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

import vcl.preprocess

# from vcl.windows import DisplayMap, DisplaySlice
from vcl.windows import DisplayMap, DisplaySlice, StatsWindow
from vcl.utils import hand_tracking
from vcl.load_data import load_preprocessed
from vcl.interactivity import uid_detection

# Global state variables for layer management
contour_show = False
height_map_show = False
compare = False
current_layer = ""  # Currently active base layer
current_overlay = ""  # Currently active overlay layer
current_tide = ""  # Currently active tide/current visualization
current_overlays = []  # Stack of active overlays

# Custom colormap for windfarm visualization
# Colors represent different windfarm categories or states
windfarm_cmap = [
    (255 / 255, 255 / 255, 255 / 255, 0.25),
    (254 / 255, 217 / 255, 142 / 255, 1),
    (254 / 255, 153 / 255, 41 / 255, 1),
    (217 / 255, 95 / 255, 14 / 255, 1),
    (153 / 255, 52 / 255, 4 / 255, 1),
    (0 / 255, 197 / 255, 255 / 255, 1),
    (0 / 255, 112 / 255, 192 / 255, 1),
    (83 / 255, 36 / 255, 118 / 255, 1),
    (193 / 255, 193 / 255, 193 / 255, 1),
]
windfarm_cmap = ListedColormap(windfarm_cmap)

# Custom colormap for bathymetry (depth) visualization
# Color scale transitions from deep blue (deep water) to yellow/red (shallow/land)
bathymetry_cmap = [
    # (0, (0 / 255, 0 / 255, 62 / 255)),
    (0, ((0 / 255, 0 / 255, 53 / 255))),
    (0.5, (10 / 255, 20 / 255, 220 / 255)),
    (0.75, (0 / 255, 0 / 255, 205 / 255)),
    (0.8875, (90 / 255, 213 / 255, 6 / 255)),
    (0.9625, (181 / 255, 211 / 255, 4 / 255)),
    (0.9875, (215 / 255, 215 / 255, 14 / 255)),
    (0.9975, (218 / 255, 6 / 255, 22 / 255)),
    (1, (222 / 255, 90 / 255, 93 / 255)),
]  # Deep sea blue → shallow water blue → coastal green → land yellow/red

# Create continuous colormap with 5000 discrete steps for smooth gradients
bathymetry_cmap = LinearSegmentedColormap.from_list(
    "bathy_cmap", bathymetry_cmap, N=5000
)


def make_listen_sockets():
    """Create and configure ZeroMQ subscriber sockets for inter-process communication.

    This function initializes multiple ZMQ subscriber sockets on different ports to
    receive messages from various input sources (keyboard, MIDI, hand tracking, UID
    detection). Each socket subscribes to specific topics and some use CONFLATE to
    ensure only the latest message is received.

    Returns:
        dict: Dictionary containing ZMQ context, sockets, and poller with keys:
            - context: ZMQ context object
            - maps: Socket for map layer change commands (port 5556, 5557)
            - pygame_2: Socket for pygame instance selection (port 5556)
            - year: Socket for year/time period changes (port 5556)
            - slice: Socket for cross-section slice updates (port 5556, 5558)
            - hands: Socket for hand tracking coordinates (port 5557, 5558)
            - uid: Socket for UID detection results (port 5558)
            - poller: ZMQ poller for multiplexing socket events

    Note:
        CONFLATE option ensures only the most recent message is kept in the queue,
        preventing lag from accumulated messages during heavy processing.
    """
    context = zmq.Context()

    socket1 = context.socket(zmq.SUB)
    socket1.setsockopt(zmq.CONFLATE, 1)
    socket1.connect("tcp://localhost:5556")
    socket1.connect("tcp://localhost:5557")
    socket1.connect("tcp://localhost:5558")
    socket1.subscribe("maps")

    socket2 = context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:5556")
    socket2.subscribe("pygame_2")

    socket3 = context.socket(zmq.SUB)
    socket3.setsockopt(zmq.CONFLATE, 1)
    socket3.connect("tcp://localhost:5556")
    socket3.connect("tcp://localhost:5558")
    socket3.subscribe("slice")

    socket4 = context.socket(zmq.SUB)
    socket4.setsockopt(zmq.CONFLATE, 1)
    socket4.connect("tcp://localhost:5556")
    socket4.subscribe("year")

    socket5 = context.socket(zmq.SUB)
    socket5.setsockopt(zmq.CONFLATE, 1)
    socket5.connect("tcp://localhost:5557")
    socket5.connect("tcp://localhost:5558")
    socket5.subscribe("hands")

    socket6 = context.socket(zmq.SUB)
    socket6.setsockopt(zmq.CONFLATE, 1)
    socket6.connect("tcp://localhost:5558")
    socket6.subscribe("uid")

    poller = zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)
    poller.register(socket3, zmq.POLLIN)
    poller.register(socket4, zmq.POLLIN)
    poller.register(socket5, zmq.POLLIN)
    poller.register(socket6, zmq.POLLIN)

    sockets = {
        "context": context,
        "maps": socket1,
        "pygame_2": socket2,
        "year": socket4,
        "slice": socket3,
        "hands": socket5,
        "uid": socket6,
        "poller": poller,
    }
    return sockets


def displaymap(data_path):
    """Main map display window showing geospatial layers and overlays.

    This function creates and runs the primary map visualization window. It handles
    layer switching, overlay management, year changes, hand tracking visualization,
    and current/tide animations. Messages are received via ZMQ sockets
    from keyboard, MIDI, and tracking modules.

    Args:
        datasets: Preprocessed dataset dictionary (not used - data is loaded internally).

    Socket Messages:
        - maps: Layer change commands in format "layer_name,layer_type"
        - slice: Slice index for cross-section line positioning
        - year: Year/time period for temporal data
        - hands: Hand tracking coordinates in format "x,y"

    Note:
        Runs in an infinite loop until the process is terminated. Uses non-blocking
        ZMQ polling with 10ms timeout to maintain responsiveness.
    """
    datasets = load_preprocessed(data_path=data_path)
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    global current_layer

    from matplotlib.colors import Normalize

    norm = Normalize(vmin=-4000, vmax=0)

    dataset_kwargs = {
        "basemap": {"type": "RGB"},
        "bathymetry": {"type": "CMAP"},
        "contamination": {"type": "RGB"},
        "ground_cover": {"type": "RGB"},
        "treat_water": {"type": "RGB"},
        "1911": {"type": "RGB"},
        "1913": {"type": "RGB"},
        "1942": {"type": "RGB"},
        "1967": {"type": "RGB"},
        "1983_on": {"type": "RGB"},
        "1983_off": {"type": "RGB"},
        "2008": {"type": "RGB"},
        "2009": {"type": "RGB"},
        "2014": {"type": "RGB"},
        "2018": {"type": "RGB"},
        "2022_lines": {"type": "RGB"},
        "energy_park": {"type": "RGB"},
        "housing": {"type": "RGB"},
        "innovation_hub": {"type": "RGB"},
    }

    available_years = sorted(int(k) for k in dataset_kwargs.keys() if str(k).isdigit())

    def get_closest_year(y):
        idx = bisect.bisect_right(available_years, y) - 1
        return str(available_years[max(idx, 0)])

    socket = sockets["maps"]
    socket_slice = sockets["slice"]
    socket_year = sockets["year"]
    socket_hands = sockets["hands"]
    try:
        display = DisplayMap.DisplayMap(
            datasets=datasets,
            start_year="",
            flow_data={},
            animations_data=datasets[""]["animations"],
            dataset_kwargs=dataset_kwargs,
            bg_layer="basemap",
            mask_layer=None,
            i_max=338,
            aspect_ratio=1920 / 1080,
        )
    except Exception as e:
        print(e)
    coords = None

    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket in socks and socks[socket] == zmq.POLLIN:
            topic, message = socket.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type = message.split(",")
            if view_type == "tide":
                display.init_arrowmanager(layer)
                # display.init_arrowmanager(layer)
            elif view_type == "overlay":
                display.display_overlay(layer)
            elif view_type == "mask":
                display.display_mask()
            elif view_type == "scenario":
                display.change_scenario(layer)
            elif view_type == "measure":
                display.change_measure(layer)
            else:
                display.change_layer(layer)

        if socket_slice in socks and socks[socket_slice] == zmq.POLLIN:
            topic, message = socket_slice.recv(zmq.DONTWAIT).split()
            slice_index = float(message)
            display.change_line_index(slice_index)
            year = display.index_to_year(min_year=1911, max_year=2250)
            closest_year = get_closest_year(year)

            if closest_year != current_layer:
                display.change_layer(closest_year)
                current_layer = closest_year

        if socket_year in socks and socks[socket_year] == zmq.POLLIN:
            topic, message = socket_year.recv(zmq.DONTWAIT).split()
            year = message.decode("utf-8")
            display.change_year(year)

        if socket_hands in socks and socks[socket_hands] == zmq.POLLIN:
            topic, coords = socket_hands.recv(zmq.DONTWAIT).split()
            coords = coords.decode("utf-8")
            xcoord, ycoord = coords.split(",")
            xcoord = float(xcoord)
            ycoord = float(ycoord)
            coords = (xcoord, ycoord)
            display.start_hand_tracking(coords)

        display.draw_layers()


def displaystats(data_path):
    """Statistics and information panel display window.

    This function creates a window showing statistical information, charts, and
    infographics for the selected layer. It responds to layer changes and UID
    detection for interactive navigation.

    Args:
        datasets: Preprocessed dataset dictionary (not used - data is loaded internally).

    Socket Messages:
        - maps: Layer selection for displaying corresponding statistics
        - uid: UID detection results for interactive layer navigation

    Note:
        Uses matplotlib's pause() for rendering updates. Certain layers are ignored
        (mask, animation, 20, 30) as they don't have associated statistics.
    """
    datasets = load_preprocessed(data_path=data_path)
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    socket = sockets["maps"]
    socket_uid = sockets["uid"]
    socket_slice = sockets["slice"]
    socket_year = sockets["year"]

    global current_layer

    dataset_kwargs = {
        "1911": {"image": {"title": ""}},
        "1913": {"image": {"title": ""}},
        "1942": {"image": {"title": ""}},
        "1967": {"image": {"title": ""}},
        "1983": {"image": {"title": ""}},
        "2008": {"image": {"title": ""}},
        "2009": {"image": {"title": ""}},
        "2014": {"image": {"title": ""}},
        "2018": {"image": {"title": "", "multiple": "sequence", "interval_s": 3.0}},
        "housing": {"image": {"title": "", "multiple": "sequence", "interval_s": 1.0}},
        "energy_park": {
            "image": {"title": "", "multiple": "sequence", "interval_s": 1.0}
        },
        "innovation_hub": {
            "image": {"title": "", "multiple": "sequence", "interval_s": 1.0}
        },
    }

    display = StatsWindow.StatsWindow(
        datasets[""]["stats"],
        dataset_kwargs=dataset_kwargs,
        layers_to_ignore=["mask", "animation", "20", "30"],
        overlay_layers=[],
    )

    available_years = sorted(int(k) for k in dataset_kwargs.keys() if str(k).isdigit())

    def get_closest_year(y):
        idx = bisect.bisect_right(available_years, y) - 1
        return str(available_years[max(idx, 0)])

    def index_to_year(index, min_year, max_year):
        year = min_year + int((max_year - min_year) * index / 338)
        return year

    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket in socks and socks[socket] == zmq.POLLIN:
            topic, message = socket.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type = message.split(",")
            print(layer)
            display.change_layer(layer)
        if socket_slice in socks and socks[socket_slice] == zmq.POLLIN:
            topic, message = socket_slice.recv(zmq.DONTWAIT).split()
            slice_index = float(message)
            slice_index = np.clip(slice_index, 0, 338)
            year = index_to_year(slice_index, min_year=1911, max_year=2250)
            closest_year = get_closest_year(year)

            if closest_year != current_layer:
                display.change_layer(closest_year)
                current_layer = closest_year
        if socket_uid in socks and socks[socket_uid] == zmq.POLLIN:
            try:
                topic, coords = socket_uid.recv(zmq.DONTWAIT).split()
                coords = coords.decode("utf-8")
                display.change_layer(coords)
            except Exception as e:
                print(e)
        plt.pause(0.01)


def displayslice(data_path):
    """Cross-section slice visualization window.

    This function creates a window displaying vertical or horizontal cross-sections
    through the data at the current slice position. The slice position is controlled
    by keyboard/MIDI input.

    Args:
        datasets: Preprocessed dataset dictionary (not used - data is loaded internally).

    Socket Messages:
        - slice: Slice index position for updating the cross-section view

    Note:
        Currently uses empty slice_datasets and dataset_kwargs - implementation
        may be incomplete or requires configuration.
    """
    datasets = load_preprocessed(data_path=data_path)
    sockets = make_listen_sockets()
    poller = sockets["poller"]
    socket_slice = sockets["slice"]

    slice_datasets = {}

    dataset_kwargs = {}

    display = DisplaySlice.DisplaySlice(slice_datasets, dataset_kwargs)
    while True:
        socks = dict(poller.poll(10))

        if socket_slice in socks and socks[socket_slice] == zmq.POLLIN:
            topic, message = socket_slice.recv(zmq.DONTWAIT).split()
            slice_index = int(message)
            display.change_index(slice_index)
        display.draw_layers()


def keyboard_publisher():
    """Keyboard input handler that publishes commands via ZMQ.

    This function creates a Pygame window for keyboard input and translates keypresses
    into ZMQ messages that control the display windows. It handles layer selection,
    overlay toggling, year changes, and slice navigation.

    Key Mappings:
        1-9: Layer/overlay selection
            1: Bathymetry layer
            2: Fishing layer
            3: Navisafe layer
            4: Windfarms layer
            5: Windfarms overlay
            6: Nature tracking layer
            7: Vessel traffic overlay
            8: EEZ (Exclusive Economic Zone) overlay
            9: OSPAR overlay
        A: Animation layer (only for windfarms)
        Q/E: Tide visualization (20th/30th timestep)
        I/O/P: Year selection (2023/2050/2100)
        M: Mask layer toggle
        N/B: Cycle next/previous in active layer collection
        LEFT/RIGHT: Navigate slice position

    ZMQ Topics:
        - maps: Layer change commands
        - year: Year/time period changes
        - slice: Slice position updates
        - pygame_1/pygame_2: Instance-specific messages

    Note:
        Runs until window is closed. Uses global variables to track current layer
        state for toggle behavior.
    """
    # --- ZMQ Setup ---
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5556")

    def change_year(year):
        socket.send_string(f"year {year}")

    # Function to send layer when button pressed
    def change_layer(text):
        layer_type = text.split(",")[1]
        global current_layer, current_overlay, current_tide, current_overlays
        if text.split(",")[0] == "":
            socket.send_string(f"maps {text}")
            current_layer = ""
            current_tide = ""
        if current_layer == text or current_tide == text:
            socket.send_string(f"maps None,{layer_type}")
            if layer_type == "layer":
                current_layer = ""
            elif layer_type == "tide":
                current_tide = ""
        else:
            socket.send_string(f"maps {text}")
            if layer_type == "layer":
                current_layer = text
            elif layer_type == "tide":
                current_tide = text
            elif layer_type == "overlay" and text in current_overlays:
                current_overlay = ""
                current_overlays.remove(text)
            elif layer_type == "overlay" and text not in current_overlays:
                current_overlay = text
                current_overlays.append(text)

    solutions = collections.deque(["energy_park", "housing", "innovation_hub"])
    risico_zone = collections.deque(["risico_zone", "risico_zone_20"])
    overstromingen = collections.deque(
        [
            "d_T1000_noord",
            "d_T1000_zuid",
            "d_T1000_oost",
            "d_T1000_west",
            "d_T1000_barendrecht",
            "d_T10_000_noord",
            "d_T10_000_zuid",
            "d_T10_000_oost",
            "d_T10_000_west",
            "d_T10_000_barendrecht",
            "d_T100_000_1_noord",
            "d_T100_000_1_zuid",
            "d_T100_000_1_oost",
            "d_T100_000_1_west",
            "d_T100_000_1_barendrecht",
            "d_T100_000_noord",
            "d_T100_000_zuid",
            "d_T100_000_oost",
            "d_T100_000_west",
        ]
    )

    collection_type_mapping = (
        {layer: "overstroming" for layer in overstromingen}
        | {layer: "solutions" for layer in solutions}
        | {layer: "risico_zone" for layer in risico_zone}
    )

    cycles = {
        "overstroming": overstromingen,
        "solutions": solutions,
        "risico_zone": risico_zone,
    }

    def cycle_collection(cycle):
        global current_layer, current_overlays

        if len(current_overlays) > 0:
            current_overlay = current_overlays[-1]
        else:
            current_overlay = ""

        if (
            current_overlay != ""
            and current_overlay.split(",")[0] in collection_type_mapping
        ):
            layer = current_overlay
        else:
            layer = current_layer

        if current_layer != "" and current_layer is not None:
            layer_name = layer.split(",")[0]
            layer_type = layer.split(",")[1]
        else:
            return

        collection_type = collection_type_mapping.get(layer_name, None)

        if collection_type is None:
            return

        if cycle == "next":
            cycles[collection_type].rotate(-1)
            next_layer = cycles[collection_type][0]
            # gxgs.rotate(-1)
            # next_gxg = gxgs[0]
        elif cycle == "prev":
            # gxgs.rotate(1)
            # next_gxg = gxgs[0]
            cycles[collection_type].rotate(1)
            next_layer = cycles[collection_type][0]

        layer = f"{next_layer},{layer_type}"

        if current_layer in [
            f"{collection},{layer_type}" for collection in cycles[collection_type]
        ]:
            if layer_type == "layer":
                change_layer(layer)
        elif current_overlay in [
            f"{collection},{layer_type}" for collection in cycles[collection_type]
        ]:
            if layer_type == "overlay":
                if cycle == "prev":
                    change_layer(f"{cycles[collection_type][1]},overlay")
                else:
                    change_layer(f"{cycles[collection_type][-1]},overlay")
                time.sleep(0.01)
                change_layer(layer)

    alpha = 1

    def change_alpha(text, alpha):
        alpha = alpha - 0.1
        socket.send_string(f"maps ")

    # --- Pygame Setup ---
    pygame.init()
    screen_width, screen_height = 400, 200
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Keyboard Publisher")
    font = pygame.font.Font(None, 36)
    keys_held = {}  # Dictionary to track held keys
    # --- Main Loop -
    slice_index = 0
    max_slices = 338
    # max_slices = 300

    R_runnig = False
    L_running = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    change_layer("bathymetry,layer")
                elif event.key == pygame.K_2:
                    change_layer(f"contamination,layer")
                elif event.key == pygame.K_3:
                    change_layer(f"ground_cover,layer")
                elif event.key == pygame.K_4:
                    change_layer("treat_water,layer")
                elif event.key == pygame.K_5:
                    change_layer("1911,layer")
                elif event.key == pygame.K_6:
                    change_layer("1913,layer")
                elif event.key == pygame.K_7:
                    change_layer("1942,layer")
                elif event.key == pygame.K_8:
                    change_layer("1967,layer")
                elif event.key == pygame.K_9:
                    change_layer("2009,layer")
                elif event.key == pygame.K_0:
                    change_layer(f"{solutions[0]},layer")

                elif event.key == pygame.K_g:
                    change_year(1970)
                elif event.key == pygame.K_h:
                    change_year(2010)
                elif event.key == pygame.K_a:
                    change_layer("animation,layer")
                elif event.key == pygame.K_s:
                    change_layer("bathymetry,layer")
                elif event.key == pygame.K_d:
                    change_layer("doorbraaklocaties,overlay")
                elif event.key == pygame.K_o:
                    change_year(2050)
                elif event.key == pygame.K_p:
                    change_year(2100)
                elif event.key == pygame.K_m:
                    change_layer("mask,mask")
                elif event.key == pygame.K_n:
                    cycle_collection("next")
                elif event.key == pygame.K_b:
                    cycle_collection("prev")
                # elif event.key == pygame.K_RIGHT:
                #     socket.send_string("slice R")
                # elif event.key == pygame.K_LEFT:
                #     socket.send_string("slice L")
                # elif event.key == pygame.K_ESCAPE:
                #     running = False
            # for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and event.key not in keys_held:
                    # slice_index = (slice_index + 1) % max_slices
                    # socket.send_string("slice R_START")
                    # keys_held[event.key] = pygame.time.get_ticks()
                    R_runnig = True
                elif event.key == pygame.K_LEFT and event.key not in keys_held:
                    # slice_index = (slice_index - 1) % max_slices
                    # socket.send_string("slice L_START")
                    # keys_held[event.key] = pygame.time.get_ticks()
                    L_running = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT:
                    # socket.send_string("slice R_STOP")
                    # keys_held.pop(event.key, None)
                    R_runnig = False
                elif event.key == pygame.K_LEFT:
                    # socket.send_string("slice L_STOP")
                    # keys_held.pop(event.key, None)
                    L_running = False

        if R_runnig:
            slice_index = (slice_index + 1) % max_slices
            socket.send_string(f"slice {slice_index}")
        if L_running:
            slice_index = (slice_index - 1) % max_slices
            socket.send_string(f"slice {slice_index}")
        # Check for held keys and send a continuous message
        current_time = pygame.time.get_ticks()
        hold_delay = 10  # milliseconds
        if (
            pygame.K_RIGHT in keys_held
            and current_time - keys_held[pygame.K_RIGHT] > hold_delay
        ):
            socket.send_string("slice R_HOLD")
            keys_held[pygame.K_RIGHT] = current_time
        if (
            pygame.K_LEFT in keys_held
            and current_time - keys_held[pygame.K_LEFT] > hold_delay
        ):
            socket.send_string("slice L_HOLD")
            keys_held[pygame.K_LEFT] = current_time

        # --- Display Instructions ---
        screen.fill((50, 50, 50))
        text = font.render(
            "Press 'H' for Instance 1, 'J' for Instance 2", True, (255, 255, 255)
        )
        screen.blit(text, (20, 80))
        pygame.display.flip()

        # Add a small delay to avoid consuming all CPU
        time.sleep(0.01)

    pygame.quit()
    sys.exit()


def start_thread_to_terminate_when_parent_process_dies(ppid):
    """Start a daemon thread to monitor parent process and terminate if parent dies.

    This function is used as an initializer for worker processes to ensure they
    terminate cleanly if the parent process crashes or is killed.

    Args:
        ppid: Parent process ID to monitor.

    Note:
        The thread is started as a daemon but the monitoring logic is not implemented.
        This is a placeholder for process lifecycle management.
    """
    thread = threading.Thread(daemon=True)
    thread.start()


def main():
    """Main entry point for the VCL display system.

    This function initializes a process pool and launches multiple display windows
    and input handlers as separate processes. Each process runs independently and
    communicates via ZMQ sockets.

    Launched Processes:
        - keyboard_publisher: Keyboard input handler
        - displaymap: Main map visualization
        - displayslice: Cross-section viewer

    Returns:
        int: Exit code (always returns 0).

    Note:
        Uses ProcessPoolExecutor with up to 10 workers. Additional processes
        (midi_board, hand_tracker, uid_detector) can be added via executor.submit().
    """
    """Console script for vcl."""

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=10,
        initializer=start_thread_to_terminate_when_parent_process_dies,
        initargs=(os.getpid(),),
    )

    executor.submit(keyboard_publisher)
    executor.submit(displaymap)
    executor.submit(displayslice)

    return 0


def midi_board(data_path):
    """MIDI controller input handler that publishes commands via ZMQ.

    This function listens for MIDI input from a connected controller and translates
    MIDI control change messages into ZMQ commands for controlling the display windows.
    It supports both button presses and slider/fader inputs.

    Args:
        datasets: Preprocessed dataset dictionary (not used - data is loaded internally).

    MIDI Mapping:
        Buttons (value 127):
            1: OWF 2030 overlay
            2: OSPAR overlay
            23: Bathymetry layer
            24: Navisafe layer
            25: Fishing layer
            26: Windfarms layer
            27: Windfarms overlay
            28: Nature tracking layer
            29: Vessel traffic overlay
            30: EEZ overlay
            31: Mask toggle
            44: OWF 2040 overlay
            45/46: Animation start/stop
            47/48: Cycle previous/next in collection
            64: Tide visualization (20)
            67: OWF all overlay

        Sliders (0-127):
            3: Year selection (2023/2050/2100)
            7: Year selection (2023/2100)
            60: Slice position

    Note:
        The function runs until the MIDI BANK button is pressed (sysex message).
        Uses exception handling to ignore unmapped controls gracefully.
    """
    # import ipdb

    # ipdb.set_trace()
    # Create publishing socket for sending midi board messages to the windows
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5556")
    # Number of values for the sliders on midi board (0-127)
    n_slider_values = 128

    # Update slider based on 128 possible values
    def slider_update(value):
        socket.send_string(f"slice {int(value)}")

    years = ["2023", "2050", "2100"]

    def change_year(value, years=years):
        index = int(value * len(years) / n_slider_values)
        year = years[index]
        socket.send_string(f"year {year}")

    # Function to send layer when button pressed
    def change_layer(text):
        layer_type = text.split(",")[1]
        global current_layer, current_overlay, current_tide, current_overlays
        if text.split(",")[0] == "":
            socket.send_string(f"maps {text}")
            current_layer = ""
            current_tide = ""
        if current_layer == text or current_tide == text:
            socket.send_string(f"maps None,{layer_type}")
            if layer_type == "layer":
                current_layer = ""
            elif layer_type == "tide":
                current_tide = ""
        else:
            socket.send_string(f"maps {text}")
            if layer_type == "layer":
                current_layer = text
            elif layer_type == "tide":
                current_tide = text
            elif layer_type == "overlay" and text in current_overlays:
                current_overlay = ""
                current_overlays.remove(text)
            elif layer_type == "overlay" and text not in current_overlays:
                current_overlay = text
                current_overlays.append(text)

    waterdiepte = collections.deque(["d_T100_000", "d_T100"])
    risico_zone = collections.deque(["risico_zone", "risico_zone_20"])
    overstromingen = collections.deque(
        [
            "d_T1000_noord",
            "d_T1000_zuid",
            "d_T1000_oost",
            "d_T1000_west",
            "d_T1000_barendrecht",
            "d_T10_000_noord",
            "d_T10_000_zuid",
            "d_T10_000_oost",
            "d_T10_000_west",
            "d_T10_000_barendrecht",
            "d_T100_000_1_noord",
            "d_T100_000_1_zuid",
            "d_T100_000_1_oost",
            "d_T100_000_1_west",
            "d_T100_000_1_barendrecht",
            "d_T100_000_noord",
            "d_T100_000_zuid",
            "d_T100_000_oost",
            "d_T100_000_west",
        ]
    )

    collection_type_mapping = (
        {layer: "overstroming" for layer in overstromingen}
        | {layer: "waterdiepte" for layer in waterdiepte}
        | {layer: "risico_zone" for layer in risico_zone}
    )

    cycles = {
        "overstroming": overstromingen,
        "waterdiepte": waterdiepte,
        "risico_zone": risico_zone,
    }

    def cycle_collection(cycle):
        global current_layer, current_overlays

        if len(current_overlays) > 0:
            current_overlay = current_overlays[-1]
        else:
            current_overlay = ""

        if (
            current_overlay != ""
            and current_overlay.split(",")[0] in collection_type_mapping
        ):
            layer = current_overlay
        else:
            layer = current_layer

        if current_layer != "" and current_layer is not None:
            layer_name = layer.split(",")[0]
            layer_type = layer.split(",")[1]
        else:
            return

        collection_type = collection_type_mapping.get(layer_name, None)

        if collection_type is None:
            return

        if cycle == "next":
            cycles[collection_type].rotate(-1)
            next_layer = cycles[collection_type][0]
            # gxgs.rotate(-1)
            # next_gxg = gxgs[0]
        elif cycle == "prev":
            # gxgs.rotate(1)
            # next_gxg = gxgs[0]
            cycles[collection_type].rotate(1)
            next_layer = cycles[collection_type][0]

        layer = f"{next_layer},{layer_type}"

        if current_layer in [
            f"{collection},{layer_type}" for collection in cycles[collection_type]
        ]:
            if layer_type == "layer":
                change_layer(layer)
        elif current_overlay in [
            f"{collection},{layer_type}" for collection in cycles[collection_type]
        ]:
            if layer_type == "overlay":
                if cycle == "prev":
                    change_layer(f"{cycles[collection_type][1]},overlay")
                else:
                    change_layer(f"{cycles[collection_type][-1]},overlay")
                time.sleep(0.01)
                change_layer(layer)

    # def change_overlay(text):
    #     global current_overlay
    #     if current_overlay

    def start_stop_animation(text):
        global current_layer
        if text == "":
            socket.send_string(f"maps animation,layer")
            socket.send_string(f"maps {current_layer}")
        else:
            socket.send_string(f"maps {text}")

    # Mapping from the midi control value to the function to update and the value to update to
    def get_midi_mapping():
        midi_mapping = {
            1: {"function": change_layer, "value": f"{overstromingen[0]},layer"},
            2: {"function": change_layer, "value": "ospar,overlay"},
            3: {"function": change_year, "value": ["2023", "2050", "2100"]},
            7: {"function": change_year, "value": ["2023", "2100"]},
            23: {"function": change_layer, "value": "bathymetry,layer"},
            24: {"function": change_layer, "value": f"{waterdiepte[0]},layer"},
            25: {"function": change_layer, "value": f"{risico_zone[0]},layer"},
            26: {"function": change_layer, "value": "aangepast_bouwen,layer"},
            27: {"function": change_layer, "value": f"bescherming,layer"},
            28: {"function": change_layer, "value": f"compartiment,layer"},
            29: {"function": change_layer, "value": "schuilen,layer"},
            30: {"function": change_layer, "value": "c_management,layer"},
            31: {"function": change_layer, "value": "doorbraaklocaties,overlay"},
            # 28: {"function": change_layer, "value": "GLG,layer"},
            # 31: {"function": change_layer, "value": ",layer"},
            # 31: {"function": change_layer, "value": "difference,layer"},
            44: {"function": change_layer, "value": "owf_2040,overlay"},
            45: {"function": start_stop_animation, "value": "animation,layer"},
            46: {"function": start_stop_animation, "value": ""},
            47: {"function": cycle_collection, "value": "prev"},
            48: {"function": cycle_collection, "value": "next"},
            60: {"function": slider_update},
            67: {"function": change_layer, "value": "owf_all,overlay"},
            64: {"function": change_layer, "value": "20,tide"},
        }
        return midi_mapping

    # List of used slider control values
    slider_keys = [3, 7, 60]
    inport = mido.open_input()
    for msg in inport:
        # If BANK button is pressed, disconnect midi board (can't reconnect)
        if msg.type == "sysex":
            inport.close()
            break
        else:
            try:
                midi_mapping = get_midi_mapping()
                # Send update if button is pressed
                if msg.value == 127 and msg.control not in slider_keys:
                    midi_mapping[msg.control]["function"](
                        midi_mapping[msg.control]["value"]
                    )
                # Send update if slider value is changed
                if msg.control in slider_keys:
                    # Some functions have another optional value, try that first
                    try:
                        midi_mapping[msg.control]["function"](
                            msg.value, midi_mapping[msg.control]["value"]
                        )
                    # Otherwise, function only needs slider value
                    except:
                        midi_mapping[msg.control]["function"](msg.value)
            except:
                continue


def hand_tracker(data_path):
    """Hand tracking module that publishes hand coordinates via ZMQ.

    This function initializes the webcam-based hand tracking system using MediaPipe.
    Detected hand positions are transformed to map coordinates based on the extent
    and published for display on the map.

    Args:
        datasets: Preprocessed dataset dictionary (not used - data is loaded internally).

    ZMQ Topics:
        - hands: Hand coordinates in format "x,y" (map coordinates)

    Note:
        Supports tracking up to 4 hands simultaneously. The calibrate parameter
        enables interactive calibration for mapping camera view to map extent.
        Publishes to port 5557.
    """
    datasets = load_preprocessed(data_path=data_path)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5557")

    socket_topic = "hands"

    extent = datasets[""]["extent"].bounds

    hand_tracking.webcam_module(
        device_index=0,
        extent=extent,
        socket=socket,
        socket_topic=socket_topic,
        max_number_of_hands=4,
        calibrate=True,
    )


def uid_detector(data_path):
    """Unique identifier (UID) detection module for interactive elements.

    This function runs AprilTag/QR code/ArUco marker detection on webcam input
    and publishes detected UIDs via ZMQ. The UIDs can trigger layer changes or
    display specific information when markers are detected.

    Args:
        datasets: Preprocessed dataset dictionary (not used - data is loaded internally).

    ZMQ Topics:
        - uid: Detected unique identifier string
        - slice: Slice position updates
        - hands: Hand position updates

    Note:
        Uses the extent to determine the spatial context for detections.
        Interactivity polygons from datasets define trigger regions.
        Publishes to port 5558. Exceptions are caught and printed but don't
        terminate the module.
    """
    try:
        datasets = load_preprocessed(data_path=data_path)
        context = zmq.Context()
        socket = context.socket(zmq.PUB)
        socket.setsockopt(zmq.CONFLATE, 1)
        socket.bind("tcp://*:5558")

        extent = datasets[""]["extent"].bounds

        uid_detection.main(
            socket=socket, extent=extent, datasets=datasets[""]["interactivity"]
        )
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # Alternative entry points for testing:
    # input_file = Path(__file__).parent / "input.json"
    # datasets = preprocess.preprocess(input_file=input_file)
    # displaymap(datasets=datasets)
    main()
