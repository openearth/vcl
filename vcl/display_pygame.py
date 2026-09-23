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

import collections
import concurrent.futures
import logging
import os
import sys
import threading
import time
from pathlib import Path
import geopandas as gpd
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import mido

# import pywinctl as gw
import pygame
import zmq
from matplotlib.colors import (
    LinearSegmentedColormap,
    ListedColormap,
    to_rgb,
    from_levels_and_colors,
)

try:
    from pynput import keyboard as pynput_keyboard
except ImportError as e:
    pynput_keyboard = None

import vcl.preprocess

# from vcl.windows import DisplayMap, DisplaySlice
from vcl.windows import DisplayMap, DisplaySlice, StatsWindow
from vcl.utils import hand_tracking
from vcl.load_data import load_preprocessed
from vcl.interactivity import uid_detection
from vcl.input_handlers.keyboard import keyboard_publisher
from vcl.input_handlers.museum import museum_button_publisher

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

logger = logging.getLogger(__name__)

# Custom colormap for bathymetry (depth) visualization
# Color scale transitions from deep blue (deep water) to yellow/red (shallow/land)
bathymetry_cmap = [
    (0, (10 / 255, 28 / 255, 92 / 255)),
    (0.13, (10 / 255, 173 / 255, 127 / 255)),  # 0m
    (0.2, (24 / 255, 181 / 255, 81 / 255)),  # 10m
    (0.5, (240 / 255, 233 / 255, 50 / 255)),
    (1, (237 / 255, 189 / 255, 92 / 255)),  # 20m and above
]

# Create continuous colormap with 5000 discrete steps for smooth gradients
bathymetry_cmap = LinearSegmentedColormap.from_list("bathy_cmap", bathymetry_cmap, N=20)

gvg_colors, gvg_levels = (
    [
        "#004da8",
        "#267300",
        "#38a800",
        "#4ce600",
        "#55ff00",
        "#a3ff73",
        "#d1ff73",
        "#ffffbe",
        "#feff73",
        "#feff00",
        "#fedd33",
        "#fec414",
        "#febf0a",
        "#feaa00",
        "#fe8c00",
        "#fe7300",
        "#ff5500",
        "#ff2a00",
        "#ed0000",
        "#d90000",
        "#bf0000",
        "#a60000",
        "#730000",
        "#4b0000",
    ],
    [
        0.0,
        0.2,
        0.4,
        0.6,
        0.8,
        1.0,
        1.2,
        1.4,
        1.6,
        1.8,
        2.0,
        2.2,
        2.4,
        2.6,
        2.8,
        3.0,
        3.2,
        3.4,
        3.6,
        3.8,
        4.0,
        6.0,
        10.0,
    ],
)
gvg_cmap, gvg_norm = from_levels_and_colors(gvg_levels, gvg_colors, extend="both")

gvg_difference_colors, gvg_difference_levels = (
    [
        "#730000",
        "#ca0000",
        "#ff6600",
        "#ecbd00",
        "#ffe375",
        "#ffffb3",
        "#c8c8c8",
        "#d2ffff",
        "#8ce8ff",
        "#00bbea",
        "#0066ff",
        "#000099",
        "#000073",
    ],
    [-2.0, -1.0, -0.5, -0.25, -0.1, -0.05, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0],
)
gvg_difference_cmap, gvg_difference_norm = from_levels_and_colors(
    gvg_difference_levels, gvg_difference_colors, extend="both"
)

colors = [
    "#0b0887",
    "#1122ff",
    "#4f8be6",
    "#63c7d8",
    "#8bdc7b",
    "#d7e44d",
    "#f0c648",
    "#ee8d33",
    "#ef4b22",
    "#c21f0f",
]
grensvlak_cmap = LinearSegmentedColormap.from_list("grensvlak_cmap", colors, N=2000)

land_use_cmap = ListedColormap(
    [
        "#4E9A51",  # 1 natuur
        "#E6C65C",  # 2 landbouw
        "#4FC3F7",  # 3 recreatie
        "#BDBDBD",  # 4 bebouwing
    ]
)


def build_dataset_kwargs(datasets: dict):
    """Build layer display configuration from preprocessed datasets.

    This avoids maintaining large hard-coded layer maps by inspecting the first
    dataset group and selecting only ndarray-based layers.
    """
    excluded_layers = {
        "extent",
        "mid_point",
        "angle",
        "crs",
        "stats",
        "animations",
        "particles",
        "interactivity",
    }

    first_group_key = next(iter(datasets.keys()))
    group = datasets[first_group_key]
    dataset_kwargs = {}

    for layer_name, layer_value in group.items():
        if layer_name in excluded_layers:
            continue
        if not isinstance(layer_value, np.ndarray):
            continue

        if layer_name == "bathymetry":
            dataset_kwargs[layer_name] = {
                "type": "CMAP",
                "cmap": bathymetry_cmap,
                "norm": mpl.colors.Normalize(vmin=-6, vmax=40),
            }
        else:
            dataset_kwargs[layer_name] = {"type": "RGB", "alpha": 0.7}

    return dataset_kwargs


def build_display_dataset_kwargs(datasets: dict):
    """Build display kwargs while preserving specialized styling for known layers."""
    dataset_kwargs = build_dataset_kwargs(datasets)

    layer_overrides = {
        "basemap": {"type": "RGB", "alpha": 1},
        "bathymetry": {
            "type": "CMAP",
            "cmap": bathymetry_cmap,
            "norm": mpl.colors.Normalize(vmin=-6, vmax=40),
        },
        "salt_concentration": {"type": "RGB", "alpha": 0.7},
        "gvg": {"type": "CMAP", "alpha": 1.0, "cmap": gvg_cmap, "norm": gvg_norm},
        "gvg_difference": {
            "type": "CMAP",
            "alpha": 1.0,
            "cmap": gvg_difference_cmap,
            "norm": gvg_difference_norm,
        },
        "grensvlak": {
            "type": "CMAP",
            "alpha": 1.0,
            "cmap": grensvlak_cmap,
            "norm": mpl.colors.Normalize(vmin=-100, vmax=5),
        },
        "land_use": {
            "type": "CMAP",
            "alpha": 1.0,
            "cmap": land_use_cmap,
        },
    }

    for layer_name, overrides in layer_overrides.items():
        if layer_name in dataset_kwargs:
            dataset_kwargs[layer_name] = dataset_kwargs[layer_name] | overrides

    return dataset_kwargs


def build_museum_keyboard_layer_map(dataset_kwargs: dict):
    """Map number keys to the first ten non-basemap layers for local testing."""
    keys = ("1", "2", "3", "4", "5", "6", "7", "8", "9", "0")
    layers = [layer for layer in dataset_kwargs if layer != "basemap"]
    return dict(zip(keys, layers[: len(keys)]))


def cycle_museum_layer(layer_names, current_layer, step):
    """Return the next or previous test layer name."""
    if not layer_names:
        return None

    if current_layer not in layer_names:
        return layer_names[0]

    current_index = layer_names.index(current_layer)
    next_index = (current_index + step) % len(layer_names)
    return layer_names[next_index]


def handle_museum_keyboard_input(event, display, keyboard_layer_map):
    """Handle focused-window keyboard shortcuts for museum-mode testing."""
    if event.type != pygame.KEYDOWN:
        return False

    key_to_label = {
        pygame.K_1: "1",
        pygame.K_2: "2",
        pygame.K_3: "3",
        pygame.K_4: "4",
        pygame.K_5: "5",
        pygame.K_6: "6",
        pygame.K_7: "7",
        pygame.K_8: "8",
        pygame.K_9: "9",
        pygame.K_0: "0",
    }
    if event.key in key_to_label:
        layer_name = keyboard_layer_map.get(key_to_label[event.key])
        if layer_name is None:
            return False
        display.change_layer(layer_name)
        return True

    layer_names = list(keyboard_layer_map.values())
    if event.key == pygame.K_LEFTBRACKET:
        next_layer = cycle_museum_layer(layer_names, display.current_layer, -1)
        if next_layer is None:
            return False
        display.change_layer(next_layer)
        return True

    if event.key == pygame.K_RIGHTBRACKET:
        next_layer = cycle_museum_layer(layer_names, display.current_layer, 1)
        if next_layer is None:
            return False
        display.change_layer(next_layer)
        return True

    if event.key == pygame.K_MINUS:
        display.change_layer(None)
        return True

    return False


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


def displaymap(
    data_path,
    museum_mode=False,
    inactivity_timeout=120.0,
    default_layer="overview",
    render_fps=60,
    year_loop_fps=1.0,
):
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

    dataset_kwargs = build_dataset_kwargs(datasets)
    dataset_kwargs = {
        "basemap": {"type": "RGB", "alpha": 1},
        "bathymetry": {
            "type": "CMAP",
            "cmap": bathymetry_cmap,
            "norm": mpl.colors.Normalize(vmin=-6, vmax=40),
        },
        "salt_concentration": {"type": "RGB", "alpha": 0.7},
        "gvg": {"type": "CMAP", "alpha": 1.0, "cmap": gvg_cmap, "norm": gvg_norm},
        "gvg_difference": {
            "type": "CMAP",
            "alpha": 1.0,
            "cmap": gvg_difference_cmap,
            "norm": gvg_difference_norm,
        },
        "grensvlak": {
            "type": "CMAP",
            "alpha": 1.0,
            "cmap": grensvlak_cmap,
            "norm": mpl.colors.Normalize(vmin=-100, vmax=5),
        },
        "land_use": {
            "type": "CMAP",
            "alpha": 0.5,
            "cmap": land_use_cmap,
        },
    }
    socket = sockets["maps"]
    socket_slice = sockets["slice"]
    socket_year = sockets["year"]
    socket_hands = sockets["hands"]
    try:
        display = DisplayMap.DisplayMap(
            datasets=datasets,
            start_year="1970",
            flow_data=datasets[""]["particles"]["current"],
            animations_data=datasets[""]["animations"],
            sounds=datasets[""]["sounds"],
            dataset_kwargs=dataset_kwargs,
            bg_layer="basemap",
            mask_layer=None,
            i_max=127,
            aspect_ratio=1920 / 1080,
            target_fps=render_fps,
        )
    except Exception as e:
        print(e)
        return

    if museum_mode:
        display.enter_fullscreen()
        display.disable_mouse()

    if default_layer in dataset_kwargs:
        display.change_layer(default_layer)
    if default_layer in datasets[""]["animations"]:
        display.play_animation(default_layer)

    available_years = sorted([year for year in datasets.keys() if year != ""])
    auto_year_loop = museum_mode and len(available_years) > 1 and year_loop_fps > 0
    year_loop_interval = 1.0 / year_loop_fps if auto_year_loop else None
    next_year_switch = time.monotonic() + year_loop_interval if auto_year_loop else None
    current_year_index = 0
    museum_keyboard_layer_map = build_museum_keyboard_layer_map(dataset_kwargs)

    if museum_mode:
        logger.info(
            "Museum keyboard shortcuts enabled: %s | '[' and ']' cycle layers | '-' clears to basemap",
            ", ".join(
                f"{key}={value}" for key, value in museum_keyboard_layer_map.items()
            )
            or "no numbered layer shortcuts available",
        )

    if auto_year_loop:
        display.change_year(available_years[current_year_index])

    # display.init_arrowmanager(170)

    last_activity = time.monotonic()

    def mark_activity():
        nonlocal last_activity
        last_activity = time.monotonic()

    coords = None
    while True:
        if museum_mode:
            for event in pygame.event.get([pygame.KEYDOWN]):
                if handle_museum_keyboard_input(
                    event, display, museum_keyboard_layer_map
                ):
                    mark_activity()

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
            elif view_type == "animation":
                display.play_animation(layer)
            else:
                display.change_layer(layer)
            mark_activity()

        if socket_slice in socks and socks[socket_slice] == zmq.POLLIN:
            topic, message = socket_slice.recv(zmq.DONTWAIT).split()
            slice_index = float(message)
            display.change_line_index(slice_index)
            mark_activity()

        if socket_year in socks and socks[socket_year] == zmq.POLLIN:
            topic, message = socket_year.recv(zmq.DONTWAIT).split()
            year = message.decode("utf-8")
            display.change_year(year)
            mark_activity()

        if (
            not museum_mode
            and socket_hands in socks
            and socks[socket_hands] == zmq.POLLIN
        ):
            topic, coords = socket_hands.recv(zmq.DONTWAIT).split()
            coords = coords.decode("utf-8")
            xcoord, ycoord = coords.split(",")
            xcoord = float(xcoord)
            ycoord = float(ycoord)
            coords = (xcoord, ycoord)
            display.start_hand_tracking(coords)
            mark_activity()

        now = time.monotonic()
        if auto_year_loop and now >= next_year_switch:
            current_year_index = (current_year_index + 1) % len(available_years)
            display.change_year(available_years[current_year_index])
            next_year_switch = now + year_loop_interval

        global current_layer
        if (
            museum_mode
            and inactivity_timeout > 0
            and now - last_activity >= inactivity_timeout
            and (display.current_layer != default_layer or display.show_animation)
        ):
            if default_layer == "basemap":
                display.change_layer(None)
            elif (
                default_layer in datasets[""]["animations"]
                and display.show_animation == False
            ):
                display.play_animation(default_layer)
                current_layer = "satellite"
            # else:
            #     display.change_layer(default_layer)
            last_activity = now

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

    # dataset_kwargs = {
    #     "fishery": {"image": {"title": ""}},
    #     "fishing_catch": {"image": {"title": ""}},
    # }
    display = StatsWindow.StatsWindow(
        datasets[""]["stats"],
        dataset_kwargs={},
        layers_to_ignore=["mask", "animation", "20", "30"],
        overlay_layers=[
            "eez",
            "vessel-traffic",
            "ospar",
            "owf_2030",
            "owf_2040",
            "owf_all",
        ],
    )

    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket in socks and socks[socket] == zmq.POLLIN:
            topic, message = socket.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type = message.split(",")
            display.change_layer(layer)
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
    try:
        hand_tracking.webcam_module(
            device_index=0,
            extent=extent,
            socket=socket,
            socket_topic=socket_topic,
            max_number_of_hands=4,
            calibrate=True,
        )
    except Exception as e:
        print(e)


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
    datasets = load_preprocessed(data_path=data_path)
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5558")

    extent = datasets[""]["extent"].bounds

    try:
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
