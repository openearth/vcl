import concurrent.futures
import os
import sys
import threading
import time
from pathlib import Path
import matplotlib as mpl

import cmocean

# import pywinctl as gw
import numpy as np
import pygame
import rasterio
import rioxarray as rxr
import zmq
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.transform import Affine, from_bounds
from rasterio.warp import reproject
from shapely.geometry import mapping

# from vcl.windows import DisplayMap, DisplaySlice
from vcl.windows import DisplayMap, DisplaySlice
import vcl.preprocess

contour_show = False
height_map_show = False
compare = False
current_layer = ""
current_overlay = ""


def make_listen_sockets():
    context = zmq.Context()

    socket1 = context.socket(zmq.SUB)
    socket1.setsockopt(zmq.CONFLATE, 1)
    socket1.connect("tcp://localhost:5556")
    socket1.subscribe("pygame_1")

    socket2 = context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:5556")
    socket2.subscribe("pygame_2")

    socket3 = context.socket(zmq.SUB)
    socket3.setsockopt(zmq.CONFLATE, 1)
    socket3.connect("tcp://localhost:5556")
    socket3.subscribe("slice")

    socket4 = context.socket(zmq.SUB)
    socket4.setsockopt(zmq.CONFLATE, 1)
    socket4.connect("tcp://localhost:5556")
    socket4.subscribe("year")

    poller = zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)
    poller.register(socket3, zmq.POLLIN)
    poller.register(socket4, zmq.POLLIN)

    sockets = {
        "context": context,
        "pygame_1": socket1,
        "pygame_2": socket2,
        "year": socket4,
        "slice": socket3,
        "poller": poller,
    }
    return sockets


def displaymap(datasets):
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    # from matplotlib.colors import Normalize

    # norm = Normalize(vmin=-6, vmax=40)

    dataset_kwargs = {
        "basemap": {"type": "RGB"},
        "bathymetry": {
            "type": "CMAP",
            "text": "Hoogtekaart",
            "text_color": (255, 255, 255),
            "cmap": mpl.colormaps.get_cmap("viridis"),
        },
        "fishery": {"type": "RGB", "text": "Fishery", "alpha": 0.6},
        "navisafe": {"type": "RGB", "text": "Navisafe"},
    }

    socket = sockets["pygame_1"]
    socket_slice = sockets["slice"]
    socket_year = sockets["year"]
    display = DisplayMap.DisplayMap(
        datasets=datasets,
        start_year="1970",
        flow_data=None,
        dataset_kwargs=dataset_kwargs,
        bg_layer="basemap",
        i_max=300,
    )

    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket in socks and socks[socket] == zmq.POLLIN:
            topic, message = socket.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type = message.split(",")
            if view_type == "overlay":
                display.init_arrowmanager(layer)
            else:
                display.change_layer(layer)

        if socket_slice in socks and socks[socket_slice] == zmq.POLLIN:
            topic, message = socket_slice.recv(zmq.DONTWAIT).split()
            slice_index = int(message)
            display.change_line_index(slice_index)

        if socket_year in socks and socks[socket_year] == zmq.POLLIN:
            topic, message = socket_year.recv(zmq.DONTWAIT).split()
            year = message.decode("utf-8")
            display.change_year(year)
        display.draw_layers()


def displayslice(datasets):
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
        global current_layer, current_overlay
        if text.split(",")[0] == "":
            socket.send_string(f"pygame_1 {text}")
            current_layer = ""
            current_overlay = ""
        if current_layer == text or current_overlay == text:
            socket.send_string(f"pygame_1 None,{layer_type}")
            if layer_type == "layer":
                current_layer = ""
            elif layer_type == "overlay":
                current_overlay = ""
        else:
            socket.send_string(f"pygame_1 {text}")
            if layer_type == "layer":
                current_layer = text
            elif layer_type == "overlay":
                current_overlay = text

    # --- Pygame Setup ---
    pygame.init()
    screen_width, screen_height = 400, 200
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Keyboard Publisher")
    font = pygame.font.Font(None, 36)
    keys_held = {}  # Dictionary to track held keys
    # --- Main Loop -
    slice_index = 0
    max_slices = 300
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
                    change_layer("fishery,layer")
                elif event.key == pygame.K_3:
                    change_layer("navisafe,layer")
                elif event.key == pygame.K_4:
                    change_layer("conc,layer")
                elif event.key == pygame.K_a:
                    change_layer("animation,layer")
                elif event.key == pygame.K_q:
                    change_layer("170,overlay")
                elif event.key == pygame.K_e:
                    change_layer("390,overlay")
                elif event.key == pygame.K_i:
                    change_year(2023)
                elif event.key == pygame.K_o:
                    change_year(2050)
                elif event.key == pygame.K_p:
                    change_year(2100)
                if event.key == pygame.K_h:
                    print("Sending 'instance_1' message")
                    # Send a message indicating an event for the first instance
                    socket.send_string("pygame_1 instance_1")
                elif event.key == pygame.K_j:
                    print("Sending 'instance_2' message")
                    # Send a message indicating an event for the second instance
                    socket.send_string("pygame_2 instance_2")
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
    thread = threading.Thread(daemon=True)
    thread.start()


def main():
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


if __name__ == "__main__":
    # input_file = Path(__file__).parent / "input.json"
    # datasets = preprocess.preprocess(input_file=input_file)
    # displaymap(datasets=datasets)
    main()
