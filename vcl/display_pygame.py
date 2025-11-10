import concurrent.futures
import os
import sys
import threading
import time
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import mido

# import pywinctl as gw
import pygame
import zmq
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb

import vcl.preprocess

# from vcl.windows import DisplayMap, DisplaySlice
from vcl.windows import DisplayMap, DisplaySlice, StatsWindow

contour_show = False
height_map_show = False
compare = False
current_layer = ""
current_overlay = ""

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
]

bathymetry_cmap = LinearSegmentedColormap.from_list(
    "bathy_cmap", bathymetry_cmap, N=5000
)


def make_listen_sockets():
    context = zmq.Context()

    socket1 = context.socket(zmq.SUB)
    socket1.setsockopt(zmq.CONFLATE, 1)
    socket1.connect("tcp://localhost:5556")
    socket1.subscribe("maps")

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
        "maps": socket1,
        "pygame_2": socket2,
        "year": socket4,
        "slice": socket3,
        "poller": poller,
    }
    return sockets


def displaymap(datasets):
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    from matplotlib.colors import Normalize

    norm = Normalize(vmin=-4000, vmax=0)

    dataset_kwargs = {
        "basemap": {"type": "RGB"},
        "bathymetry": {
            "type": "CMAP",
            "text": "Hoogtekaart",
            "text_color": (255, 255, 255),
            "cmap": bathymetry_cmap,
            "norm": norm,
        },
        "fishery": {"type": "RGB", "text": "Fishing effort", "alpha": 0.6},
        "fishing_catch": {"type": "RGB", "text": "Fishing catch", "alpha": 0.6},
        "navisafe": {"type": "RGB", "text": "Navisafe"},
        "windfarms": {"type": "RGB", "text": "Windfarms", "cmap": windfarm_cmap},
        "mask": {
            "type": "CMAP",
            "text": "",
            "cmap": ListedColormap(["gray", "orange"]),
        },
        "cods": {"type": "RGB", "text": "Cods"},
    }

    socket = sockets["maps"]
    socket_slice = sockets["slice"]
    socket_year = sockets["year"]

    display = DisplayMap.DisplayMap(
        datasets=datasets,
        start_year="1970",
        flow_data=None,
        animations_data=datasets[""]["animations"],
        dataset_kwargs=dataset_kwargs,
        bg_layer="basemap",
        mask_layer="mask",
        i_max=127,
    )
    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket in socks and socks[socket] == zmq.POLLIN:
            topic, message = socket.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type = message.split(",")
            if view_type == "overlay":
                # display.init_arrowmanager(layer)
                display.display_mask()
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


def displaystats(datasets):
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    socket = sockets["maps"]
    socket_slice = sockets["slice"]
    socket_year = sockets["year"]

    dataset_kwargs = {
        "fishery": {"image": {"title": ""}},
        "fishing_catch": {"image": {"title": ""}},
    }
    display = StatsWindow.StatsWindow(
        datasets[""]["stats"],
        dataset_kwargs=dataset_kwargs,
        layers_to_ignore=["mask", "animation"],
    )

    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket in socks and socks[socket] == zmq.POLLIN:
            topic, message = socket.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type = message.split(",")
            display.change_layer(layer)
        plt.pause(0.01)


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
                    change_layer("fishing_catch,layer")
                elif event.key == pygame.K_4:
                    change_layer("navisafe,layer")
                elif event.key == pygame.K_5:
                    change_layer("windfarms,layer")
                elif event.key == pygame.K_6:
                    change_layer("cods,layer")
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
                elif event.key == pygame.K_m:
                    change_layer("mask,overlay")
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


def midi_board(datasets):
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
        global current_layer, current_overlay
        if text.split(",")[0] == "":
            socket.send_string(f"maps {text}")
            current_layer = ""
            current_overlay = ""
        if current_layer == text or current_overlay == text:
            socket.send_string(f"maps None,{layer_type}")
            if layer_type == "layer":
                current_layer = ""
            elif layer_type == "overlay":
                current_overlay = ""
        else:
            socket.send_string(f"maps {text}")
            if layer_type == "layer":
                current_layer = text
            elif layer_type == "overlay":
                current_overlay = text

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
            # 1: {"function": change_scenario, "value": "prev"},
            # 2: {"function": change_scenario, "value": "next"},
            3: {"function": change_year, "value": ["2023", "2050", "2100"]},
            7: {"function": change_year, "value": ["2023", "2100"]},
            23: {"function": change_layer, "value": "bathymetry,layer"},
            24: {"function": change_layer, "value": "navisafe,layer"},
            25: {"function": change_layer, "value": "fishery,layer"},
            26: {"function": change_layer, "value": "fishing_catch,layer"},
            27: {"function": change_layer, "value": "windfarms,layer"},
            28: {"function": change_layer, "value": "cods,layer"},
            31: {"function": change_layer, "value": "mask,overlay"},
            # 28: {"function": change_layer, "value": "GLG,layer"},
            # 31: {"function": change_layer, "value": ",layer"},
            # 31: {"function": change_layer, "value": "difference,layer"},
            45: {"function": start_stop_animation, "value": "animation,layer"},
            46: {"function": start_stop_animation, "value": ""},
            60: {"function": slider_update},
            64: {"function": change_layer, "value": "tidal_flows:170,overlay"},
            67: {"function": change_layer, "value": "tidal_flows:390,overlay"},
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


if __name__ == "__main__":
    # input_file = Path(__file__).parent / "input.json"
    # datasets = preprocess.preprocess(input_file=input_file)
    # displaymap(datasets=datasets)
    main()
