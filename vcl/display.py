import collections
import itertools
import pathlib

import cmocean
import cv2
import imod
import matplotlib
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import mido
import numpy as np
import scipy
import zmq
from matplotlib import colormaps
from matplotlib.colors import (
    LightSource,
    LinearSegmentedColormap,
    ListedColormap,
    from_levels_and_colors,
)
from matplotlib.widgets import Button, Slider
from mpl_toolkits.axes_grid1 import make_axes_locatable

import vcl.data
import vcl.DisplayMap
import vcl.DisplaySlice
import vcl.prep_data

cmap = ListedColormap(["royalblue", "coral"])
cmap = ListedColormap([cmocean.cm.haline(0), cmocean.cm.haline(0.99)])
cmap_n = ListedColormap(["royalblue", "coral", "red"])
cmap_n = ListedColormap([cmocean.cm.haline(0), cmocean.cm.haline(0.99), "red"])
cmap_tidal = colormaps.get_cmap("plasma")

# Define the colors for the blue-green-orange gradient
colors = [(0, 0, 1), (0, 1, 0), (1, 0.5, 0)]  # Blue -> Green -> Orange
cmap_colors = [
    (0, (10 / 255, 28 / 255, 92 / 255)),
    (0.13, (10 / 255, 173 / 255, 127 / 255)),  # 0m
    (0.2, (24 / 255, 181 / 255, 81 / 255)),  # 10m
    (0.5, (240 / 255, 233 / 255, 50 / 255)),
    (1, (237 / 255, 189 / 255, 92 / 255)),  # 20m and above
]
from matplotlib.colors import Normalize

norm = Normalize(vmin=-6, vmax=40)
n_bins = [20]  # Number of bins for each color

# Create the colormap
cmap_bodem = "custom_blue_green_orange"
cmap_bodem = LinearSegmentedColormap.from_list(cmap_bodem, cmap_colors, N=sum(n_bins))

src_dir = pathlib.Path(
    r"P:\11210262-001-stakeholders-tools\Virtual_Climate_Lab\01_data\dataset\new\Freatische GXG"
)
colors, levels = imod.visualize.read_imod_legend(src_dir / "residu_detail.leg")
cmap_GXG, norm_GXG = from_levels_and_colors(levels, colors, extend="both")


contour_show = False
height_map_show = False
compare = False
current_layer = ""
current_overlay = ""
matplotlib.rcParams["toolbar"] = "None"


def make_listen_sockets():
    context = zmq.Context()

    socket1 = context.socket(zmq.SUB)
    socket1.setsockopt(zmq.CONFLATE, 1)
    socket1.connect("tcp://localhost:5556")
    socket1.subscribe("x_slice")

    socket2 = context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:5556")
    socket2.subscribe("top_view")

    socket3 = context.socket(zmq.SUB)
    socket3.setsockopt(zmq.CONFLATE, 1)
    socket3.connect("tcp://localhost:5556")
    socket3.subscribe("x_slice")

    socket4 = context.socket(zmq.SUB)
    socket4.setsockopt(zmq.CONFLATE, 1)
    socket4.connect("tcp://localhost:5556")
    socket4.subscribe("year")

    socket5 = context.socket(zmq.SUB)
    socket5.setsockopt(zmq.CONFLATE, 1)
    socket5.connect("tcp://localhost:5556")
    socket5.subscribe("scenario")

    poller = zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)
    poller.register(socket3, zmq.POLLIN)
    poller.register(socket4, zmq.POLLIN)
    poller.register(socket5, zmq.POLLIN)

    sockets = {
        "context": context,
        "x_slice": socket1,
        "top_view": socket2,
        "x_slice_c": socket3,
        "year": socket4,
        "scenario": socket5,
        "poller": poller,
    }
    return sockets


def satellite_window(datasets):
    # Retrieve sockets for communication
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    # Socket for slider
    socket1 = sockets["x_slice"]
    # Socket for changing layer
    socket2 = sockets["top_view"]
    # Socket for changing scenario
    socket3 = sockets["year"]
    socket4 = sockets["scenario"]

    # Get plot lims
    xmin, ymin, xmax, ymax = datasets["2023"]["plt_lims"]

    # Get extent of different layers
    sat_extent = datasets["2023"]["sat_extent"]
    xmin_b, ymin_b, xmax_b, ymax_b = datasets["2023"]["bodem_bounds"]
    xmin_e, ymin_e, xmax_e, ymax_e = datasets["2023"]["ecotoop_extent"]
    xmin_gsr, ymin_gsr, xmax_gsr, ymax_gsr = datasets["2023"]["GSR_extent"]
    # xmin_gxg, ymin_gxg, xmax_gxg, ymax_gxg = datasets["2023"]["GXG_extent"]
    xmin_gxg, ymin_gxg, xmax_gxg, ymax_gxg = datasets["2023"]["ssp_nat"]["GXG_extent"]
    xmin_f, ymin_f, xmax_f, ymax_f = datasets["2023"]["floodmap_extent"]

    # Get mid point and rotation angle for transform
    angle = datasets["2023"]["angle"]
    mid_point = datasets["2023"]["mid_point"]
    # Define rotation transform
    transform = mtransforms.Affine2D().rotate_deg_around(
        mid_point[0], mid_point[1], -angle
    )

    print("satellite window")

    # Create animation frames from file paths (ideally in prep_data, but gave errors so for now moved to here)
    animation_files = datasets["2023"]["animation_data"]
    animation_data = {}
    for i, frame in enumerate(animation_files):
        animation_data[i] = vcl.data.get_frame_data(frame)

    datasets["2023"]["animation_data"] = animation_data
    datasets["2050"]["animation_data"] = animation_data
    datasets["2100"]["animation_data"] = animation_data
    # datasets["2100"]["GXG"] = datasets["2100"]["GXG"] - datasets["2023"]["GXG"]

    # Define dictionary with plot kwargs for the different layers
    maps_2023 = {
        "sat": {"extent": sat_extent, "zorder": 0, "label": ""},
        "conc_contour_top_view": {
            "extent": (xmin, xmax, ymin, ymax),
            "cmap": cmap,
            "alpha": 0.5,
            "zorder": 2,
            "label": "Concentratie zout in grondwater",
        },
        "GSR": {
            "extent": (xmin_gsr, xmax_gsr, ymin_gsr, ymax_gsr),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "label": "Recreatieterreinen",
        },
        "bodem": {
            "extent": (xmin_b, xmax_b, ymin_b, ymax_b),
            "vmin": -6,
            "vmax": 40,
            "cmap": cmap_bodem,
            "alpha": 1,
            "transform": transform,
            "zorder": 1,
            "label": "Hoogtekaart",
        },
        "ecotoop": {
            "extent": (xmin_e, xmax_e, ymin_e, ymax_e),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "label": "",
        },
        "GVG": {
            "extent": (xmin_gxg, xmax_gxg, ymin_gxg, ymax_gxg),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "label": "GVG",
            "cmap": cmap_GXG,
            "norm": norm_GXG,
        },
        "GLG": {
            "extent": (xmin_gxg, xmax_gxg, ymin_gxg, ymax_gxg),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "label": "GLG",
            "cmap": cmap_GXG,
            "norm": norm_GXG,
        },
        "GHG": {
            "extent": (xmin_gxg, xmax_gxg, ymin_gxg, ymax_gxg),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "label": "GHG",
            "cmap": cmap_GXG,
            "norm": norm_GXG,
        },
        "GXG": {
            "extent": (xmin_gxg, xmax_gxg, ymin_gxg, ymax_gxg),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "label": "GXG",
            "cmap": cmap_GXG,
            "norm": norm_GXG,
        },
        "floodmap": {
            "extent": (xmin_f, xmax_f, ymin_f, ymax_f),
            "alpha": 0.7,
            "transform": transform,
            "zorder": 1,
            "cmap": "RdBu",
            "label": "Floodmap",
        },
        "animation_data": {"transform": transform},
        "tidal_flows": {
            "transform": transform,
            "scale": 30,
            "minshaft": 2,
            "cmap": cmap_tidal,
            "zorder": 2,
            "label": "Getijdenstroom",
        },
    }

    maps_2050 = maps_2023.copy()
    maps_2100 = maps_2023.copy()
    # maps_2100["GXG"] = {
    #     "extent": (xmin_gxg, xmax_gxg, ymin_gxg, ymax_gxg),
    #     "alpha": 0.7,
    #     "vmin": np.nanpercentile(datasets["2100"]["GXG"], 10),
    #     "vmax": np.nanpercentile(datasets["2100"]["GXG"], 90),
    #     "transform": transform,
    #     "zorder": 1,
    #     "label": "Verschil GXG met nu",
    # }

    maps = {"2023": maps_2023, "2050": maps_2050, "2100": maps_2100}
    start_scenario = "ssp_ref"
    scenario_layers = ["conc_contour_top_view", "GXG", "GVG", "GLG"]

    # Create display window with satellite image
    display = vcl.DisplayMap.DisplayMap(
        datasets,
        maps,
        "sat",
        datasets["2023"]["plt_lims"],
        "2023",
        start_scenario=start_scenario,
        scenario_layers=scenario_layers,
        transform=transform,
    )

    # Start position for vertical line
    init_x = xmin

    # Pause so you can move window around
    plt.pause(10)
    # interactive
    plt.ion()
    print("image shown")
    plt.show(block=False)

    # Plot vertical lines on display
    (line_blue,) = display.line_plot(
        [init_x, init_x], [ymin, ymax], color="#255070", linewidth=3, alpha=0.7
    )
    (line_white,) = display.line_plot(
        [init_x, init_x], [ymin, ymax], color="white", linewidth=1
    )

    plt.axis("off")
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show(block=False)
    plt.pause(0.1)

    # Loop to check if new message is received
    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update vertical line
        if socket1 in socks and socks[socket1] == zmq.POLLIN:
            topic, message = socket1.recv(zmq.DONTWAIT).split()
            slider_val = int(message)
            line_blue.set_xdata([slider_val, slider_val])
            line_white.set_xdata([slider_val, slider_val])

        # If button for new layer is pressed, display new layer
        if socket2 in socks and socks[socket2] == zmq.POLLIN:
            topic, message = socket2.recv(zmq.DONTWAIT).split()
            message = message.decode("utf-8")
            layer, view_type, message = message.split(",")
            if view_type == "layer":
                if layer == "":
                    display.change_layer()
                    display.change_overlay()
                elif message == "0":
                    display.change_layer()
                elif message == "1":
                    display.change_layer(layer)
            elif view_type == "overlay":
                if message == "0" or layer == "":
                    display.change_overlay()
                elif message == "1":
                    display.change_overlay(layer)

        # If button for different scenario is pressed, change scenario
        if socket3 in socks and socks[socket3] == zmq.POLLIN:
            topic, message = socket3.recv(zmq.DONTWAIT).split()
            year = message.decode("utf-8")
            display.change_year(year)

        if socket4 in socks and socks[socket4] == zmq.POLLIN:
            topic, message = socket4.recv(zmq.DONTWAIT).split()
            scenario = message.decode("utf-8")
            display.change_scenario(scenario)

        plt.pause(0.01)


def contour_slice_window(datasets):
    # Retrieve sockets for communication
    sockets = make_listen_sockets()
    poller = sockets["poller"]

    # Socket for slider
    socket1 = sockets["x_slice_c"]
    # Socket for changing layer
    socket2 = sockets["top_view"]
    # Socket for changing scenario
    socket3 = sockets["year"]
    socket4 = sockets["scenario"]

    print("starting matplotlib")

    # Get extent
    xmin, ymin, xmax, ymax = datasets["2023"]["plt_lims"]
    extent_x = (ymin, ymax, -140, 25.5)

    # Get smoothed bathymetry lines
    bodem = datasets["2023"]["smooth_bodem"]
    # Create x coordinates for plotting the lines
    yb0 = np.linspace(ymax, ymin, bodem.shape[0])

    # Change "bodem" in datasets to bathymetry lines (so the dict key corresponds to the dict key for satellite window)
    datasets["2023"]["bodem"] = bodem
    datasets["2050"]["bodem"] = datasets["2050"]["smooth_bodem"]

    # Define dictionary with plot kwargs for the different layers
    maps = {
        "bodem": {"x": yb0, "color": "#70543e", "linewidth": 3},
        "conc_contours_x": {
            "extent": extent_x,
            "aspect": "auto",
            "cmap": cmap,
            "vmin": 0,
            "vmax": 1.5,
        },
    }
    start_scenario = "ssp_ref"
    scenario_layers = ["conc_contours_x"]
    # Create display with concentration contours
    display = vcl.DisplaySlice.DisplaySlice(
        datasets,
        maps,
        "conc_contours_x",
        "2023",
        start_scenario,
        extent_x,
        datasets["2023"]["plt_lims"],
        scenario_layers=scenario_layers,
    )

    # Pause so you can move window around
    plt.pause(20)

    # Loop to check if new message is received
    while True:
        socks = dict(poller.poll(10))
        # If slider sends message, update contour or line
        if socket1 in socks and socks[socket1] == zmq.POLLIN:
            topic, message = socket1.recv(zmq.DONTWAIT).split()
            slider_val = int(message)
            bodem_val = int((slider_val - xmin) / datasets["2023"]["dx_bodem"])
            conc_val = int((slider_val - xmin) / datasets["2023"]["ssp_nat"]["dx_conc"])
            display.change_slice(conc_val, bodem_val)
            plt.pause(0.01)
            # fig.canvas.draw_idle()

        # This part is for line (bathymetry) data on the slice, no longer works for all scenarios and years, so disabled for now

        # If button for new layer is pressed, display new layer (if they are 3D)
        # if socket2 in socks and socks[socket2] == zmq.POLLIN:
        #     topic, message = socket2.recv(zmq.DONTWAIT).split()
        #     message = message.decode("utf-8")
        #     layer, view_type, message = message.split(",")
        #     if message == "0":
        #         display.change_line_data()
        #     if message == "1":
        #         if layer in maps.keys():
        #             display.change_line_data(layer)
        #         else:
        #             display.change_line_data()
        #             # if view_type == "overlay":
        #             #     import ipdb

        #             #     ipdb.set_trace()
        #             #     display.show_difference("2023")
        #  plt.pause(0.01)

        # If button for different scenario is pressed, change scenario
        if socket3 in socks and socks[socket3] == zmq.POLLIN:
            topic, message = socket3.recv(zmq.DONTWAIT).split()
            year = message.decode("utf-8")
            display.change_year(year)

        if socket4 in socks and socks[socket4] == zmq.POLLIN:
            topic, message = socket4.recv(zmq.DONTWAIT).split()
            scenario = message.decode("utf-8")
            display.change_scenario(scenario)

        plt.pause(0.01)


def slider_window(datasets):
    # Create publishing socket
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5556")

    # Get plot lims, for starting position slider
    xmin, ymin, xmax, ymax = datasets["2023"]["plt_lims"]
    init_x = xmin

    # Figure for slider and buttons
    fig, ax = plt.subplots()
    plt.axis("off")
    ax.set_axis_off()
    ax.set_frame_on(False)

    # Add axis for slider
    x_ax = fig.add_axes([0.25, 0.2, 0.5, 0.03])
    x_slider = Slider(
        ax=x_ax,
        label="x",
        valmin=int(xmin),
        valmax=int(xmax),
        valinit=init_x,
        valstep=100,
    )

    # Function to send layer when button pressed
    def change_layer(event, text):
        global current_layer
        if current_layer == text:
            socket.send_string(f"top_view {text},{0}")
            current_layer = ""
        else:
            socket.send_string(f"top_view {text},{1}")
            current_layer = text

    # The function to be called anytime a slider's value changes
    def update(val):
        socket.send_string("x_slice %d" % val)
        fig.canvas.draw_idle()

    # Create buttons
    contourax = fig.add_axes([0.8, 0.025, 0.1, 0.04])
    contour_button = Button(contourax, "Contour", hovercolor="0.600")

    height_map_ax = fig.add_axes([0.65, 0.025, 0.13, 0.04])
    height_map_button = Button(height_map_ax, "Hoogtekaart", hovercolor="0.600")

    ecotoop_ax = fig.add_axes([0.50, 0.025, 0.1, 0.04])
    ecotoop_button = Button(ecotoop_ax, "Ecotopen", hovercolor="0.600")

    GSR_ax = fig.add_axes([0.35, 0.025, 0.1, 0.04])
    GSR_button = Button(GSR_ax, "GSR", hovercolor="0.600")

    GVG_ax = fig.add_axes([0.20, 0.025, 0.1, 0.04])
    GVG_button = Button(GVG_ax, "GVG", hovercolor="0.600")

    scenario_ax = fig.add_axes([0.05, 0.025, 0.1, 0.04])
    scenario_button = Button(scenario_ax, "2050", hovercolor="0.600")

    animation_ax = fig.add_axes([0.05, 0.085, 0.1, 0.04])
    animation_button = Button(animation_ax, "animation", hovercolor="0.600")

    tidal_flow_eb_ax = fig.add_axes([0.20, 0.085, 0.1, 0.04])
    tidal_flow_eb_button = Button(tidal_flow_eb_ax, "Eb", hovercolor="0.600")

    tidal_flow_flow_ax = fig.add_axes([0.35, 0.085, 0.1, 0.04])
    tidal_flow_flow_button = Button(tidal_flow_flow_ax, "Vloed", hovercolor="0.600")

    GXG_ax = fig.add_axes([0.5, 0.085, 0.1, 0.04])
    GXG_button = Button(GXG_ax, "GXG", hovercolor="0.600")

    floodmap_ax = fig.add_axes([0.65, 0.085, 0.1, 0.04])
    floodmap_button = Button(floodmap_ax, "floodmap", hovercolor="0.600")

    # Call functions when buttons are pressed or slider is slid
    x_slider.on_changed(update)
    contour_button.on_clicked(
        lambda x: change_layer(x, "conc_contour_top_view,overlay")
    )
    height_map_button.on_clicked(lambda x: change_layer(x, "bodem,layer"))
    ecotoop_button.on_clicked(lambda x: change_layer(x, "ecotoop,layer"))
    GSR_button.on_clicked(lambda x: change_layer(x, "GSR,layer"))
    GVG_button.on_clicked(lambda x: change_layer(x, "GVG,layer"))
    animation_button.on_clicked(lambda x: change_layer(x, "animation_data,layer"))
    tidal_flow_eb_button.on_clicked(
        lambda x: change_layer(x, "tidal_flows:390,overlay")
    )
    tidal_flow_flow_button.on_clicked(
        lambda x: change_layer(x, "tidal_flows:170,overlay")
    )
    GXG_button.on_clicked(lambda x: change_layer(x, "GXG,layer"))
    floodmap_button.on_clicked(lambda x: change_layer(x, "floodmap,layer"))

    # Slightly different function for scenario button
    def change_scenario(event):
        # Get currently shown scenario and change scenario to other scenario when pressed
        # Update button text when pressed as well
        next_scenario = scenario_button.label.get_text()
        if next_scenario == "2023":
            socket.send_string("scenario 2023")
            scenario_button.label.set_text("2050")
            fig.canvas.draw_idle()
        else:
            socket.send_string("scenario 2050")
            scenario_button.label.set_text("2023")
            fig.canvas.draw_idle()

    scenario_button.on_clicked(change_scenario)

    plt.show()


def midi_board(datasets):
    # import ipdb

    # ipdb.set_trace()
    # Create publishing socket for sending midi board messages to the windows
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5556")

    xmin, ymin, xmax, ymax = datasets["2023"]["plt_lims"]

    # Max and min x values of the extent
    valmin = int(xmin)
    valmax = int(xmax)

    # Number of values for the sliders on midi board (0-127)
    n_slider_values = 128

    # Update slider based on 128 possible values
    def slider_update(value):
        slider_value = valmin + value * (valmax - valmin) / (n_slider_values - 1)
        socket.send_string(f"x_slice {int(slider_value)}")

    scenarios = collections.deque(["ref", "nat", "droog"])
    # scenarios = collections.deque(["nat", "droog"])

    # Slightly different function for scenario button
    def change_scenario(scenario):
        # Get currently shown scenario and change scenario to other scenario when pressed
        # Update button text when pressed as well

        if scenario == "next":
            scenarios.rotate(-1)
            next_scenario = scenarios[0]
        elif scenario == "prev":
            scenarios.rotate(1)
            next_scenario = scenarios[0]

        socket.send_string(f"scenario ssp_{next_scenario}")

    gxgs = collections.deque(["GLG", "GVG", "GHG"])

    def change_gxg(gxg):
        global current_layer

        if gxg == "next":
            gxgs.rotate(-1)
            next_gxg = gxgs[0]
        elif gxg == "prev":
            gxgs.rotate(1)
            next_gxg = gxgs[0]

        layer = f"{next_gxg},layer"

        if current_layer in [f"{gxg},layer" for gxg in gxgs]:
            change_layer(layer)

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
            socket.send_string(f"top_view {text},{0}")
            current_layer = ""
            current_overlay = ""
        if current_layer == text or current_overlay == text:
            socket.send_string(f"top_view {text},{0}")
            if layer_type == "layer":
                current_layer = ""
            elif layer_type == "overlay":
                current_overlay = ""
        else:
            socket.send_string(f"top_view {text},{1}")
            if layer_type == "layer":
                current_layer = text
            elif layer_type == "overlay":
                current_overlay = text

    # def change_overlay(text):
    #     global current_overlay
    #     if current_overlay

    def start_stop_animation(text):
        if text == "":
            socket.send_string(f"top_view {text},layer,{0}")
        else:
            socket.send_string(f"top_view {text},{1}")

    # Mapping from the midi control value to the function to update and the value to update to
    def get_midi_mapping():
        midi_mapping = {
            1: {"function": change_scenario, "value": "prev"},
            2: {"function": change_scenario, "value": "next"},
            3: {"function": change_year, "value": ["2023", "2050", "2100"]},
            7: {"function": change_year, "value": ["2023", "2100"]},
            23: {"function": change_layer, "value": "conc_contour_top_view,layer"},
            24: {"function": change_layer, "value": "GSR,layer"},
            25: {"function": change_layer, "value": f"{gxgs[0]},layer"},
            26: {"function": change_layer, "value": "ecotoop,layer"},
            27: {"function": change_layer, "value": "bodem,layer"},
            28: {"function": change_layer, "value": "floodmap,layer"},
            # 28: {"function": change_layer, "value": "GLG,layer"},
            # 31: {"function": change_layer, "value": ",layer"},
            # 31: {"function": change_layer, "value": "difference,layer"},
            45: {"function": start_stop_animation, "value": "animation_data,layer"},
            46: {"function": start_stop_animation, "value": ""},
            47: {"function": change_gxg, "value": "prev"},
            48: {"function": change_gxg, "value": "next"},
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
