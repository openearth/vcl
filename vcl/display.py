import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zmq
import cmocean
from matplotlib.colors import LightSource, ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.widgets import Slider, Button
import scipy

import vcl.prep_data

cmap = ListedColormap(["royalblue", "coral"])
cmap = ListedColormap([cmocean.cm.haline(0), cmocean.cm.haline(0.99)])
cmap_n = ListedColormap(["royalblue", "coral", "red"])
cmap_n = ListedColormap([cmocean.cm.haline(0), cmocean.cm.haline(0.99), "red"])
contour_show = False
compare = False
matplotlib.rcParams["toolbar"] = "None"


def opencv_window():
    img = np.zeros([100, 100, 3])
    cv2.namedWindow("window", cv2.WINDOW_NORMAL)
    cv2.imshow("window", img)

    while True:
        k = cv2.waitKey(0)
        if k == ord("f"):
            cv2.setWindowProperty(
                "window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
            )
        elif k == ord("n"):
            cv2.setWindowProperty("window", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        elif k == ord("q"):
            cv2.destroyWindow("window")
            break
        else:
            break


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

    poller = zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)
    poller.register(socket3, zmq.POLLIN)

    sockets = {
        "context": context,
        "x_slice": socket1,
        "top_view": socket2,
        "x_slice_c": socket3,
        "poller": poller,
    }
    return sockets


def satellite_window(datasets):
    print("satellite window")
    rot_img_shade = datasets["sat"]
    extent_n = datasets["extent_n"]
    X2 = datasets["X2"]
    Y2 = datasets["Y2"]
    conc = datasets["conc"]

    print(conc[..., -10])
    matplotlib.use("qtagg")

    sockets = make_listen_sockets()

    poller = sockets["poller"]

    socket1 = sockets["x_slice"]
    socket2 = sockets["top_view"]

    init_x = 100

    fig, ax = plt.subplots()
    # Set background color
    fig.patch.set_facecolor("black")
    # No margins
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # No axis
    ax.set_axis_off()
    ax.set_frame_on(False)

    # fullscreen
    manager = plt.get_current_fig_manager()
    try:
        manager.resize(*manager.window.maxsize())
    except AttributeError:
        # no resize available
        pass

    im_sat = ax.imshow(rot_img_shade, extent=extent_n)  # keep window open
    plt.pause(10)
    # interactive
    plt.ion()
    print("image shown")
    plt.show(block=False)
    print("I am not here")

    im_c = ax.contourf(
        X2,
        Y2,
        conc[-10, :, :],
        levels=[0, 1.5, 16],
        vmin=0,
        vmax=15,
        extent=extent_n,
        alpha=0,
        cmap=cmap,
    )
    (line,) = ax.plot(
        [init_x, init_x],
        [extent_n[2], extent_n[3]],
        color="#255070",
        linewidth=3,
        alpha=0.7,
    )

    (line_white,) = ax.plot(
        [init_x, init_x],
        [extent_n[2], extent_n[3]],
        color="white",
        linewidth=1,
    )

    nm, lbl = im_c.legend_elements()
    lbl[0] = "Zoet water"
    lbl[1] = "Zout water"
    legend = ax.legend(nm, lbl, fontsize=8, loc="upper left", framealpha=1)

    for c in im_c.collections:
        c.set_alpha(0)
    for i in range(2):
        legend.get_patches()[i].set(alpha=0)
        legend.get_texts()[i].set(alpha=0)
    legend.draw_frame(False)

    plt.axis("off")
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show(block=False)
    plt.pause(0.1)

    while True:
        socks = dict(poller.poll(10))
        if socket1 in socks and socks[socket1] == zmq.POLLIN:
            # message = json.loads(socket1.recv().decode())
            topic, message = socket1.recv(zmq.DONTWAIT).split()
            # if message[0] == 'x_slice':
            slider_val = int(message)
            line.set_xdata([slider_val, slider_val])
            line_white.set_xdata([slider_val, slider_val])
            # if message[0] == 'top_view':
            plt.pause(0.01)
        if socket2 in socks and socks[socket2] == zmq.POLLIN:
            topic2, message2 = socket2.recv(zmq.DONTWAIT).split()
            alpha = int(message2)
            for c in im_c.collections:
                c.set_alpha(alpha * 0.3)
            for i in range(2):
                legend.get_patches()[i].set(alpha=alpha * 0.3)
                legend.get_texts()[i].set(alpha=alpha * 0.3)
            plt.pause(0.01)
        plt.pause(0.01)


def contour_slice_window(datasets):
    print("Ook al gelukt!")
    matplotlib.use("qtagg")

    sockets = make_listen_sockets()

    poller = sockets["poller"]

    socket3 = sockets["x_slice_c"]

    print("starting matplotlib")

    nbpixels_y = datasets["nbpixels_y"]
    conc_contours_x = datasets["conc_contours_x"]
    conc_contours_x_n = datasets["conc_contours_x_n"]
    conc = datasets["conc"]
    bodem = datasets["bodem0"]
    sat = datasets["sat"]

    diff = np.copy(conc_contours_x_n)
    # diff[(conc_contours_x_n == 0.0) & (conc_contours_x == 0.0)] = 0
    diff[(conc_contours_x_n == 0.0) & (conc_contours_x != 0.0)] = 20
    diff[(conc_contours_x_n != 0.0) & (conc_contours_x == 0.0)] = 20
    diff[(conc_contours_x_n != 0.0) & (conc_contours_x != 0.0)] = 10
    diff[(np.isnan(conc_contours_x_n)) & (np.isnan(conc_contours_x))] = np.nan

    # Define initial parameters (index instead of x value)
    init_x = 100
    extent_x = (0, nbpixels_y, -140, 25.5)
    yb0 = np.linspace(0, sat.shape[0], sat.shape[0])

    # Calculate index for bathymetry dataset (due to varying grid size)
    x_index = init_x
    if x_index % 2 == 0:
        xb_index = int(2.5 * x_index)
    else:
        xb_index = int(np.ceil(2.5 * x_index))

    fig, ax = plt.subplots()

    # adjust the main plot to make room for the slider
    fig.subplots_adjust(left=0.05, bottom=0.25)

    plt.ion()
    print("image shown")
    plt.show(block=False)
    print("I am not here")

    # zorder=0 is default order for images
    ax.fill_between(
        [extent_x[0], extent_x[1]], y1=-140, y2=0, facecolor="#255070", zorder=0
    )

    # pcolormesh is faster, but not as smooth as contourf
    im_x = ax.imshow(
        conc_contours_x[:, :, init_x],
        vmin=0,
        vmax=1.5,
        extent=extent_x,
        aspect="auto",
        cmap=cmap,
    )

    im_x_n = ax.imshow(
        diff[:, :, init_x],
        vmin=0,
        vmax=20,
        extent=extent_x,
        aspect="auto",
        cmap=cmap_n,
        alpha=0,
    )

    plt.pause(10)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im_x, cax=cax)
    cbar.ax.get_yaxis().set_ticks([])
    for i, label in enumerate(["Zoet water", "Zout water"]):
        cbar.ax.text(3.5, (3 + i * 6) / 8, label, ha="center", va="center")
    ax.set_ylim(-100, 25)
    fig.tight_layout()

    plt.axis("off")
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show(block=False)
    plt.pause(0.1)

    while True:
        socks = dict(poller.poll(10))
        if socket3 in socks and socks[socket3] == zmq.POLLIN:
            topic, message = socket3.recv(zmq.DONTWAIT).split()
            slider_val = int(message)
            if slider_val % 2 == 0:
                xb_index = int(2.5 * slider_val)
            else:
                xb_index = int(np.ceil(2.5 * slider_val))
            im_x.set_data(conc_contours_x[:, :, slider_val])
            # im_b.set_ydata(bodem[:, xb_index])
            plt.pause(0.01)
            fig.canvas.draw_idle()
        plt.pause(0.01)
    # print("matplotlib socket", socket)


def slider_window(datasets):
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    conc = datasets["conc"]
    init_x = 100

    fig, ax = plt.subplots()
    plt.axis("off")
    ax.set_axis_off()
    ax.set_frame_on(False)

    x_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    x_slider = Slider(
        ax=ax,
        label="x",
        valmin=0,
        valmax=conc.shape[2] - 1,
        valinit=init_x,
        valstep=1,
    )

    def contour(event):
        global contour_show
        if not contour_show:
            # socket.send(json.dumps(['top_view', 0.3]).encode())
            socket.send_string("top_view %d" % 1)
            contour_show = True
        else:
            # socket.send(json.dumps(['top_view', 0]).encode())
            socket.send_string("top_view %d" % 0)
            contour_show = False

    # The function to be called anytime a slider's value changes
    def update(val):
        socket.send_string("x_slice %d" % val)
        fig.canvas.draw_idle()
        # plt.draw()

    contourax = fig.add_axes([0.6, 0.025, 0.1, 0.04])
    contour_button = Button(contourax, "Contour", hovercolor="0.600")

    x_slider.on_changed(update)
    contour_button.on_clicked(contour)

    plt.show()


def satellite_window2(datasets):
    print("satellite window")
    nbpixels_y = datasets["nbpixels_y"]
    conc_contours_x = datasets["conc_contours_x"]
    conc_contours_x_n = datasets["conc_contours_x_n"]
    conc = datasets["conc"]
    bodem = datasets["bodem"]
    sat = datasets["sat"]

    diff = np.copy(conc_contours_x_n)
    # diff[(conc_contours_x_n == 0.0) & (conc_contours_x == 0.0)] = 0
    diff[(conc_contours_x_n == 0.0) & (conc_contours_x != 0.0)] = 20
    diff[(conc_contours_x_n != 0.0) & (conc_contours_x == 0.0)] = 20
    diff[(conc_contours_x_n != 0.0) & (conc_contours_x != 0.0)] = 10
    diff[(np.isnan(conc_contours_x_n)) & (np.isnan(conc_contours_x))] = np.nan

    # Define initial parameters (index instead of x value)
    init_x = 100
    extent_x = (0, nbpixels_y, -140, 25.5)
    yb0 = np.linspace(0, sat.shape[0], sat.shape[0])

    # Calculate index for bathymetry dataset (due to varying grid size)
    x_index = init_x
    if x_index % 2 == 0:
        xb_index = int(2.5 * x_index)
    else:
        xb_index = int(np.ceil(2.5 * x_index))

    matplotlib.use("qtagg")
    # def key_press(event):
    #    if event.key == 'x':
    #        line.set_ydata()

    print("starting matplotlib")

    # print("matplotlib socket", socket)
    """
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt(zmq.SUBSCRIBE, b'')
    """

    print("constructed subs")

    sockets = make_listen_sockets()

    poller = sockets["poller"]
    socket1 = sockets["x_slice_c"]

    init_x = 100

    fig, ax = plt.subplots()
    # Set background color
    fig.patch.set_facecolor("black")
    # No margins
    fig.tight_layout()
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # No axis
    ax.set_axis_off()
    ax.set_frame_on(False)

    # fullscreen
    manager = plt.get_current_fig_manager()
    try:
        manager.resize(*manager.window.maxsize())
    except AttributeError:
        # no resize available
        pass

    # interactive
    plt.ion()
    print("image shown")
    plt.show(block=False)
    print("I am not here")

    # for i in range(10):
    #     plt.pause(1)
    ax.fill_between(
        [extent_x[0], extent_x[1]], y1=-140, y2=0, facecolor="#255070", zorder=0
    )

    # pcolormesh is faster, but not as smooth as contourf
    im_x = ax.imshow(
        conc_contours_x[:, :, init_x],
        vmin=0,
        vmax=1.5,
        extent=extent_x,
        aspect="auto",
        cmap=cmap,
    )
    plt.pause(10)

    (im_b,) = ax.plot(yb0, bodem[:, xb_index], color="#70543e", linewidth=3)

    plt.axis("off")
    manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.show(block=False)
    plt.pause(0.1)

    while True:
        socks = dict(poller.poll(10))
        if socket1 in socks and socks[socket1] == zmq.POLLIN:
            # message = json.loads(socket1.recv().decode())
            topic, message = socket1.recv(zmq.DONTWAIT).split()
            # if message[0] == 'x_slice':
            slider_val = int(message)
            if slider_val % 2 == 0:
                xb_index = int(2.5 * slider_val)
            else:
                xb_index = int(np.ceil(2.5 * slider_val))
            im_x.set_data(conc_contours_x[:, :, slider_val])
            im_b.set_ydata(bodem[:, xb_index])
            plt.pause(0.001)
        plt.pause(0.001)
