import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import zmq


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


def satellite_window():
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

    context = zmq.Context()
    socket1 = context.socket(zmq.SUB)
    socket1.connect("tcp://localhost:5556")
    socket1.subscribe("x_slice")

    socket2 = context.socket(zmq.SUB)
    socket2.connect("tcp://localhost:5556")
    socket2.subscribe("top_view")

    poller = zmq.Poller()
    poller.register(socket1, zmq.POLLIN)
    poller.register(socket2, zmq.POLLIN)

    print('constructed subs')

    init_x = 100
    matplotlib.use("qtagg")
    fig, ax = plt.subplots()
    return
    im_sat = ax.imshow(rot_img_shade, extent=extent_n)  # keep window open
    im_c = ax.contourf(
        X2,
        Y2,
        rot_ds[-10, :, :],
        levels=[0, 1.5, 16],
        vmin=0,
        vmax=15,
        extent=extent_n,
        alpha=0,
        cmap=cmap,
    )
    (line,) = ax.plot(
        [2.5 * init_x, 2.5 * init_x],
        [extent_n[2], extent_n[3]],
        color="blue",
        linewidth=2,
    )

    nm, lbl = im_c.legend_elements()
    lbl[0] = "Zoet water"
    lbl[1] = "Zout water"
    legend = ax.legend(nm, lbl, fontsize=8, loc="upper left", framealpha=1)

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
            line.set_xdata([2.5 * slider_val, 2.5 * slider_val])
            # if message[0] == 'top_view':
            plt.pause(0.04)

        if socket2 in socks and socks[socket2] == zmq.POLLIN:
            # message = json.loads(socket2.recv().decode())
            topic, message2 = socket2.recv(zmq.DONTWAIT).split()
            alpha = float(message2)
            # alpha = 0
            if alpha == 0:
                for c in im_c.collections:
                    c.set_alpha(0)
                for i in range(2):
                    legend.get_patches()[i].set(alpha=0)
                    legend.get_texts()[i].set(alpha=0)
                legend.draw_frame(False)
            else:
                for c in im_c.collections:
                    c.set_alpha(0.3)
                for i in range(2):
                    legend.get_patches()[i].set_alpha(0.3)
                    legend.get_texts()[i].set_alpha(1)
                legend.draw_frame(True)
            plt.pause(0.04)


def contour_slice_window():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    # def key_press(event):
    #    if event.key == 'x':
    #        line.set_ydata()

    print("starting matplotlib")

    # Define initial parameters (index instead of x value)
    init_x = 100
    extent_x = (0, nbpixels_y, -140, 25.5)

    matplotlib.use("qtagg")
    fig, axes = plt.subplots()

    # adjust the main plot to make room for the slider
    fig.subplots_adjust(left=0.05, bottom=0.25)

    # pcolormesh is faster, but not as smooth as contourf
    im_x = axes.imshow(
        conc_contours_x[:, :, init_x],
        vmin=0,
        vmax=1.5,
        extent=extent_x,
        aspect="auto",
        cmap=cmap,
    )

    # Make a horizontal slider to control the position on the x-axis.
    x_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    x_slider = Slider(
        ax=x_ax,
        label="x",
        valmin=0,
        valmax=rot_ds.shape[2] - 1,
        valinit=init_x,
        valstep=1,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        # socket.send(json.dumps(['x_slice', val]).encode())
        socket.send_string("x_slice %d" % val)
        im_x.set_data(conc_contours_x[:, :, val])
        fig.canvas.draw_idle()
        # plt.draw()

    # Change the transparency for each individual element, especially the legend had to be made transparent partswise
    # Contour plot does not have Artist, hence call Artist of im_c.collections
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

    contourax = fig.add_axes([0.6, 0.025, 0.1, 0.04])
    contour_button = Button(contourax, "Contour", hovercolor="0.900")

    # register the update function with each slider
    x_slider.on_changed(update)
    contour_button.on_clicked(contour)

    # register the update function with each slider
    x_slider.on_changed(update)

    divider = make_axes_locatable(axes)
    cax = divider.append_axes("right", size="5%", pad=0.10)
    cbar = plt.colorbar(im_x, cax=cax)
    cbar.ax.get_yaxis().set_ticks([])
    for i, label in enumerate(["Zoet water", "Zout water"]):
        cbar.ax.text(3.5, (3 + i * 6) / 8, label, ha="center", va="center")

    plt.show()
    # print("matplotlib socket", socket)


def slider_window():
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5556")

    fig, axes = plt.subplots()

    x_ax = fig.add_axes([0.25, 0.1, 0.5, 0.03])
    x_slider = Slider(
        ax=x_ax,
        label="x",
        valmin=0,
        valmax=6,
        valinit=1,
        valstep=1,
    )

    # The function to be called anytime a slider's value changes
    def update(val):
        socket.send(str(val).encode())
        fig.canvas.draw_idle()
        # plt.draw()

    plt.show()