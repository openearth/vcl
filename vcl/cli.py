"""Console script for vcl."""
import concurrent.futures
import json
import sys
import time


import click

import matplotlib

import zmq



matplotlib.use("qtagg")


import vcl.display
import vcl.data


def make_sockets():
    sockets = {}

    context = zmq.Context()

    socket = context.socket(zmq.SUB)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    sockets["SUB"] = socket
    sockets["context"] = context

    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    sockets["PUB"] = socket
    return sockets


@click.command()
@click.option("--satellite/--no-satellite", default=False)
@click.option("--contour/--no-contour", default=False)
def main(satellite, contour, args=None):
    """Console script for vcl."""

    sockets = make_sockets()

    executor = concurrent.futures.ProcessPoolExecutor(max_workers=10)

    if satellite:
        executor.submit(vcl.display.satellite_window, rot_img_shade, extent_n)
    # executor.submit(mayavi_window)
    # executor.submit(vcl.display.opencv_window)
    # executor.submit(slider_window)
    if contour:
        executor.submit(vcl.display.contour_slice_window)

    while True:
        time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
