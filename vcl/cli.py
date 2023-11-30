"""Console script for vcl."""
import concurrent.futures
import json
import sys
import os
import psutil
import threading
import signal

import click

import matplotlib

import zmq
import time

# import pdb

matplotlib.use("qtagg")


import vcl.display
import vcl.data
import vcl.prep_data


def make_sockets():
    sockets = {}

    context = zmq.Context()
    sockets["context"] = context

    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.connect("tcp://localhost:5556")
    socket.setsockopt(zmq.SUBSCRIBE, b"")
    sockets["SUB"] = socket

    socket = context.socket(zmq.PUB)
    socket.bind("tcp://*:5555")
    sockets["PUB"] = socket
    return sockets


def start_thread_to_terminate_when_parent_process_dies(ppid):
    pid = os.getpid()

    # def f():
    #     while True:
    #         try:
    #             os.kill(ppid, 0)
    #             # for proc in psutil.process_iter():
    #             #     if proc.pid == ppid:
    #             #         proc.terminate()
    #             #p = psutil.Process(ppid)
    #             #p.terminate()

    #         except:
    #             os.kill(pid, signal.SIGTERM)
    #             # for proc in psutil.process_iter():
    #             #     if proc.pid == pid:
    #             #         proc.terminate()
    #         time.sleep(1)

    # thread = threading.Thread(target=f, daemon=True)
    thread = threading.Thread(daemon=True)
    thread.start()


def test(datasets):
    print("data loaded")
    for key, val in datasets.items():
        print(key, type(val))
    return "ok"


@click.command()
@click.option("--satellite/--no-satellite", default=False)
@click.option("--contour/--no-contour", default=False)
@click.option("-s", "--size")
def main(satellite, contour, size, args=None):
    """Console script for vcl."""

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=10,
        initializer=start_thread_to_terminate_when_parent_process_dies,
        initargs=(os.getpid(),),
    )

    datasets = vcl.load_data.load()
    datasets = vcl.prep_data.preprocess(datasets)
    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     task = executor.submit(test, datasets)
    import ipdb

    ipdb.set_trace()
    if satellite:
        executor.submit(vcl.display.satellite_window, datasets[size])
    # # executor.submit(mayavi_window)
    # # executor.submit(vcl.display.opencv_window)
    executor.submit(vcl.display.slider_window, datasets[size])
    if contour:
        # executor.submit(vcl.display.satellite_window2, datasets)
        executor.submit(vcl.display.contour_slice_window, datasets[size])

    # while True:
    #     time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover


# """Console script for vcl."""
# import concurrent.futures
# import json
# import sys
# import os
# import psutil
# import threading
# import signal

# import click

# import matplotlib

# import zmq
# import time
# # import pdb

# matplotlib.use("qtagg")


# import vcl.display
# import vcl.data
# import vcl.prep_data


# def make_sockets():
#     sockets = {}

#     context = zmq.Context()
#     sockets["context"] = context

#     socket = context.socket(zmq.SUB)
#     socket.setsockopt(zmq.CONFLATE, 1)
#     socket.connect("tcp://localhost:5556")
#     socket.setsockopt(zmq.SUBSCRIBE, b"")
#     sockets["SUB"] = socket

#     socket = context.socket(zmq.PUB)
#     socket.bind("tcp://*:5555")
#     sockets["PUB"] = socket
#     return sockets


# def start_thread_to_terminate_when_parent_process_dies(ppid):
#     pid = os.getpid()

#     def f():
#         while True:
#             try:
#                 os.kill(ppid, 0)
#                 # for proc in psutil.process_iter():
#                 #     if proc.pid == ppid:
#                 #         proc.terminate()
#                 #p = psutil.Process(ppid)
#                 #p.terminate()

#             except:
#                 os.kill(pid, signal.SIGTERM)
#                 # for proc in psutil.process_iter():
#                 #     if proc.pid == pid:
#                 #         proc.terminate()
#             time.sleep(1)

#     thread = threading.Thread(target=f, daemon=True)
#     thread.start()


# def test(datasets):
#     print("data loaded")
#     for key, val in datasets.items():
#         print(key, type(val))
#     return "ok"


# @click.command()
# @click.option("--satellite/--no-satellite", default=False)
# @click.option("--contour/--no-contour", default=False)
# @click.option("-s", "--size")
# def main(satellite, contour, size, args=None):
#     """Console script for vcl."""


#     datasets = vcl.load_data.load()
#     datasets = vcl.prep_data.preprocess(datasets)
#     print("gaat nog prima 6")

#     if satellite:
#         # vcl.display.satellite_window(datasets[size])
#         vcl.display.contour_slice_window(datasets[size])
#         print("gaat nog prima 7")
#     return 0


# if __name__ == "__main__":
#     sys.exit(main())  # pragma: no cover
