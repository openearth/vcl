"""Console script for vcl."""

import concurrent.futures
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import click
import matplotlib
import numpy as np
import psutil
import zmq

import vcl.data
import vcl.display
import vcl.load_data
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
    # pid = os.getpid()

    # def f():
    #     while True:
    #         try:
    #             os.kill(ppid, 0)
    #             # for proc in psutil.process_iter():
    #             #     if proc.pid == ppid:
    #             #         proc.terminate()
    #             # p = psutil.Process(ppid)
    #             # p.terminate()

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
@click.option("--midi/--no-midi", default=True)
@click.option("--preprocess/--no-preprocess", default=False)
@click.option("--save/--no-save", default=False)
def main(satellite, contour, midi, preprocess, save, args=None):
    """Console script for vcl."""

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=10,
        initializer=start_thread_to_terminate_when_parent_process_dies,
        initargs=(os.getpid(),),
    )
    if preprocess:
        common_datasets, unique_datasets = vcl.load_data.load()
        datasets = vcl.prep_data.preprocess(common_datasets, unique_datasets)
        if save:
            data_dir = Path("~/data/vcl/dataset").expanduser()
            np.save(data_dir / "preprocessed-data.npy", datasets)
    else:
        datasets = vcl.load_data.load_preprocessed()

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     task = executor.submit(test, datasets)

    if midi:
        executor.submit(vcl.display.midi_board, datasets)
    else:
        executor.submit(vcl.display.slider_window, datasets)
    if satellite:
        executor.submit(vcl.display.satellite_window, datasets)
    if contour:
        # executor.submit(vcl.display.satellite_window2, datasets)
        executor.submit(vcl.display.contour_slice_window, datasets)

    # while True:
    #     time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
