"""Console script for vcl."""

import concurrent.futures
import json
import os
import pickle
import signal
import sys
import threading
import time
from pathlib import Path

import click
import numpy as np
import zmq

import vcl.data
import vcl.display_pygame

# import vcl.display
import vcl.load_data
import vcl.preprocess


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
@click.option("--stats/--no-stats", default=False)
@click.option("--midi/--no-midi", default=True)
@click.option("--hand_tracking/--no-hand_tracking", default=False)
@click.option("--preprocess/--no-preprocess", default=False)
@click.option("--save/--no-save", default=False)
def main(satellite, contour, stats, midi, hand_tracking, preprocess, save, args=None):
    """Console script for vcl."""

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=10,
        initializer=start_thread_to_terminate_when_parent_process_dies,
        initargs=(os.getpid(),),
    )
    if preprocess:
        input_file = Path(__file__).parent / "input.json"
        datasets = vcl.preprocess.preprocess(input_file=input_file)
        if save:
            data_dir = Path("~/data/vcl/gnsbi").expanduser()
            # np.save(
            #     data_dir / "preprocessed-data.npy",
            #     datasets,
            #     allow_pickle=True,
            # )
            with open(data_dir / "preprocessed-data.npy", "wb") as f:
                pickle.dump(datasets, f, protocol=4)
    # else:
    #     datasets = vcl.load_data.load_preprocessed()

    datasets = data_dir / "preprocessed-data.npy"

    # with concurrent.futures.ProcessPoolExecutor() as executor:
    #     task = executor.submit(test, datasets)

    if midi:
        executor.submit(vcl.display_pygame.midi_board, datasets)
    else:
        executor.submit(vcl.display_pygame.keyboard_publisher)
    if satellite:
        executor.submit(vcl.display_pygame.displaymap, datasets)
    if contour:
        # executor.submit(vcl.display.satellite_window2, datasets)
        executor.submit(vcl.display_pygame.displayslice, datasets)
    if stats:
        executor.submit(vcl.display_pygame.displaystats, datasets)
    if hand_tracking:
        executor.submit(vcl.display_pygame.hand_tracker, datasets)

    # while True:
    #     time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
