"""Command-line interface for Virtual Climate Lab (VCL).

This module provides the CLI entry point for launching the VCL visualization system.
It uses Click for command-line argument parsing and manages multiple display processes
via concurrent futures. The CLI allows selection of different components including
the main map display, statistics panels, MIDI/keyboard control, hand tracking, and
UID detection.

The system uses ZeroMQ for inter-process communication between display windows and
input handlers. All processes are launched in a process pool and communicate via
publish-subscribe sockets on localhost ports 5555-5558.

Usage:
    # Launch with MIDI control and main map display
    vcl --midi --satellite

    # Full system with all features
    vcl --midi --satellite --stats --hand_tracking --uid

    # Preprocess data and save
    vcl --preprocess --save

Environment:
    Requires input.json in the vcl package directory with configuration including
    basepath for data files.
"""

import concurrent.futures
import json
import logging
import os
import sys
import threading
from pathlib import Path

import click
import numpy as np
import zmq

import vcl.display_pygame
import vcl.interactivity.calibration
import vcl.preprocess
import vcl.serialize


def make_sockets():
    """Create ZeroMQ sockets for inter-process communication.

    Initializes subscriber and publisher sockets for receiving and sending messages
    between CLI and display processes. Currently not actively used in the main CLI
    flow but available for potential extensions.

    Returns:
        dict: Dictionary containing ZMQ context and sockets:
            - context: ZMQ context object
            - SUB: Subscriber socket connected to port 5556
            - PUB: Publisher socket bound to port 5555

    Note:
        SUB socket uses CONFLATE to only keep the latest message, preventing
        message queue buildup.
    """
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
    """Initialize worker process to terminate when parent dies.

    This function is used as an initializer for worker processes in the process pool.
    It starts a daemon thread that would monitor the parent process and terminate
    the worker if the parent dies unexpectedly.

    Args:
        ppid: Parent process ID to monitor.

    Note:
        Current implementation is a placeholder with monitoring logic commented out.
        The daemon thread is started but doesn't actively monitor the parent.
        Uncomment the monitoring code if child process cleanup is needed.
    """
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
@click.option("--uid/--no-uid", default=False)
@click.option("--preprocess/--no-preprocess", default=False)
@click.option(
    "--calibrate",
    is_flag=True,
    help="Run camera calibration using 4 AprilTags and exit.",
)
def main(
    satellite,
    contour,
    stats,
    midi,
    hand_tracking,
    uid,
    preprocess,
    calibrate,
    args=None,
):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    """Main CLI entry point for Virtual Climate Lab.

    Launches selected VCL components in separate processes. Each component runs
    independently and communicates via ZeroMQ sockets. The process pool manages
    up to 10 worker processes.

    Args:
        satellite: If True, launch the main satellite/map display window.
        contour: If True, launch the cross-section/slice viewer window.
        stats: If True, launch the statistics and information panel window.
        midi: If True, use MIDI controller for input (default). If False, use keyboard.
        hand_tracking: If True, enable webcam-based hand tracking for gesture control.
        uid: If True, enable UID detection (AprilTag/QR/ArUco) for interactive elements.
        preprocess: If True, run data preprocessing from input.json configuration.
        save: If True with --preprocess, save preprocessed data to disk.
        args: Additional arguments (not used).

    Returns:
        int: Exit code (always 0).

    Raises:
        FileNotFoundError: If input.json is not found in the package directory.

    Note:
        The function reads input.json for configuration and preprocessed-data.npy
        for cached data. Data preprocessing can be time-consuming, so use --save
        to cache results.

    Examples:
        Launch full system:
            $ vcl --midi --satellite --stats --hand_tracking --uid

        Preprocess and save data:
            $ vcl --preprocess --save

        Launch minimal system with keyboard:
            $ vcl --no-midi --satellite
    """
    with open(Path(__file__).parent / "input.json") as f:
        input_dict = json.load(f)

    if calibrate:
        vcl.interactivity.calibration.run_calibration()
        return 0

    data_dir = input_dict.get("basepath")
    data_dir = Path(data_dir)

    executor = concurrent.futures.ProcessPoolExecutor(
        max_workers=10,
        initializer=start_thread_to_terminate_when_parent_process_dies,
        initargs=(os.getpid(),),
    )
    if preprocess:
        input_file = Path(__file__).parent / "input.json"
        datasets = vcl.preprocess.preprocess(input_file=input_file)
        save_path = data_dir / "preprocessed-data"
        vcl.serialize.save(datasets, save_path)

    datasets = data_dir / "preprocessed-data"

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
    if uid:
        executor.submit(vcl.display_pygame.uid_detector, datasets)

    # while True:
    #     time.sleep(0.1)
    return 0


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
