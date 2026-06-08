"""Camera calibration for the Virtual Climate Lab projector–camera setup.

Run ``vcl --calibrate`` to open a live webcam view.  Click the four corners
of the table surface (top-left \u2192 top-right \u2192 bottom-right \u2192 bottom-left) and
press **c** to save the normalised coordinates.  Press **r** to reset the
clicked points or **q** / **ESC** to abort without saving.

The saved coordinates are used by :mod:`vcl.interactivity.camera` to compute
a homography that maps camera-space pixels to table-space coordinates.
"""

import cv2
import numpy as np
import time

from vcl.interactivity.camera import Camera
from vcl.interactivity.uid_detector import Detector
from vcl.config import save_camera_points


clicked_points = []


def mouse_callback(event, x, y, flags, param):
    global clicked_points
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_points) < 4:
            clicked_points.append((x, y))
            print(f"Clicked point {len(clicked_points)}: {(x, y)}")


def run_calibration(source=0):
    """
    Run the webcam calibration script.
    Looks for AprilTags 0, 1, 2, and 3.
    Renders live feed.
    Press 'c' to capture and save the camera points and exit.
    """
    print("Starting Calibration...")
    print("Please arrange exactly 4 AprilTags (IDs 0, 1, 2, 3) in the corners:")
    print("  Tag 0: Top-Left")
    print("  Tag 1: Top-Right")
    print("  Tag 2: Bottom-Right")
    print("  Tag 3: Bottom-Left")
    print("Press 'c' to capture the points and store them.")
    print("Press 'q' or 'ESC' to abort without saving.")
    global clicked_points

    try:
        camera = Camera(source=source)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Disable the homography transform in Camera class manually for calibration,
    # because we need the raw camera feed points.
    detector = Detector()

    try:
        cv2.namedWindow("Calibration")
        cv2.setMouseCallback("Calibration", mouse_callback)

        while True:
            ret, frame = camera.cap.read()
            if not ret:
                break

            h, w, _ = frame.shape

            # Draw clicked points
            for p in clicked_points:
                cv2.circle(frame, p, 6, (0, 0, 255), -1)

            # Draw polygon in the clicked order
            if len(clicked_points) == 4:
                poly = np.array(clicked_points, dtype=np.int32)
                cv2.polylines(frame, [poly], True, (255, 255, 0), 2)
                cv2.putText(
                    frame,
                    "Press 'c' to save",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )

            cv2.putText(
                frame,
                "Click 4 corners (TL, TR, BR, BL)",
                (50, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            cv2.imshow("Calibration", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                print("Aborted.")
                break

            if key == ord("r"):
                clicked_points = []
                print("Reset points.")

            if key == ord("c") and len(clicked_points) == 4:
                normalized = [(x / w, y / h) for (x, y) in clicked_points]
                save_camera_points(normalized)
                print("Saved:", normalized)
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
