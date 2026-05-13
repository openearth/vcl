import cv2
import numpy as np
import time

from vcl.interactivity.camera import Camera
from vcl.interactivity.uid_detector import Detector
from vcl.config import save_camera_points


def order_corners(pts):
    pts = np.array(pts, dtype=np.float32)

    # Compute centroid
    cx = np.mean(pts[:, 0])
    cy = np.mean(pts[:, 1])

    # Compute angle of each point relative to centroid
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)

    # Sort points by angle (counter‑clockwise)
    sorted_idx = np.argsort(angles)
    pts = pts[sorted_idx]

    # After sorting CCW, identify top-left as the point with smallest (x+y)
    s = pts.sum(axis=1)
    tl_idx = np.argmin(s)

    # Rotate array so that TL is first
    ordered = np.roll(pts, -tl_idx, axis=0)

    # Now order is TL, TR, BR, BL
    return ordered.tolist()


def get_calibration_corner(tag_id, bbox):
    """
    Tag 0 -> top-left corner
    Tag 1 -> top-right corner
    Tag 2 -> bottom-right corner
    Tag 3 -> bottom-left corner
    In OpenCV ArUco returned corners: 0=TL, 1=TR, 2=BR, 3=BL relative to marker orientation.
    """
    if tag_id == 0:
        return tuple(bbox[0])
    elif tag_id == 1:
        return tuple(bbox[1])
    elif tag_id == 2:
        return tuple(bbox[2])
    elif tag_id == 3:
        return tuple(bbox[3])
    return None


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
