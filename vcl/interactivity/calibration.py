import cv2
import numpy as np
import time

from vcl.interactivity.camera import Camera
from vcl.interactivity.uid_detector import Detector
from vcl.config import save_camera_points


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

    try:
        camera = Camera(source=source)
    except ValueError as e:
        print(f"Error: {e}")
        return

    # Disable the homography transform in Camera class manually for calibration,
    # because we need the raw camera feed points.
    detector = Detector()

    try:
        while True:
            ret, frame = camera.cap.read()
            if not ret:
                break

            # frame = cv2.flip(frame, 1)
            # frame = cv2.resize(frame, camera.frame_size)
            h, w, _ = frame.shape

            detections = detector.detect(frame)
            calibration_points = {}

            for d in detections:
                tag_id = d["id"]
                bbox = d["bbox"]

                corner = get_calibration_corner(tag_id, bbox)
                if corner:
                    calibration_points[tag_id] = corner

                cv2.polylines(frame, [bbox], True, (0, 255, 0), 2)

                if corner:
                    cv2.circle(frame, corner, 5, (0, 0, 255), -1)
                    cv2.putText(
                        frame,
                        f"ID {tag_id} (Corner)",
                        (corner[0] - 10, corner[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                else:
                    cv2.putText(
                        frame,
                        f"Ignored ID {tag_id}",
                        (bbox[0][0], bbox[0][1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (128, 128, 128),
                        2,
                    )

            required_ids = {0, 1, 2, 3}
            detected_required = set(calibration_points.keys())

            if required_ids.issubset(detected_required):
                sorted_pixels = [
                    calibration_points[0],
                    calibration_points[1],
                    calibration_points[2],
                    calibration_points[3],
                ]
                cv2.polylines(
                    frame,
                    [np.array(sorted_pixels, dtype=np.int32)],
                    True,
                    (255, 255, 0),
                    2,
                )
                cv2.putText(
                    frame,
                    "4 TAGS DETECTED - PRESS 'c' to CAPTURE",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 255),
                    2,
                )
            else:
                missing = required_ids - detected_required
                cv2.putText(
                    frame,
                    f"Missing Tags: {missing}",
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2,
                )

            cv2.imshow("Calibration - Press 'c' to capture, 'q' to abort", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q") or key == 27:  # ESC
                print("Aborted calibration.")
                break
            elif key == ord("c"):
                if required_ids.issubset(detected_required):
                    sorted_pixels = [
                        calibration_points[0],
                        calibration_points[1],
                        calibration_points[2],
                        calibration_points[3],
                    ]
                    # Normalize points (0-1) across the resized frame dim
                    normalized_points = [
                        (float(x) / w, float(y) / h) for (x, y) in sorted_pixels
                    ]

                    save_camera_points(normalized_points)
                    print(f"Captured points: {normalized_points}")
                    print("Calibration completed successfully!")

                    cv2.putText(
                        frame,
                        "SAVED!",
                        (w // 2 - 100, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        2,
                        (0, 255, 0),
                        4,
                    )
                    cv2.imshow(
                        "Calibration - Press 'c' to capture, 'q' to abort", frame
                    )
                    cv2.waitKey(1500)
                    break
                else:
                    missing = required_ids - detected_required
                    print(f"Need tags 0, 1, 2, and 3. Missing: {missing}")

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_calibration()
