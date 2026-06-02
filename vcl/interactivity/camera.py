import cv2
import numpy as np

from vcl.config import load_camera_points

CAMERA_POINTS = load_camera_points()

TABLE_POINTS = np.array(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32
)

M = cv2.getPerspectiveTransform(CAMERA_POINTS, TABLE_POINTS)


class Camera:
    def __init__(self, source=0, frame_size=(1920, 1080)):
        self.cap = cv2.VideoCapture(source)
        self.frame_size = frame_size
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video source {source}")

    def get_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return None
        # Preprocess frame
        frame = cv2.resize(frame, self.frame_size)
        h, w, _ = frame.shape

        src_points = (CAMERA_POINTS * [w, h]).astype(np.float32)
        dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_points, dst_points)
        # M, _ = cv2.findHomography(CAMERA_POINTS, dst_points)
        frame = cv2.warpPerspective(frame, M, (w, h))

        return frame

    def release(self):
        self.cap.release()
