import cv2
import numpy as np


class UIDDetector:
    def __init__(self):
        self.qr_detector = cv2.QRCodeDetector()

    def detect(self, frame):
        """
        Detects QR codes in the frame.
        Returns a list of dictionaries with 'id', 'bbox', and 'center'.
        """
        # detectAndDecodeMulti finds multiple QR codes
        retval, decoded_info, points, straight_qrcode = (
            self.qr_detector.detectAndDecodeMulti(frame)
        )

        detections = []
        if retval:
            for i, data in enumerate(decoded_info):
                if not data:
                    continue  # Skip empty detections

                # points[i] is a set of 4 corners
                qr_points = points[i]

                # Calculate center
                # qr_points is shape (4, 2)
                center_x = np.mean(qr_points[:, 0])
                center_y = np.mean(qr_points[:, 1])

                detections.append(
                    {
                        "id": data,
                        "bbox": qr_points.astype(int),
                        "center": (int(center_x), int(center_y)),
                    }
                )

        return detections
