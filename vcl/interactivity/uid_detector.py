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


class Detector:
    def __init__(self):
        # Initialize QR Code Detector
        self.qr_detector = cv2.QRCodeDetector()

        # Initialize ArUco Detector for AprilTags (using 4x4 dictionary)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(
            self.aruco_dict, self.aruco_params
        )

    def detect(self, frame):
        """
        Detects QR codes and AprilTags in the frame.
        Returns a list of dictionaries: {'type': 'qr'/'tag', 'id': data/id, 'bbox': points}
        """
        detections = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 1. Detect QR Codes
        # detectAndDecodeMulti returns (retval, decoded_info, points, straight_qrcode)
        retval, decoded_info, points, _ = self.qr_detector.detectAndDecodeMulti(gray)
        if retval:
            for data, bbox in zip(decoded_info, points):
                if data:  # Only add if data is successfully decoded
                    bbox = bbox.astype(int)
                    # Calculate center
                    center_x = int(np.mean(bbox[:, 0]))
                    center_y = int(np.mean(bbox[:, 1]))

                    detections.append(
                        {
                            "type": "qr",
                            "id": data,
                            "bbox": bbox,
                            "center": (center_x, center_y),
                        }
                    )

        # 2. Detect AprilTags
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        if ids is not None:
            for i, marker_id in enumerate(ids.flatten()):
                # corners[i] is [1, 4, 2], need to reshape to [4, 2] for consistency
                bbox = corners[i].reshape(4, 2).astype(int)

                # Calculate center
                center_x = int(np.mean(bbox[:, 0]))
                center_y = int(np.mean(bbox[:, 1]))

                detections.append(
                    {
                        "type": "tag",
                        "id": int(marker_id),
                        "bbox": bbox,
                        "center": (center_x, center_y),
                    }
                )

        return detections
