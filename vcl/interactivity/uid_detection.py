import cv2
import time
import numpy as np
import shapely
import geopandas as gpd
import zmq
from vcl.interactivity.camera import Camera
from vcl.interactivity.uid_detector import UIDDetector
from vcl.interactivity.actions import ActionManager

TABLE_POINTS = np.array(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32
)


def specify_eez(point, gdf: gpd.GeoDataFrame):
    point = shapely.Point(point)
    if gdf.intersects(point).any():
        return gdf[gdf.intersects(point)]["ISO_TER1"].values[0]
    else:
        return


def main(socket: zmq.Socket = None, extent=None, datasets={}):
    # Initialize components
    try:
        camera = Camera()
    except ValueError as e:
        print(f"Error: {e}")
        return

    detector = UIDDetector()
    action_manager = ActionManager()

    # Define some sample actions
    def on_land_detection(uid, context):
        eez = specify_eez(context["map_coords"], datasets["eez"])
        socket.send_string(f"uid {eez}_detection")
        # print(f"*** HELLO WORLD DETECTED *** at {context['center']}")

    def on_land_info(uid, context):
        eez = specify_eez(context["map_coords"], datasets["eez"])
        socket.send_string(f"uid {eez}_info")

    def on_stop(uid, context):
        print("!!! STOP DETECTED !!!")

    # Register actions (Example IDs - these would be the content of your QR codes)
    action_manager.register_action("LAND_DETECTION", on_land_detection)
    action_manager.register_action("LAND_INFO", on_land_info)
    action_manager.register_action("STOP", on_stop)

    print("Starting Main Loop. Press 'q' to exit.")

    xmin, ymin, xmax, ymax = extent
    MAP_POINTS = np.array(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]
    )
    H_table_to_map, _ = cv2.findHomography(TABLE_POINTS, MAP_POINTS)

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            # Detection
            detections = detector.detect(frame)

            # Processing and Feedback
            for d in detections:
                uid = d["id"]
                bbox = d["bbox"]
                center = d["center"]

                # Draw bounding box
                # bbox is np array of shape (4, 2)
                for i in range(4):
                    cv2.line(
                        frame, tuple(bbox[i]), tuple(bbox[(i + 1) % 4]), (0, 255, 0), 2
                    )

                # Draw center
                cv2.circle(frame, center, 5, (0, 0, 255), -1)

                # Draw ID text
                cv2.putText(
                    frame,
                    uid,
                    (bbox[0][0], bbox[0][1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
                # Normalize center
                center = (
                    center[0] / camera.frame_size[0],
                    center[1] / camera.frame_size[1],
                )

                map_coords = cv2.perspectiveTransform(
                    np.array([[[center[0], center[1]]]]), H_table_to_map
                )[0][0]
                map_x, map_y = map_coords

                # Trigger Action
                context = {"center": center, "bbox": bbox, "map_coords": (map_x, map_y)}
                action_manager.execute_action(uid, context)

            cv2.imshow("UID Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
