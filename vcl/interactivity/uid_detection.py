import cv2
import time
import numpy as np
import shapely
import geopandas as gpd
import zmq
from vcl.interactivity.camera import Camera
from vcl.interactivity.uid_detector import Detector
from vcl.interactivity.actions import ActionManager

TABLE_POINTS = np.array(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32
)


def draw_detections(frame, detections):
    """Draws bounding boxes and IDs on the frame."""
    for d in detections:
        bbox = d["bbox"]
        # Draw bounding box
        cv2.polylines(frame, [bbox], True, (0, 255, 0), 2)

        # Draw ID/Data text
        # Find top-left corner for text placement
        top_left = bbox[0]
        cv2.putText(
            frame,
            str(d["id"]),
            (top_left[0], top_left[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )
    return frame


def specify_eez(point, gdf: gpd.GeoDataFrame):
    point = shapely.Point(point)
    if gdf.intersects(point).any():
        return gdf[gdf.intersects(point)]["ISO_TER1"].values[0]
    else:
        return


def specifiy_breach_location(point, gdf: gpd.GeoDataFrame):
    point = shapely.Point(point)
    if gdf.intersects(point).any():
        return gdf[gdf.intersects(point)]["location"].values[0]
    else:
        return


def main(socket: zmq.Socket = None, extent=None, datasets={}):
    # Initialize components
    try:
        camera = Camera()
    except ValueError as e:
        print(f"Error: {e}")
        return

    detector = Detector()
    action_manager = ActionManager()

    # Define some sample actions
    def on_T10_000(uid, context):
        location = specifiy_breach_location(
            context["map_coords"], datasets["overstromingen"]
        )
        socket.send_string(f"maps d_T100_000_{location},layer")
        time.sleep(0.1)
        socket.send_string(f"maps animation,layer")
        # print(f"*** HELLO WORLD DETECTED *** at {context['center']}")

    def on_land_info(uid, context):
        eez = specify_eez(context["map_coords"], datasets["eez"])
        socket.send_string(f"uid {eez}_info")

    def on_slice_change(uid, context):
        i = context["center"][0]
        socket.send_string(f"slice {i}")

    def on_land_height(uid, context):
        center = context["center"]
        socket.send_string(f"hands {center[0]},{center[1]}")

    def on_stop(uid, context):
        socket.send_string(f"maps None,layer")

    action_manager.register_action("0", on_T10_000)
    action_manager.register_action("1", on_land_info)
    action_manager.register_action("2", on_slice_change)
    action_manager.register_action("3", on_land_height)

    action_manager.register_lost_action("3", on_stop)

    # Register actions (Example IDs - these would be the content of your QR codes)
    action_manager.register_action("LAND_INFO", on_land_info)
    action_manager.register_action("STOP", on_stop)

    print("Starting Main Loop. Press 'q' to exit.")

    xmin, ymin, xmax, ymax = extent
    MAP_POINTS = np.array(
        [
            (xmin, ymax),
            (xmax, ymax),
            (xmax, ymin),
            (xmin, ymin),
        ]
    )
    H_table_to_map, _ = cv2.findHomography(TABLE_POINTS, MAP_POINTS)

    last_triggered = {}
    last_positions = {"3": (0.5, 0.5)}
    trigger_cooldowns = {"0": np.inf, "2": 0.05, "3": 2}

    # State for persistence tracking: {id: first_observed_time}
    active_tags = {}
    PERSISTENCE_THRESHOLD = 3.0  # seconds

    try:
        while True:
            frame = camera.get_frame()
            if frame is None:
                break

            # Detection
            detections = detector.detect(frame)
            current_time = time.time()
            current_ids = set()

            # Processing and Feedback
            for d in detections:
                # Normalize identifier to string for consistent handling in ActionManager
                identifier = str(d["id"])
                current_ids.add(identifier)

                # Update active_tags start time if new
                # if identifier not in active_tags:
                active_tags[identifier] = current_time
                trigger_cooldown = trigger_cooldowns.get(identifier, 2.0)

                # Check cooldown
                if identifier not in last_triggered or (
                    current_time - last_triggered[identifier] > trigger_cooldown
                ):
                    # Context can include bbox, frame, timestamp, etc.
                    context = {
                        "bbox": d["bbox"],
                        "center": d["center"],
                        "timestamp": current_time,
                    }

                    center = d["center"]

                    # Normalize center
                    center = (
                        center[0] / camera.frame_size[0],
                        center[1] / camera.frame_size[1],
                    )

                    map_coords = cv2.perspectiveTransform(
                        np.array([[[center[0], center[1]]]]), H_table_to_map
                    )[0][0]
                    map_x, map_y = map_coords
                    context["center"] = center
                    context["map_coords"] = (map_x, map_y)

                    if identifier in last_positions:
                        last_pos = last_positions[identifier]
                        distance = np.sqrt(
                            (center[0] - last_pos[0]) ** 2
                            + (center[1] - last_pos[1]) ** 2
                        )
                        if distance < 0.05:
                            action_manager.execute_action(identifier, context)
                        last_positions[identifier] = center
                    else:
                        action_manager.execute_action(identifier, context)
                    last_triggered[identifier] = current_time

            # Check for lost tags
            for identifier in list(active_tags.keys()):
                if identifier not in current_ids:
                    # Tag is no longer detected, check how long it's been missing
                    last_seen = active_tags[identifier]
                    time_since_lost = current_time - last_seen

                    if time_since_lost > PERSISTENCE_THRESHOLD:
                        context = {
                            "duration": time_since_lost,
                            "timestamp": current_time,
                        }
                        action_manager.execute_lost_action(identifier, context)
                        # Remove from last_seen to avoid re-triggering
                        del active_tags[identifier]
                        del last_triggered[identifier]

            # Draw detections
            frame = draw_detections(frame, detections)
            cv2.imshow("UID Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
