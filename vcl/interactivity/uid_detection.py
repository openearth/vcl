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


def specify_slider_position(point, gdf: gpd.GeoDataFrame):
    point = shapely.Point(point)
    if gdf.buffer(100).intersects(point).any():
        xmin, ymin, xmax, ymax = gdf.total_bounds
        slider_position = (point.x - xmin) / (xmax - xmin)
        slider_position = np.clip(slider_position, a_min=0, a_max=1)
        slice_index = int(slider_position * 338)
        return slice_index
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

    uid_to_event = {
        "0": "1000",
        "1": "housing",
        "2": "innovation_hub",
        "3": "energy_park",
        "4": "ground_cover",
        "5": "treat_water",
        "6": "destroy_plants",
    }

    def on_scenario_event(uid, context):
        event = uid_to_event[uid]
        socket.send_string(f"maps {event},scenario")

    def on_measure_event(uid, context):
        event = uid_to_event[uid]
        socket.send_string(f"maps {event},add_measure")

    def on_slider_change(uid, context):
        try:
            slice_index = specify_slider_position(
                context["map_coords"], datasets["slider"]
            )
            if slice_index is not None:
                socket.send_string(f"slice {slice_index}")
        except Exception as e:
            print(e)

    def on_stop_scenario(uid, context):
        socket.send_string(f"maps none,scenario")

    def on_stop_measure(uid, context):
        event = uid_to_event[uid]
        socket.send_string(f"maps {event},remove_measure")

    action_manager.register_action("0", on_slider_change)
    action_manager.register_action("1", on_scenario_event)
    action_manager.register_action("2", on_scenario_event)
    action_manager.register_action("3", on_scenario_event)
    action_manager.register_action("4", on_measure_event)
    action_manager.register_action("5", on_measure_event)
    action_manager.register_action("6", on_measure_event)

    # action_manager.register_lost_action("0", on_stop)
    # action_manager.register_lost_action("1", on_stop_scenario)
    # action_manager.register_lost_action("2", on_stop_scenario)
    # action_manager.register_lost_action("3", on_stop_scenario)
    action_manager.register_lost_action("4", on_stop_measure)
    action_manager.register_lost_action("5", on_stop_measure)
    action_manager.register_lost_action("6", on_stop_measure)

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
    last_positions = {"0": (0, 0), "1": (0, 0), "2": (0, 0), "3": (0.5, 0.5)}
    trigger_cooldowns = {"0": 0.05, "1": np.inf, "2": np.inf, "3": np.inf}
    # If a tag moves more than this normalised distance, reset its trigger so
    # the action fires again at the new location (relevant for infinite-cooldown
    # tags like "0" that would otherwise never re-trigger).
    movement_retrigger_thresholds = {"0": 0.05, "1": 0.05, "2": 0.05, "3": 0.05}

    # State for persistence tracking: {id: first_observed_time}
    active_tags = {}
    PERSISTENCE_THRESHOLD = 1.5  # seconds
    SCENARIO_TAGS = {"1", "2", "3"}
    scenario_last_seen = None
    scenario_stopped = True  # assume stopped until we see one

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

                center = d["center"]

                # Normalize center to [0, 1] range
                center = (
                    center[0] / camera.frame_size[0],
                    center[1] / camera.frame_size[1],
                )

                # Update active_tags timestamp
                active_tags[identifier] = current_time
                trigger_cooldown = trigger_cooldowns.get(identifier, 2.0)
                retrigger_dist = movement_retrigger_thresholds.get(identifier, 1)

                last_pos = last_positions.get(identifier)
                distance = (
                    np.sqrt(
                        (center[0] - last_pos[0]) ** 2 + (center[1] - last_pos[1]) ** 2
                    )
                    if last_pos is not None
                    else None
                )

                cooldown_expired = identifier not in last_triggered or (
                    current_time - last_triggered[identifier] > trigger_cooldown
                )
                # Re-trigger when the tag moves significantly (e.g. placed at a new spot)
                moved_to_new_location = (
                    distance is not None and distance > retrigger_dist
                )

                if cooldown_expired or moved_to_new_location:
                    map_coords = cv2.perspectiveTransform(
                        np.array([[[center[0], center[1]]]]), H_table_to_map
                    )[0][0]
                    map_x, map_y = map_coords
                    context = {
                        "bbox": d["bbox"],
                        "center": center,
                        "map_coords": (map_x, map_y),
                        "timestamp": current_time,
                    }
                    action_manager.execute_action(identifier, context)
                    last_triggered[identifier] = current_time

                # Always keep last_positions up to date so the next frame's
                # distance calculation is accurate
                last_positions[identifier] = center

            # --- Scenario group stop logic (1/2/3) ---
            scenario_present = any(t in current_ids for t in SCENARIO_TAGS)

            if scenario_present:
                scenario_last_seen = current_time
                # If we were stopped, we are now "active" again
                scenario_stopped = False
            else:
                # No scenario tags in frame this frame
                if not scenario_stopped:
                    # Start / continue the "missing" timer
                    if scenario_last_seen is None:
                        scenario_last_seen = current_time

                    if (current_time - scenario_last_seen) > PERSISTENCE_THRESHOLD:
                        on_stop_scenario(None, {"timestamp": current_time})
                        scenario_stopped = True

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
