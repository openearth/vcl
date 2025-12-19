import cv2
import mediapipe as mp
import numpy as np
import time
from typing import Iterable
import zmq
from typing import Union

# Define reference points (top-left, top-right, bottom-right, bottom-left)
CAMERA_POINTS = np.array(
    [(0.01, 0.88), (0.97, 0.87), (0.88, 0.13), (0.09, 0.16)], dtype=np.float32
)
TABLE_POINTS = np.array(
    [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)], dtype=np.float32
)


M = cv2.getPerspectiveTransform(CAMERA_POINTS, TABLE_POINTS)


def get_table_coordinate(normalized_x, normalized_y):
    """
    Takes a normalized point in the camera frame (0-1)
    Returns a normalized point on the table (0-1)
    """
    # Shape must be (1, 1, 2)
    point_array = np.array([[[normalized_x, normalized_y]]], dtype=np.float32)

    dst = cv2.perspectiveTransform(point_array, M)

    # Extract result
    x_out = dst[0][0][0]
    y_out = dst[0][0][1]

    return x_out, y_out


fgbg = cv2.createBackgroundSubtractorMOG2(
    history=500, varThreshold=50, detectShadows=False
)

# Initialize CLAHE object
# clipLimit: Threshold for contrast limiting. Higher values give more contrast. (e.g., 2.0 to 4.0)
# tileGridSize: Size of the grid for histogram equalization. (e.g., (8,8))
clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))


def apply_clahe_lab(image):
    """Applies CLAHE to the L channel of an image in LAB color space."""
    # 1. Convert BGR image to LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # 2. Split the LAB image into L, A, and B channels
    l, a, b = cv2.split(lab)

    # 3. Apply CLAHE to the L-channel (Lightness)
    cl = clahe.apply(l)

    # 4. Merge the CLAHE-enhanced L-channel back with the A and B channels
    limg = cv2.merge((cl, a, b))

    # 5. Convert the LAB image back to BGR (standard for display/MediaPipe input)
    final_frame = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)

    return final_frame


def distance_tips(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def angle(a, b, c):
    # angle at b formed by a-b-c
    ab = (a.x - b.x, a.y - b.y)
    cb = (c.x - b.x, c.y - b.y)
    dot = ab[0] * cb[0] + ab[1] * cb[1]
    mag_ab = np.sqrt(ab[0] ** 2 + ab[1] ** 2)
    mag_cb = np.sqrt(cb[0] ** 2 + cb[1] ** 2)
    return np.degrees(np.arccos(dot / (mag_ab * mag_cb)))


def get_finger_angles(lms, mp_hands, extended_angle=140, folded_angle=90):
    thumb_tip = lms.landmark[mp_hands.HandLandmark.THUMB_TIP]
    thumb_ip = lms.landmark[mp_hands.HandLandmark.THUMB_IP]
    thumb_mcp = lms.landmark[mp_hands.HandLandmark.THUMB_MCP]

    # Get index finger tip in camera space
    index_tip = lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
    index_pip = lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
    index_mcp = lms.landmark[mp_hands.HandLandmark.INDEX_FINGER_MCP]

    middle_tip = lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
    middle_pip = lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
    middle_mcp = lms.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_MCP]

    ring_tip = lms.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
    ring_pip = lms.landmark[mp_hands.HandLandmark.RING_FINGER_PIP]
    ring_mcp = lms.landmark[mp_hands.HandLandmark.RING_FINGER_MCP]

    pinky_tip = lms.landmark[mp_hands.HandLandmark.PINKY_TIP]
    pinky_pip = lms.landmark[mp_hands.HandLandmark.PINKY_PIP]
    pinky_mcp = lms.landmark[mp_hands.HandLandmark.PINKY_MCP]

    fingers = {
        "thumb": {"tip": thumb_tip, "pip": thumb_ip, "mcp": thumb_mcp},
        "index": {"tip": index_tip, "pip": index_pip, "mcp": index_mcp},
        "middle": {"tip": middle_tip, "pip": middle_pip, "mcp": middle_mcp},
        "ring": {"tip": ring_tip, "pip": ring_pip, "mcp": ring_mcp},
        "pinky": {"tip": pinky_tip, "pip": pinky_pip, "mcp": pinky_mcp},
    }
    for finger in fingers:
        finger_angle = angle(
            fingers[finger]["tip"], fingers[finger]["pip"], fingers[finger]["mcp"]
        )
        fingers[finger]["angle"] = finger_angle
        fingers[finger]["extended"] = finger_angle > extended_angle
        fingers[finger]["folded"] = finger_angle < folded_angle

    return fingers


def webcam_module(
    extent: Union[tuple, list],
    socket: zmq.Socket,
    socket_topic: str,
    device_index: int = 0,
    max_number_of_hands: int = 1,
    frame_size: Iterable[float] = None,
    click_hold_time: float = 1.5,
    circle_hold_time: float = 1.0,
    click_cooldown_time: float = 2.5,
    click_tolerance: float = 0.01,
    show_polygon: bool = True,
    show_hands: bool = True,
    show_mappings: bool = True,
    print_mappings: bool = False,
    calibrate: bool = False,
) -> None:
    """Webcame module that captures video from a specified device, detects hand landmarks using Mediapipe,
    and maps the index finger tip position through camera, table, and map coordinate spaces.
    Also detects click gestures when finger stays in position for specified duration.

    Args:
        device_index (int): Index of the video capture device.
        max_number_of_hands (int): Maximum number of hands to detect.
        frame_size (Iterable[float]): Desired frame size as (width, height). If None, defaults to (1280, 720).
        click_hold_time (float): Time in seconds to hold position for click detection.
        click_cooldown_time (float): Time in seconds for cooldown after a click is detected.
        click_tolerance (float): Maximum distance (normalized) finger can move while "holding" position.
        show_polygon (bool): Whether to display the reference polygon in camera space.
        show_hands (bool): Whether to display the hand landmarks.
        show_mappings (bool): Whether to display the coordinate mappings on the video feed.
        print_mappings (bool): Whether to print the coordinate mappings to the console.

    Returns:
        None
    """
    # Frame dimensions
    if frame_size is None:
        frame_size = (1280, 720)

    xmin, ymin, xmax, ymax = extent
    MAP_POINTS = np.array(
        [
            (xmin, ymin),
            (xmax, ymin),
            (xmax, ymax),
            (xmin, ymax),
        ]
    )

    # Compute homographies
    H_camera_to_table, _ = cv2.findHomography(CAMERA_POINTS, TABLE_POINTS)
    H_table_to_map, _ = cv2.findHomography(TABLE_POINTS, MAP_POINTS)

    # Setup Mediapipe
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        model_complexity=1,
        max_num_hands=max_number_of_hands,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    )

    # Click detection state tracking
    hand_states = {}  # Dictionary to track state for each hand

    # Start video capture
    print("Starting video capture (press ESC to exit)")
    cap = cv2.VideoCapture(device_index)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Video capture device could not be opened")
        return

    # Video processing loop
    n_clicks = 0
    while cap.isOpened():
        # Read frame
        ret, frame = cap.read()

        if not ret:
            break

        # Preprocess frame
        frame = cv2.flip(frame, 1)
        frame = cv2.resize(frame, frame_size)
        h, w, _ = frame.shape

        if not calibrate:
            src_points = (CAMERA_POINTS * [w, h]).astype(np.float32)
            dst_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

            M = cv2.getPerspectiveTransform(src_points, dst_points)
            # M, _ = cv2.findHomography(CAMERA_POINTS, dst_points)
            frame = cv2.warpPerspective(frame, M, (w, h))

        fgmask = fgbg.apply(frame)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, None)
        # frame = cv2.cvtColor(fgmask, cv2.COLOR_GRAY2BGR)

        # # Find contours in mask
        # contours, _ = cv2.findContours(
        #     fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        # )

        # frame = apply_clahe_lab(frame)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        # Get current time
        current_time = time.time()

        # Initialize hand states set
        current_hands = set()

        # Draw reference polygon in camera space
        if calibrate:
            polygon_pixels = (CAMERA_POINTS * [w, h]).astype(int)
            for i in range(4):
                p1 = tuple(polygon_pixels[i])
                p2 = tuple(polygon_pixels[(i + 1) % 4])
                cv2.line(frame, p1, p2, (0, 255, 0), 2)
            for p in polygon_pixels:
                cv2.circle(frame, tuple(p), 5, (0, 255, 0), -1)

        # Hand detection and mapping
        if result.multi_hand_landmarks:
            for i, lms in enumerate(result.multi_hand_landmarks):
                # Add hand index to current hands set
                current_hands.add(i)

                fingers = get_finger_angles(lms=lms, mp_hands=mp_hands)

                camera_x, camera_y = (
                    fingers["index"]["tip"].x,
                    fingers["index"]["tip"].y,
                )

                # Convert to table space
                table_coords = cv2.perspectiveTransform(
                    np.array([[[camera_x, camera_y]]]), H_camera_to_table
                )[0][0]
                table_x, table_y = table_coords
                table_x, table_y = get_table_coordinate(camera_x, camera_y)
                if not calibrate:
                    table_x, table_y = camera_x, camera_y

                # Convert to map space
                map_coords = cv2.perspectiveTransform(
                    np.array([[[table_x, table_y]]]), H_table_to_map
                )[0][0]
                map_x, map_y = map_coords

                # Initialize hand state if not exists
                if i not in hand_states:
                    hand_states[i] = {
                        "last_position": (camera_x, camera_y),
                        "hold_start_time": None,
                        "last_click_time": 0,
                        "is_holding": False,
                        "click_detected": False,
                        "circle_start_time": None,
                        "last_circle_time": 0,
                        "circle_detected": False,
                        "rclick_start_time": None,
                        "last_rclick_time": 0,
                        "rclick_detected": False,
                    }

                # Click detection logic
                state = hand_states[i]
                last_pos = state["last_position"]

                # Calculate distance from last position
                distance = np.sqrt(
                    (camera_x - last_pos[0]) ** 2 + (camera_y - last_pos[1]) ** 2
                )

                index_pointing = (
                    fingers["index"]["extended"]
                    and fingers["middle"]["folded"]
                    and fingers["ring"]["folded"]
                    and fingers["pinky"]["folded"]
                )

                index_pointing_2 = (
                    fingers["index"]["extended"]
                    and fingers["middle"]["extended"]
                    and fingers["ring"]["folded"]
                    and fingers["pinky"]["folded"]
                )

                thumb_index_d = distance_tips(
                    fingers["thumb"]["tip"], fingers["index"]["tip"]
                )
                finger_circle = thumb_index_d < 0.05

                # Check if hand is in cooldown period
                in_cooldown = (
                    current_time - state["last_click_time"]
                ) < click_cooldown_time

                if distance <= click_tolerance and not in_cooldown and index_pointing:
                    # Hand is steady
                    if state["hold_start_time"] is None:
                        # Start holding
                        state["hold_start_time"] = current_time
                        state["is_holding"] = True
                    else:
                        # Check if held long enough for click
                        hold_duration = current_time - state["hold_start_time"]
                        if (
                            hold_duration >= click_hold_time
                            and not state["click_detected"]
                        ):
                            # Click detected!
                            state["click_detected"] = True
                            state["last_click_time"] = current_time
                            n_clicks += 1
                            print(f"\nHand {i}: Clicked! {n_clicks}")

                            # Reset hold state
                            state["hold_start_time"] = None
                            state["is_holding"] = False

                else:
                    # Hand moved too much or in cooldown, reset hold state
                    state["hold_start_time"] = None
                    state["is_holding"] = False
                    state["click_detected"] = False

                if finger_circle:
                    if state["circle_start_time"] is None:
                        state["circle_start_time"] = current_time
                    else:
                        circle_duration = current_time - state["circle_start_time"]
                        if (
                            circle_duration >= circle_hold_time
                            and not state["circle_detected"]
                        ):
                            state["circle_detected"] = True
                            state["last_circle_time"] = current_time
                            state["circle_start_time"] = None

                if distance <= click_tolerance and index_pointing_2:
                    # Hand is steady
                    if state["rclick_start_time"] is None:
                        # Start holding
                        state["rclick_start_time"] = current_time
                    else:
                        # Check if held long enough for click
                        rclick_duration = current_time - state["rclick_start_time"]
                        if (
                            rclick_duration >= click_hold_time
                            and not state["rclick_detected"]
                        ):
                            # Click detected!
                            state["rclick_detected"] = True
                            state["last_rclick_time"] = current_time
                            # Reset hold state
                            state["rclick_start_time"] = None

                else:
                    # Hand moved too much or in cooldown, reset hold state
                    state["rclick_start_time"] = None
                    state["rclick_detected"] = False

                # Update last position
                state["last_position"] = (camera_x, camera_y)

                # Show hands
                if show_hands:
                    mp_drawing.draw_landmarks(frame, lms, mp_hands.HAND_CONNECTIONS)

                # Visual feedback for click detection
                finger_pixel = (int(camera_x * w), int(camera_y * h))

                if in_cooldown:
                    # Show cooldown state (blue circle)
                    cv2.circle(frame, finger_pixel, 15, (255, 0, 0), 3)
                    cv2.putText(
                        frame,
                        "COOLDOWN",
                        (finger_pixel[0] - 30, finger_pixel[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
                elif state["is_holding"]:
                    # Show holding progress (yellow circle with progress arc)
                    hold_duration = current_time - state["hold_start_time"]
                    progress = min(hold_duration / click_hold_time, 1.0)
                    end_angle = int(360 * progress)

                    cv2.circle(frame, finger_pixel, 15, (0, 255, 255), 3)
                    cv2.ellipse(
                        frame, finger_pixel, (15, 15), 0, 0, end_angle, (0, 255, 0), 3
                    )
                    cv2.putText(
                        frame,
                        f"HOLD {progress:.1%}",
                        (finger_pixel[0] - 30, finger_pixel[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )
                elif (
                    state["click_detected"]
                    and (current_time - state["last_click_time"]) < 1.0
                ):
                    # Show click confirmation (green circle)
                    cv2.circle(frame, finger_pixel, 15, (0, 255, 0), -1)
                    cv2.putText(
                        frame,
                        "CLICK",
                        (finger_pixel[0] - 30, finger_pixel[1] - 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

                    socket.send_string(f"{socket_topic} {table_x},{table_y}")

                if (
                    state["circle_detected"]
                    and (current_time - state["last_circle_time"]) < 1.0
                    and table_x >= 0
                    and table_x <= 1
                    and table_y >= 0
                    and table_y <= 1
                ):
                    state["circle_detected"] = False
                    # socket.send_string(f"maps animation,layer")

                if (
                    state["rclick_detected"]
                    and (current_time - state["last_rclick_time"]) < 1.0
                ):
                    socket.send_string(f"{socket_topic} -1,-1")

                # Show mappings
                if show_mappings:
                    cv2.putText(
                        frame,
                        f"Camera: ({camera_x:.2f}, {camera_y:.2f})",
                        (10, 30 + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 0, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Table: ({table_x:.2f}, {table_y:.2f})",
                        (10, 50 + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        frame,
                        f"Map: ({map_x:.2f}, {map_y:.2f})",
                        (10, 70 + i * 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 0, 255),
                        2,
                    )

                # Print mappings
                if print_mappings:
                    hold_status = "-"
                    if in_cooldown:
                        hold_status = "COOLDOWN"
                    elif state["is_holding"]:
                        hold_duration = current_time - state["hold_start_time"]
                        progress = min(hold_duration / click_hold_time, 1.0)
                        hold_status = f"HOLD {progress:.1%}"
                    elif (
                        state["click_detected"]
                        and (current_time - state["last_click_time"]) < 1.0
                    ):
                        hold_status = "CLICK"

                    print(
                        f"Hand {i}: Camera({camera_x:.2f}, {camera_y:.2f}) | Table({table_x:.2f}, {table_y:.2f}) | Map({map_x:.2f}, {map_y:.2f}) [{hold_status}]         ",
                        end="\r",
                    )
        else:
            if print_mappings:
                print(
                    "No hands detected.                                                             ",
                    end="\r",
                )

        # Clean up hand states for hands that are no longer detected
        hands_to_remove = []
        for hand_id in hand_states.keys():
            if hand_id not in current_hands:
                hands_to_remove.append(hand_id)
        for hand_id in hands_to_remove:
            del hand_states[hand_id]

        # Show frame
        cv2.imshow("Webcam Module", frame)

        # Exit on ESC key
        if cv2.waitKey(1) & 0xFF == 27:
            break

    print("Exiting video capture")

    # Release resources
    cap.release()
    cv2.destroyAllWindows()
