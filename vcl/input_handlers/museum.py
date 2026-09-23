"""Museum mode input handler with joystick and keyboard support."""

import logging
import sys
import time
import pygame
import zmq

try:
    from pynput import keyboard as pynput_keyboard
except ImportError:
    pynput_keyboard = None

from vcl.input_handlers.base import StateTracker, DoubleClickDetector

logger = logging.getLogger(__name__)


def museum_button_publisher(inactivity_timeout=120.0):
    """Publish a fixed 5-button command set for museum kiosk usage.

    This handler manages input from joystick buttons and global keyboard for museum mode.
    Supports double-press toggle feature for animation playback.

    Mappings:
        Joystick Buttons:
            0: land_use,layer
            1: bathymetry,layer
            2: gvg,layer
            3: grensvlak,layer
            4: gvg_difference,layer

        Keyboard (when available via pynput):
            1: bathymetry,layer
            2: satellite,animation
            3: grensvlak,layer
            4: gvg,layer
            5: gvg_difference,layer
            6: land_use,layer

    Args:
        inactivity_timeout: Seconds before state expires and double-press toggle resets
                           (matches displaymap inactivity_timeout).

    Note:
        Uses CONFLATE on ZMQ socket to only keep latest message.
        Requires pygame and optionally pynput for keyboard support.
    """
    # Initialize state tracking
    state_tracker = StateTracker()
    double_click_detector = DoubleClickDetector(timeout=inactivity_timeout)
    
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5556")

    joy_button_to_layer = {
        0: "land_use,layer",
        1: "bathymetry,layer",
        2: "gvg,layer",
        3: "grensvlak,layer",
        4: "gvg_difference,layer",
    }
    keyboard_key_to_layer = {
        "1": "bathymetry,layer",
        "2": "satellite,animation",
        "3": "grensvlak,layer",
        "4": "gvg,layer",
        "5": "gvg_difference,layer",
        "6": "land_use,layer",
    }

    def change_layer(text):
        """Handle layer change with double-press detection.
        
        Args:
            text: Layer specification in format "layer_name,layer_type"
        """
        layer_name = text.split(",")[0]
        layer_type = text.split(",")[1]

        # Check for double-press within timeout window
        is_double_press = double_click_detector.is_double_press(text)

        if text.split(",")[0] == "":
            socket.send_string(f"maps {text}")
            state_tracker.set_layer("")
            state_tracker.set_tide("")
        elif (
            (state_tracker.current_layer == text or state_tracker.current_tide == text)
            and is_double_press
        ):
            # Only trigger toggle if same button pressed within inactivity_timeout window
            socket.send_string(f"maps satellite,animation")
            if layer_type == "layer":
                state_tracker.set_layer("")  # Clear state after toggle
            elif layer_type == "tide":
                state_tracker.set_tide("")
        else:
            socket.send_string(f"maps {text}")
            if layer_type == "layer":
                state_tracker.set_layer(text)
            elif layer_type == "tide":
                state_tracker.set_tide(text)
            elif layer_type == "overlay":
                state_tracker.set_overlay(text, active=text not in state_tracker.current_overlays)
            elif layer_type == "animation":
                state_tracker.set_layer("")
                state_tracker.set_tide("")

    pressed_keyboard_keys = set()
    keyboard_listener = None

    def on_press(key):
        """Handle global keyboard press."""
        key_char = getattr(key, "char", None)
        if key_char not in keyboard_key_to_layer or key_char in pressed_keyboard_keys:
            return
        pressed_keyboard_keys.add(key_char)
        change_layer(keyboard_key_to_layer[key_char])

    def on_release(key):
        """Handle global keyboard release."""
        key_char = getattr(key, "char", None)
        if key_char is not None:
            pressed_keyboard_keys.discard(key_char)

    # Setup global keyboard listener if available
    if pynput_keyboard is not None:
        try:
            keyboard_listener = pynput_keyboard.Listener(
                on_press=on_press,
                on_release=on_release,
            )
            keyboard_listener.start()
            logger.info("Global keyboard listener started for museum mode")
        except Exception as exc:
            keyboard_listener = None
            logger.warning("Global museum keyboard listener unavailable: %s", exc)
    else:
        logger.warning(
            "pynput is not installed; use the focused map window keyboard shortcuts for museum-mode testing"
        )

    # Setup joystick support
    pygame.init()
    pygame.joystick.init()
    joysticks = []
    for joystick_index in range(pygame.joystick.get_count()):
        joystick = pygame.joystick.Joystick(joystick_index)
        joystick.init()
        joysticks.append(joystick)
        logger.info(f"Initialized joystick {joystick_index}: {joystick.get_name()}")

    if not joysticks and keyboard_listener is None:
        logger.warning(
            "museum_button_publisher running without joystick or global keyboard listener; "
            "use the map window keyboard shortcuts instead"
        )

    pressed_joystick_buttons = set()

    try:
        while True:
            pygame.event.pump()
            for joystick_index, joystick in enumerate(joysticks):
                for button_index, layer_name in joy_button_to_layer.items():
                    if button_index >= joystick.get_numbuttons():
                        pressed_joystick_buttons.discard((joystick_index, button_index))
                        continue

                    is_pressed = bool(joystick.get_button(button_index))
                    button_key = (joystick_index, button_index)
                    if is_pressed and button_key not in pressed_joystick_buttons:
                        pressed_joystick_buttons.add(button_key)
                        change_layer(layer_name)
                        logger.debug(f"Joystick {joystick_index} button {button_index} pressed: {layer_name}")
                    elif not is_pressed:
                        pressed_joystick_buttons.discard(button_key)

            time.sleep(0.02)
    finally:
        if keyboard_listener is not None:
            keyboard_listener.stop()
        pygame.joystick.quit()
        pygame.quit()
        sys.exit()
