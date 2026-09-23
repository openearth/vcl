"""Keyboard input handler for interactive testing and development."""

import collections
import logging
import sys
import time
import pygame
import zmq

from vcl.input_handlers.base import StateTracker

logger = logging.getLogger(__name__)


def keyboard_publisher():
    """Keyboard input handler that publishes commands via ZMQ.

    This handler creates a Pygame window for keyboard input and translates keypresses
    into ZMQ messages that control the display windows. It handles layer selection,
    overlay toggling, year changes, and slice navigation.

    Key Mappings:
        Layer Selection:
            1: Bathymetry layer
            2: Satellite animation
            3: Risk zone
            4: Adjusted building strategy
            5: Protection strategy
            6: Compartment layer
            7: Shelter layer
            8: Crisis management layer
            0: Flooding layers

        Other Controls:
            A: Animation
            S: Bathymetry
            D: Breach locations overlay
            O: Overview
            P: Year 2100
            M: Toggle mask layer
            N: Cycle next layer collection
            B: Cycle previous layer collection
            LEFT/RIGHT: Navigate slice position

    ZMQ Topics:
        - maps: Layer change commands in format "layer_name,layer_type"
        - year: Year/time period changes
        - slice: Slice index position updates

    Note:
        Runs until window is closed. Uses StateTracker to maintain current layer state.
    """
    # Initialize state tracking
    state_tracker = StateTracker()
    
    # --- ZMQ Setup ---
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.setsockopt(zmq.CONFLATE, 1)
    socket.bind("tcp://*:5556")

    def change_year(year):
        """Send year change message."""
        socket.send_string(f"year {year}")

    # State tracking for layer collections
    waterdiepte = collections.deque(["d_T100_000", "d_T100"])
    risico_zone = collections.deque(["risico_zone", "risico_zone_20"])
    overviews = collections.deque(["overview", "overview_tags"])
    overstromingen = collections.deque(
        [
            "d_T1000_noord",
            "d_T1000_zuid",
            "d_T1000_oost",
            "d_T1000_west",
            "d_T1000_barendrecht",
            "d_T10_000_noord",
            "d_T10_000_zuid",
            "d_T10_000_oost",
            "d_T10_000_west",
            "d_T10_000_barendrecht",
            "d_T100_000_1_noord",
            "d_T100_000_1_zuid",
            "d_T100_000_1_oost",
            "d_T100_000_1_west",
            "d_T100_000_1_barendrecht",
            "d_T100_000_1_vp",
            "d_T100_000_1_hw",
            "d_T100_000_noord",
            "d_T100_000_zuid",
            "d_T100_000_oost",
            "d_T100_000_west",
        ]
    )

    collection_type_mapping = (
        {layer: "overstroming" for layer in overstromingen}
        | {layer: "waterdiepte" for layer in waterdiepte}
        | {layer: "risico_zone" for layer in risico_zone}
        | {layer: "overviews" for layer in overviews}
    )

    cycles = {
        "overstroming": overstromingen,
        "waterdiepte": waterdiepte,
        "risico_zone": risico_zone,
        "overviews": overviews,
    }

    def change_layer(text):
        """Handle layer change with toggle detection.
        
        Args:
            text: Layer specification in format "layer_name,layer_type"
        """
        layer_type = text.split(",")[1]
        if text.split(",")[0] == "":
            socket.send_string(f"maps {text}")
            state_tracker.set_layer("")
            state_tracker.set_tide("")
        if state_tracker.current_layer == text or state_tracker.current_tide == text:
            socket.send_string(f"maps None,{layer_type}")
            if layer_type == "layer":
                state_tracker.set_layer("")
            elif layer_type == "tide":
                state_tracker.set_tide("")
        else:
            socket.send_string(f"maps {text}")
            if layer_type == "layer":
                state_tracker.set_layer(text)
            elif layer_type == "tide":
                state_tracker.set_tide(text)
            elif layer_type == "overlay" and text in state_tracker.current_overlays:
                state_tracker.set_overlay(text, active=False)
            elif layer_type == "overlay" and text not in state_tracker.current_overlays:
                state_tracker.set_overlay(text, active=True)

    def cycle_collection(cycle):
        """Cycle through layer collections.
        
        Args:
            cycle: "next" or "prev" direction
        """
        if len(state_tracker.current_overlays) > 0:
            current_overlay = state_tracker.current_overlays[-1]
        else:
            current_overlay = ""

        if (
            current_overlay != ""
            and current_overlay.split(",")[0] in collection_type_mapping
        ):
            layer = current_overlay
        else:
            layer = state_tracker.current_layer

        if state_tracker.current_layer != "" and state_tracker.current_layer is not None:
            layer_name = layer.split(",")[0]
            layer_type = layer.split(",")[1]
        else:
            return

        collection_type = collection_type_mapping.get(layer_name, None)

        if collection_type is None:
            return

        if cycle == "next":
            cycles[collection_type].rotate(-1)
            next_layer = cycles[collection_type][0]
        elif cycle == "prev":
            cycles[collection_type].rotate(1)
            next_layer = cycles[collection_type][0]

        layer = f"{next_layer},{layer_type}"

        if state_tracker.current_layer in [
            f"{collection},{layer_type}" for collection in cycles[collection_type]
        ]:
            if layer_type == "layer":
                change_layer(layer)
        elif state_tracker.current_overlay in [
            f"{collection},{layer_type}" for collection in cycles[collection_type]
        ]:
            if layer_type == "overlay":
                if cycle == "prev":
                    change_layer(f"{cycles[collection_type][1]},overlay")
                else:
                    change_layer(f"{cycles[collection_type][-1]},overlay")
                time.sleep(0.01)
                change_layer(layer)

    # --- Pygame Setup ---
    pygame.init()
    screen_width, screen_height = 400, 200
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("Keyboard Publisher")
    font = pygame.font.Font(None, 36)
    keys_held = {}  # Dictionary to track held keys

    # --- Main Loop ---
    slice_index = 0
    max_slices = 300

    R_running = False
    L_running = False
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    change_layer("bathymetry,layer")
                elif event.key == pygame.K_2:
                    change_layer("satellite,animation")
                elif event.key == pygame.K_3:
                    change_layer(f"{risico_zone[0]},layer")
                elif event.key == pygame.K_4:
                    change_layer("aangepast_bouwen,layer")
                elif event.key == pygame.K_5:
                    change_layer("bescherming,layer")
                elif event.key == pygame.K_6:
                    change_layer("compartiment,layer")
                elif event.key == pygame.K_7:
                    change_layer("schuilen,layer")
                elif event.key == pygame.K_8:
                    change_layer("c_management,layer")
                elif event.key == pygame.K_0:
                    change_layer(f"{overstromingen[0]},layer")
                elif event.key == pygame.K_a:
                    change_layer("animation,layer")
                elif event.key == pygame.K_s:
                    change_layer("bathymetry,layer")
                elif event.key == pygame.K_d:
                    change_layer("doorbraaklocaties,overlay")
                elif event.key == pygame.K_o:
                    change_layer(f"{overviews[0]},layer")
                elif event.key == pygame.K_p:
                    change_year(2100)
                elif event.key == pygame.K_m:
                    change_layer("mask,mask")
                elif event.key == pygame.K_n:
                    cycle_collection("next")
                elif event.key == pygame.K_b:
                    cycle_collection("prev")

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RIGHT and event.key not in keys_held:
                    R_running = True
                elif event.key == pygame.K_LEFT and event.key not in keys_held:
                    L_running = True

            if event.type == pygame.KEYUP:
                if event.key == pygame.K_RIGHT:
                    R_running = False
                elif event.key == pygame.K_LEFT:
                    L_running = False

        # Handle continuous slice navigation
        if R_running:
            slice_index = (slice_index + 1) % max_slices
            socket.send_string(f"slice {slice_index}")
        if L_running:
            slice_index = (slice_index - 1) % max_slices
            socket.send_string(f"slice {slice_index}")

        # --- Display Instructions ---
        screen.fill((50, 50, 50))
        text = font.render(
            "Keyboard Publisher - Press 'ESC' to quit", True, (255, 255, 255)
        )
        screen.blit(text, (20, 80))
        pygame.display.flip()

        # Small delay to avoid consuming all CPU
        time.sleep(0.01)

    pygame.quit()
    sys.exit()
