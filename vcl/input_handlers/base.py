"""Base classes and utilities for input handlers."""

import logging
import time
from typing import Dict, Tuple
import zmq

logger = logging.getLogger(__name__)


class InputHandlerBase:
    """Base class for input handlers that publish to ZMQ sockets.
    
    Provides common functionality for keyboard, MIDI, and museum button handlers
    including ZMQ socket creation and message publishing.
    """
    
    def __init__(self, topic: str, port: int, bind: bool = True):
        """Initialize input handler with ZMQ socket.
        
        Args:
            topic: ZMQ topic name for this handler (e.g., 'maps', 'year')
            port: Port number to bind/connect to
            bind: If True, bind as publisher. If False, connect as subscriber.
        """
        self.topic = topic
        self.port = port
        self.bind = bind
        self.context = None
        self.socket = None
        
    def setup_socket(self) -> zmq.Socket:
        """Create and configure ZMQ socket.
        
        Returns:
            zmq.Socket: Configured socket ready for use
        """
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.PUB)
        self.socket.setsockopt(zmq.CONFLATE, 1)
        
        if self.bind:
            self.socket.bind(f"tcp://*:{self.port}")
            logger.info(f"Input handler '{self.topic}' binding to port {self.port}")
        else:
            self.socket.connect(f"tcp://localhost:{self.port}")
            logger.info(f"Input handler '{self.topic}' connecting to port {self.port}")
        
        return self.socket
    
    def send_message(self, message: str) -> None:
        """Send a message via ZMQ.
        
        Args:
            message: Message content to send
        """
        if self.socket is None:
            raise RuntimeError("Socket not initialized. Call setup_socket() first.")
        
        self.socket.send_string(f"{self.topic} {message}")
    
    def cleanup(self) -> None:
        """Clean up socket and context."""
        if self.socket is not None:
            self.socket.close()
        if self.context is not None:
            self.context.term()


class DoubleClickDetector:
    """Detect double-clicks/double-presses within a time window.
    
    Useful for toggle features: single-click selects layer, double-click toggles animation.
    """
    
    def __init__(self, timeout: float = 120.0):
        """Initialize double-click detector.
        
        Args:
            timeout: Time window (seconds) within which second press is considered a double-press
        """
        self.timeout = timeout
        self.last_press_times: Dict[str, float] = {}
    
    def is_double_press(self, key: str) -> bool:
        """Check if this key press is a double-press.
        
        Args:
            key: Identifier for the button/key (e.g., 'gvg,layer')
            
        Returns:
            bool: True if same key pressed within timeout window
        """
        current_time = time.time()
        time_since_last = current_time - self.last_press_times.get(key, 0)
        
        is_double = time_since_last < self.timeout
        self.last_press_times[key] = current_time
        
        return is_double
    
    def reset(self, key: str) -> None:
        """Reset the timer for a key."""
        self.last_press_times[key] = 0


class StateTracker:
    """Track current state of displayed layers and overlays.
    
    Maintains state across asynchronous input events and inactivity timeouts.
    """
    
    def __init__(self):
        """Initialize state tracker."""
        self.current_layer: str = ""
        self.current_overlay: str = ""
        self.current_tide: str = ""
        self.current_overlays: list = []
    
    def set_layer(self, layer: str) -> None:
        """Set the current layer."""
        self.current_layer = layer
        self.current_tide = ""
    
    def set_overlay(self, overlay: str, active: bool = True) -> None:
        """Add or remove an overlay."""
        if active:
            if overlay not in self.current_overlays:
                self.current_overlays.append(overlay)
            self.current_overlay = overlay
        else:
            if overlay in self.current_overlays:
                self.current_overlays.remove(overlay)
            self.current_overlay = ""
    
    def set_tide(self, tide: str) -> None:
        """Set the current tide visualization."""
        self.current_tide = tide
        self.current_layer = ""
    
    def clear(self) -> None:
        """Reset all state."""
        self.current_layer = ""
        self.current_overlay = ""
        self.current_tide = ""
        self.current_overlays = []
