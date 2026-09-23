"""Services for managing VCL processes and components.

This package provides service wrappers around display windows and input handlers,
with dependency injection and lifecycle management.

Services:
- process_manager: Coordinates launching and managing all processes
- display_service: Wraps display windows with configuration
- tracking_service: Wraps hand tracking
- detection_service: Wraps UID detection
"""

from vcl.services.process_manager import ProcessManager

__all__ = ["ProcessManager"]
