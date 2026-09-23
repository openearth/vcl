"""Configuration management for VCL.

This module provides centralized configuration loading and management for the
Virtual Climate Lab system. Configuration can come from:
- input.json (dataset and preprocessing config)
- camera_calibration.json (camera calibration)
- CLI arguments (overrides)
- Environment variables (museum mode settings)
"""

from vcl.config.settings import (
    load_config,
    load_camera_calibration,
    save_camera_calibration,
    get_config_path,
)

__all__ = [
    "load_config",
    "load_camera_calibration",
    "save_camera_calibration",
    "get_config_path",
]
