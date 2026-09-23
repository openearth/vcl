"""Configuration loading and management."""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import numpy as np

logger = logging.getLogger(__name__)


def get_config_path(filename: str) -> Path:
    """Get the path to a configuration file in the config directory.

    Args:
        filename: Name of the config file (e.g., 'input.json', 'calibration.json')

    Returns:
        Path: Full path to the configuration file
    """
    config_dir = Path(__file__).parent.parent
    return config_dir / filename


def load_config(config_file: Optional[Path] = None) -> Dict[str, Any]:
    """Load main configuration from input.json.

    This configuration includes dataset paths, preprocessing settings, and
    rendering options.

    Args:
        config_file: Path to config file. If None, uses default input.json location.

    Returns:
        dict: Configuration dictionary with keys like 'basepath', 'layers', etc.

    Raises:
        FileNotFoundError: If config file doesn't exist
        json.JSONDecodeError: If config file is invalid JSON
    """
    if config_file is None:
        config_file = get_config_path("input.json")

    config_file = Path(config_file)

    if not config_file.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_file}")

    try:
        with open(config_file, "r") as f:
            config = json.load(f)
        logger.info(f"Loaded configuration from {config_file}")
        return config
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in configuration file {config_file}: {e}")
        raise


def load_camera_calibration(
    calib_file: Optional[Path] = None, default_points: Optional[list] = None
) -> np.ndarray:
    """Load camera calibration points from configuration file.

    Camera calibration maps screen coordinates to map coordinates for hand tracking
    and touch detection. Uses 4 corner points by default.

    Args:
        calib_file: Path to calibration file. If None, uses camera_calibration.json.
        default_points: Default points if file doesn't exist or fails to load.
                       Defaults to [(0.01, 0.88), (0.97, 0.87), (0.88, 0.13), (0.09, 0.16)]

    Returns:
        np.ndarray: Array of shape (4, 2) with calibration points
    """
    if default_points is None:
        default_points = [(0.01, 0.88), (0.97, 0.87), (0.88, 0.13), (0.09, 0.16)]

    if calib_file is None:
        calib_file = get_config_path("camera_calibration.json")

    calib_file = Path(calib_file)

    if calib_file.exists():
        try:
            with open(calib_file, "r") as f:
                data = json.load(f)
                points = data.get("camera_points", default_points)
                # Validate: must be 4 points with 2 coordinates each
                if len(points) == 4 and all(len(p) == 2 for p in points):
                    logger.info("Loaded camera calibration from %s", calib_file)
                    return np.array(points, dtype=np.float32)
                else:
                    logger.warning(
                        "Invalid calibration points in %s, using defaults", calib_file
                    )
        except Exception as e:
            logger.warning("Failed to load calibration from %s: %s", calib_file, e)

    logger.info("Using default camera calibration points")
    return np.array(default_points, dtype=np.float32)


def save_camera_calibration(points: list, calib_file: Optional[Path] = None) -> None:
    """Save camera calibration points to configuration file.

    Args:
        points: List of 4 (x, y) tuples representing calibration points
        calib_file: Path to calibration file. If None, uses camera_calibration.json.

    Raises:
        ValueError: If points don't have the expected format
        IOError: If file cannot be written
    """
    if calib_file is None:
        calib_file = get_config_path("camera_calibration.json")

    calib_file = Path(calib_file)

    # Validate input
    if len(points) != 4 or not all(len(p) == 2 for p in points):
        raise ValueError(
            "Camera calibration requires exactly 4 points with 2 coordinates each"
        )

    try:
        calib_file.parent.mkdir(parents=True, exist_ok=True)
        with open(calib_file, "w") as f:
            json.dump({"camera_points": points}, f, indent=4)
        logger.info("Saved camera calibration to %s", calib_file)
    except IOError as e:
        logger.error("Failed to save calibration to %s: %s", calib_file, e)
        raise
