import json
import logging
from pathlib import Path
import numpy as np

# Use an absolute path or relative to this file to store the config
CONFIG_FILE = Path(__file__).parent / "camera_calibration.json"

DEFAULT_CAMERA_POINTS = [(0.01, 0.88), (0.97, 0.87), (0.88, 0.13), (0.09, 0.16)]

logger = logging.getLogger(__name__)


def load_camera_points() -> np.ndarray:
    """Load camera calibration points from the config file, fallback to defaults."""
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                data = json.load(f)
                points = data.get("camera_points", DEFAULT_CAMERA_POINTS)
                # Ensure it's exactly 4 points
                if len(points) == 4 and all(len(p) == 2 for p in points):
                    logger.info("Loaded custom camera calibration points.")
                    return np.array(points, dtype=np.float32)
                else:
                    logger.warning(
                        "Invalid configuration in %s, falling back to defaults.",
                        CONFIG_FILE,
                    )
        except Exception as e:
            logger.error("Failed to load %s: %s", CONFIG_FILE, e)

    logger.info("Using default camera calibration points.")
    return np.array(DEFAULT_CAMERA_POINTS, dtype=np.float32)


def save_camera_points(points: list[tuple[float, float]]):
    """Save camera calibration points to the config file."""
    try:
        with open(CONFIG_FILE, "w") as f:
            json.dump({"camera_points": points}, f, indent=4)
        logger.info("Camera calibration points saved to %s", CONFIG_FILE)
    except Exception as e:
        logger.error("Failed to save camera points to %s: %s", CONFIG_FILE, e)
