"""
Configuration module for cross-platform path handling and application settings.
"""

import os
from pathlib import Path

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Paths for resources
PICTURES_DIR = BASE_DIR / "pictures"
CLASSIFIER_PATH = BASE_DIR / "classifierLBPH.yml"
HAARCASCADE_PATH = BASE_DIR / "haarcascade_frontalface_default.xml"
CAMERA_OFF_IMAGE = BASE_DIR / "camera_desligada.jpg"
CAMERA_NOT_FOUND_IMAGE = BASE_DIR / "camera_nao_encontrada.jpg"

# Ensure pictures directory exists
PICTURES_DIR.mkdir(exist_ok=True)

# Camera settings
USE_WEBCAM_FALLBACK = os.getenv("USE_WEBCAM_FALLBACK", "true").lower() == "true"
DEFAULT_WEBCAM_INDEX = int(os.getenv("DEFAULT_WEBCAM_INDEX", "0"))

# For macOS, we may need to use AVFoundation backend
WEBCAM_BACKEND = None
import platform

if platform.system() == "Darwin":
    import cv2

    WEBCAM_BACKEND = cv2.CAP_AVFOUNDATION


def get_webcam_capture(index: int = None):
    """
    Get a VideoCapture object for the local webcam.
    Handles platform-specific backends.
    """
    import cv2

    if index is None:
        index = DEFAULT_WEBCAM_INDEX

    if WEBCAM_BACKEND is not None:
        return cv2.VideoCapture(index, WEBCAM_BACKEND)
    return cv2.VideoCapture(index)


def get_ip_camera_capture(user: str, password: str, ip: str):
    """
    Get a VideoCapture object for an IP camera via RTSP.
    """
    import cv2

    rtsp_url = f"rtsp://{user}:{password}@{ip}/"
    return cv2.VideoCapture(rtsp_url)


def classifier_exists() -> bool:
    """Check if the trained classifier file exists."""
    return CLASSIFIER_PATH.exists()
