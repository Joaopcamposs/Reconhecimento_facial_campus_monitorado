"""Configuration module for cross-platform path handling and application settings."""

import platform
from pathlib import Path

import cv2
from cv2 import VideoCapture

# Base directory detection
BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent

# Asset paths
HAARCASCADE_PATH: Path = BASE_DIR / "src/recognizer/haarcascade_frontalface_default.xml"
CLASSIFIER_PATH: Path = BASE_DIR / "src/recognizer/classifierLBPH.yml"
CAMERA_NOT_FOUND_IMAGE: Path = BASE_DIR / "templates/assets/camera_nao_encontrada.jpg"
CAMERA_OFF_IMAGE: Path = BASE_DIR / "templates/assets/camera_desligada.jpg"
PICTURES_DIR: Path = BASE_DIR / "pictures"
VIDEOS_DIR: Path = BASE_DIR / "videos"

# Ensure directories exist
PICTURES_DIR.mkdir(exist_ok=True)
VIDEOS_DIR.mkdir(exist_ok=True)

# Webcam settings
DEFAULT_WEBCAM_INDEX: int = 0
USE_WEBCAM_FALLBACK: bool = True

# For macOS, we may need to use AVFoundation backend
WEBCAM_BACKEND: int | None = None
if platform.system() == "Darwin":
    WEBCAM_BACKEND = cv2.CAP_AVFOUNDATION


def get_webcam_capture(index: int | None = None) -> VideoCapture:
    """Get a VideoCapture object for the local webcam."""
    if index is None:
        index = DEFAULT_WEBCAM_INDEX

    if WEBCAM_BACKEND is not None:
        return cv2.VideoCapture(index, WEBCAM_BACKEND)
    return cv2.VideoCapture(index)


def get_ip_camera_capture(user: str, password: str, ip: str) -> VideoCapture:
    """Get a VideoCapture object for an IP camera via RTSP."""
    rtsp_url: str = f"rtsp://{user}:{password}@{ip}/"
    return cv2.VideoCapture(rtsp_url)


def classifier_exists() -> bool:
    """Check if the trained classifier file exists."""
    return CLASSIFIER_PATH.exists()
