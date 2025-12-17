"""
Tests for the config module.
"""

import sys
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    BASE_DIR,
    PICTURES_DIR,
    CLASSIFIER_PATH,
    HAARCASCADE_PATH,
    classifier_exists,
    get_webcam_capture,
    get_ip_camera_capture,
)


class TestPaths:
    """Tests for path configuration."""

    def test_base_dir_exists(self):
        """Test that BASE_DIR is set correctly."""
        assert BASE_DIR.exists()

    def test_pictures_dir_exists(self):
        """Test that PICTURES_DIR is created."""
        assert PICTURES_DIR.exists()
        assert PICTURES_DIR.is_dir()

    def test_haarcascade_path_exists(self):
        """Test that haarcascade file exists."""
        assert HAARCASCADE_PATH.exists()

    def test_classifier_path_is_path(self):
        """Test that CLASSIFIER_PATH is a Path object."""
        assert isinstance(CLASSIFIER_PATH, Path)


class TestClassifierExists:
    """Tests for classifier_exists function."""

    def test_classifier_exists_returns_bool(self):
        """Test classifier_exists returns a boolean."""
        result = classifier_exists()
        assert isinstance(result, bool)

    def test_classifier_exists_matches_path_exists(self):
        """Test classifier_exists matches CLASSIFIER_PATH.exists()."""
        assert classifier_exists() == CLASSIFIER_PATH.exists()


class TestCameraCapture:
    """Tests for camera capture functions."""

    def test_get_webcam_capture_returns_video_capture(self):
        """Test that get_webcam_capture returns a VideoCapture object."""
        with patch("cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_vc.return_value = mock_cap

            result = get_webcam_capture(0)
            assert result == mock_cap

    def test_get_ip_camera_capture_builds_correct_url(self):
        """Test that get_ip_camera_capture builds the correct RTSP URL."""
        with patch("cv2.VideoCapture") as mock_vc:
            mock_cap = MagicMock()
            mock_vc.return_value = mock_cap

            result = get_ip_camera_capture("admin", "password", "192.168.1.100")

            mock_vc.assert_called_once_with("rtsp://admin:password@192.168.1.100/")
            assert result == mock_cap
