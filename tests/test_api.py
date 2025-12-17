"""
Tests for the facial recognition API endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def client():
    """Create a test client."""
    # Mock database before importing app
    with patch("database.obter_uri_do_banco_de_dados") as mock_db:
        mock_db.return_value = "sqlite:///:memory:"
        with patch("crud.create_db"):
            from main import app

            with TestClient(app) as test_client:
                yield test_client


@pytest.fixture
def mock_db_session():
    """Create a mock database session."""
    session = MagicMock()
    return session


class TestStatusEndpoint:
    """Tests for the /status endpoint."""

    def test_status_endpoint_returns_ok(self, client):
        """Test that status endpoint returns proper structure."""
        with patch("api.get_webcam_capture") as mock_webcam:
            mock_cap = MagicMock()
            mock_cap.isOpened.return_value = False
            mock_webcam.return_value = mock_cap

            with patch("api.classifier_exists") as mock_classifier:
                mock_classifier.return_value = False

                response = client.get("/status")
                assert response.status_code == 200
                data = response.json()
                assert "status" in data
                assert data["status"] == "online"
                assert "webcam_available" in data
                assert "classifier_trained" in data
                assert "pictures_count" in data


class TestCameraEndpoints:
    """Tests for camera-related endpoints."""

    def test_list_cameras_empty(self, client):
        """Test listing cameras when none exist."""
        with patch("api.get_all_cameras") as mock_get:
            mock_get.return_value = []
            response = client.get("/cameras")
            assert response.status_code == 200

    def test_get_camera_not_found(self, client):
        """Test getting a camera that doesn't exist."""
        with patch("api.get_camera_by_id") as mock_get:
            mock_get.return_value = None
            response = client.get("/camera/999")
            assert response.status_code == 200


class TestPersonEndpoints:
    """Tests for person-related endpoints."""

    def test_list_persons_empty(self, client):
        """Test listing persons when none exist."""
        with patch("api.get_all_persons") as mock_get:
            mock_get.return_value = []
            response = client.get("/pessoas")
            assert response.status_code == 200


class TestTrainingEndpoint:
    """Tests for the /treinamento endpoint."""

    def test_training_no_images(self, client):
        """Test training when no images are available."""
        with patch("api.trainLBPH") as mock_train:
            mock_train.return_value = (False, "Nenhuma imagem encontrada")
            response = client.get("/treinamento")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "error"

    def test_training_success(self, client):
        """Test successful training."""
        with patch("api.trainLBPH") as mock_train:
            mock_train.return_value = (True, "Treinamento concluído com 20 imagens.")
            response = client.get("/treinamento")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"


class TestPicturesEndpoints:
    """Tests for picture management endpoints."""

    def test_list_pictures(self, client):
        """Test listing captured pictures."""
        with patch("api.PICTURES_DIR") as mock_dir:
            mock_dir.glob.return_value = []
            response = client.get("/fotos")
            assert response.status_code == 200
            data = response.json()
            assert "count" in data
            assert "pictures" in data

    def test_delete_pictures(self, client):
        """Test deleting all pictures."""
        with patch("api.PICTURES_DIR") as mock_dir:
            mock_dir.glob.return_value = []
            response = client.delete("/fotos")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "success"
            assert data["deleted"] == 0
