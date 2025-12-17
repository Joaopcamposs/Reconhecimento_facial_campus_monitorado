"""
Tests for the training module.
"""

import pytest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from training import trainLBPH, TrainingError


class TestTrainLBPH:
    """Tests for the trainLBPH function."""

    def test_training_no_images(self):
        """Test training fails gracefully when no images exist."""
        with patch("training.PICTURES_DIR") as mock_dir:
            mock_dir.glob.return_value = []

            success, message = trainLBPH()

            assert success is False
            assert "Nenhuma imagem encontrada" in message

    def test_training_with_images(self):
        """Test training succeeds with valid images."""
        with patch("training.PICTURES_DIR") as mock_dir:
            # Create mock paths
            mock_path1 = MagicMock()
            mock_path1.stem = "person.1.1"
            mock_path1.__str__ = lambda x: "/fake/person.1.1.jpg"

            mock_path2 = MagicMock()
            mock_path2.stem = "person.1.2"
            mock_path2.__str__ = lambda x: "/fake/person.1.2.jpg"

            mock_dir.glob.return_value = [mock_path1, mock_path2]

            with patch("cv2.imread") as mock_imread:
                # Create a fake grayscale image
                fake_image = np.zeros((220, 220, 3), dtype=np.uint8)
                mock_imread.return_value = fake_image

                with patch("cv2.cvtColor") as mock_cvt:
                    mock_cvt.return_value = np.zeros((220, 220), dtype=np.uint8)

                    with patch("cv2.face.LBPHFaceRecognizer_create") as mock_lbph:
                        mock_recognizer = MagicMock()
                        mock_lbph.return_value = mock_recognizer

                        with patch("training.CLASSIFIER_PATH") as mock_path:
                            mock_path.__str__ = lambda x: "/fake/classifier.yml"

                            success, message = trainLBPH()

                            assert success is True
                            assert "Treinamento concluído" in message
                            mock_recognizer.train.assert_called_once()
                            mock_recognizer.write.assert_called_once()

    def test_training_handles_invalid_image(self):
        """Test training skips invalid images."""
        with patch("training.PICTURES_DIR") as mock_dir:
            mock_path = MagicMock()
            mock_path.stem = "person.1.1"
            mock_path.__str__ = lambda x: "/fake/person.1.1.jpg"
            mock_dir.glob.return_value = [mock_path]

            with patch("cv2.imread") as mock_imread:
                # Return None to simulate failed image read
                mock_imread.return_value = None

                success, message = trainLBPH()

                assert success is False
                assert "Nenhuma imagem válida" in message


class TestTrainingError:
    """Tests for the TrainingError exception."""

    def test_training_error_message(self):
        """Test TrainingError can be raised with a message."""
        with pytest.raises(TrainingError) as exc_info:
            raise TrainingError("Test error message")

        assert "Test error message" in str(exc_info.value)
