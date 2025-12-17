"""Training module for facial recognition classifier."""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from src.infra.config import CLASSIFIER_PATH, PICTURES_DIR


class TrainingError(Exception):
    """Exception raised when training fails."""

    pass


def trainLBPH() -> tuple[bool, str]:
    """Train the LBPH face recognizer with captured images."""
    recognizer = cv2.face.LBPHFaceRecognizer_create()

    def getImageAndId() -> tuple[NDArray[Any], list[NDArray[Any]]]:
        # Get all jpg files from pictures directory
        paths = list(PICTURES_DIR.glob("person.*.*.jpg"))

        if not paths:
            raise TrainingError(
                f"Nenhuma imagem encontrada em {PICTURES_DIR}. "
                "Capture fotos primeiro usando /captura"
            )

        faces: list[NDArray[Any]] = []
        ids: list[int] = []

        for imagePath in paths:
            try:
                # Extract person_id from filename pattern: person.{id}.{sample}.jpg
                person_id = int(imagePath.stem.split(".")[1])
                face_image = cv2.imread(str(imagePath))
                if face_image is not None:
                    gray_face = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
                    faces.append(gray_face)
                    ids.append(person_id)
            except Exception as e:
                print(f"Erro ao processar {imagePath}: {e}")
                continue

        if not faces:
            raise TrainingError("Nenhuma imagem válida foi processada.")

        return np.array(ids), faces

    try:
        ids, faces = getImageAndId()

        print(f"Treinando com {len(faces)} imagens...")
        recognizer.train(faces, ids)
        recognizer.write(str(CLASSIFIER_PATH))

        print("Treinamento concluído!")
        return True, f"Treinamento concluído com {len(faces)} imagens."

    except TrainingError as e:
        print(f"Erro no treinamento: {e}")
        return False, str(e)
    except Exception as e:
        print(f"Erro inesperado no treinamento: {e}")
        return False, f"Erro inesperado: {e}"
