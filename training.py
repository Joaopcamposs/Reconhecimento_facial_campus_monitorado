import cv2
import numpy as np
from config import PICTURES_DIR, CLASSIFIER_PATH


class TrainingError(Exception):
    """Exception raised when training fails."""

    pass


def trainLBPH():
    """
    Train the LBPH face recognizer using images from the pictures directory.
    Returns a tuple (success: bool, message: str)
    """
    lbph = cv2.face.LBPHFaceRecognizer_create()

    def getImageAndId():
        # Get all jpg files from pictures directory
        paths = list(PICTURES_DIR.glob("person.*.*.jpg"))

        if not paths:
            raise TrainingError(
                f"Nenhuma imagem encontrada em {PICTURES_DIR}. "
                "Capture fotos primeiro usando o endpoint /fotos/{{camera_id}}&{{nome_pessoa}}"
            )

        faces = []
        ids = []

        for imagePath in paths:
            try:
                img = cv2.imread(str(imagePath))
                if img is None:
                    print(f"Aviso: Não foi possível ler a imagem {imagePath}")
                    continue
                face_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Extract ID from filename: person.{id}.{sample}.jpg
                id = int(imagePath.stem.split(".")[1])
                ids.append(id)
                faces.append(face_image)
            except Exception as e:
                print(f"Erro ao processar {imagePath}: {e}")
                continue

        if not faces:
            raise TrainingError("Nenhuma imagem válida foi processada.")

        return np.array(ids), faces

    try:
        ids, faces = getImageAndId()

        print(f"Treinando com {len(faces)} imagens...")

        lbph.train(faces, ids)
        lbph.write(str(CLASSIFIER_PATH))

        print("Treinamento concluído!")
        return True, f"Treinamento concluído com {len(faces)} imagens."

    except TrainingError as e:
        print(f"Erro no treinamento: {e}")
        return False, str(e)
    except Exception as e:
        print(f"Erro inesperado no treinamento: {e}")
        return False, f"Erro inesperado: {e}"
