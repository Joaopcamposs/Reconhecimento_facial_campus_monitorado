"""Video file analysis for facial recognition."""

from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy.orm import Session

from src.infra.config import CLASSIFIER_PATH, HAARCASCADE_PATH, classifier_exists
from src.repositories.person_repository import get_all_persons

# Initialize face detector and recognizer
faceDetector = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
recognizer = cv2.face.LBPHFaceRecognizer_create()

if classifier_exists():
    recognizer.read(str(CLASSIFIER_PATH))

font: int = cv2.FONT_HERSHEY_COMPLEX_SMALL
width: int = 220
height: int = 220


def verifyPerson(session: Session, person_id: int) -> str:
    """Verify person name by ID."""
    persons = get_all_persons(session=session)
    for p in persons:
        if person_id == p.person_id:
            return p.name
    return "Desconhecido"


async def analyze_video_file(
    session: Session,
    video_path: str,
    output_path: str | None = None,
    skip_frames: int = 2,
) -> AsyncGenerator[bytes, None]:
    """Analyze a video file and perform facial recognition."""
    if not classifier_exists():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_frame,
            "Erro: Modelo nao treinado!",
            (50, 200),
            font,
            1.5,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            error_frame,
            "Execute /treinamento primeiro",
            (50, 250),
            font,
            1,
            (255, 255, 255),
            1,
        )
        _, encodedImage = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    recognizer.read(str(CLASSIFIER_PATH))

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_frame,
            "Erro: Video nao encontrado",
            (50, 200),
            font,
            1.5,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            error_frame,
            str(video_path),
            (50, 250),
            font,
            0.7,
            (255, 255, 255),
            1,
        )
        _, encodedImage = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    cap = cv2.VideoCapture(str(video_path_obj))

    if not cap.isOpened():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_frame,
            "Erro: Nao foi possivel abrir o video",
            (50, 200),
            font,
            1,
            (0, 0, 255),
            2,
        )
        _, encodedImage = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps: float = cap.get(cv2.CAP_PROP_FPS)

    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width: int = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height: int = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )

    frame_count: int = 0
    faces_detected: int = 0
    recognized_persons: dict[str, dict[str, Any]] = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            if frame_count % skip_frames != 0:
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = faceDetector.detectMultiScale(
                    gray_image, scaleFactor=1.5, minSize=(30, 30)
                )

                for x, y, w, h in detected_faces:
                    faces_detected += 1
                    face_image = cv2.resize(
                        gray_image[y : y + h, x : x + w], (width, height)
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    try:
                        person_id, trust = recognizer.predict(face_image)
                        name = verifyPerson(session, person_id)

                        if name != "Desconhecido":
                            if name not in recognized_persons:
                                recognized_persons[name] = {
                                    "count": 0,
                                    "best_trust": trust,
                                }
                            recognized_persons[name]["count"] += 1
                            if trust < recognized_persons[name]["best_trust"]:
                                recognized_persons[name]["best_trust"] = trust

                        cv2.putText(frame, name, (x, y - 10), font, 1, (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Conf: {round(trust, 1)}",
                            (x, y + h + 20),
                            font,
                            1,
                            (0, 255, 0),
                            1,
                        )
                    except Exception:
                        cv2.putText(frame, "Erro", (x, y - 10), font, 1, (0, 0, 255), 2)

                progress: int = int((frame_count / total_frames) * 100)
                cv2.putText(
                    frame,
                    f"Progresso: {progress}% | Faces: {faces_detected}",
                    (10, 30),
                    font,
                    1,
                    (255, 255, 0),
                    1,
                )

                if video_writer:
                    video_writer.write(frame)

                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")

    finally:
        cap.release()
        if video_writer:
            video_writer.release()

    # Show summary
    summary_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(
        summary_frame,
        "Analise Concluida!",
        (50, 50),
        font,
        1.5,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        summary_frame,
        f"Frames processados: {frame_count}",
        (50, 100),
        font,
        1,
        (255, 255, 255),
        1,
    )
    cv2.putText(
        summary_frame,
        f"Faces detectadas: {faces_detected}",
        (50, 130),
        font,
        1,
        (255, 255, 255),
        1,
    )

    y_pos: int = 180
    cv2.putText(
        summary_frame,
        "Pessoas reconhecidas:",
        (50, y_pos),
        font,
        1,
        (0, 255, 255),
        1,
    )
    y_pos += 30

    for name, data in recognized_persons.items():
        cv2.putText(
            summary_frame,
            f"  {name}: {data['count']} deteccoes",
            (50, y_pos),
            font,
            1,
            (255, 255, 255),
            1,
        )
        y_pos += 25

    _, encodedImage = cv2.imencode(".jpg", summary_frame)
    yield (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
    )


def analyze_video_file_sync(
    session: Session, video_path: str, output_path: str | None = None
) -> dict[str, Any]:
    """Synchronous version that returns a summary instead of streaming."""
    if not classifier_exists():
        return {"status": "error", "message": "Modelo não treinado"}

    recognizer.read(str(CLASSIFIER_PATH))

    video_path_obj = Path(video_path)
    if not video_path_obj.exists():
        return {"status": "error", "message": f"Vídeo não encontrado: {video_path}"}

    cap = cv2.VideoCapture(str(video_path_obj))
    if not cap.isOpened():
        return {"status": "error", "message": "Não foi possível abrir o vídeo"}

    total_frames: int = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps: float = cap.get(cv2.CAP_PROP_FPS)

    frame_count: int = 0
    faces_detected: int = 0
    recognized_persons: dict[str, dict[str, Any]] = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % 3 != 0:
            continue

        try:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = faceDetector.detectMultiScale(
                gray_image, scaleFactor=1.5, minSize=(30, 30)
            )

            for x, y, w, h in detected_faces:
                faces_detected += 1
                face_image = cv2.resize(
                    gray_image[y : y + h, x : x + w], (width, height)
                )

                try:
                    person_id, trust = recognizer.predict(face_image)
                    name = verifyPerson(session, person_id)

                    if name not in recognized_persons:
                        recognized_persons[name] = {
                            "count": 0,
                            "best_confidence": float("inf"),
                            "person_id": person_id,
                        }
                    recognized_persons[name]["count"] += 1
                    if trust < recognized_persons[name]["best_confidence"]:
                        recognized_persons[name]["best_confidence"] = trust
                except Exception:
                    pass

        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()

    persons_list: list[dict[str, Any]] = []
    for name, data in recognized_persons.items():
        persons_list.append(
            {
                "name": name,
                "person_id": data["person_id"],
                "detections": data["count"],
                "best_confidence": round(data["best_confidence"], 2),
            }
        )

    return {
        "status": "success",
        "video_file": str(video_path),
        "total_frames": total_frames,
        "frames_processed": frame_count // 3,
        "fps": fps,
        "faces_detected": faces_detected,
        "recognized_persons": persons_list,
    }
