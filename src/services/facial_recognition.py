"""Facial recognition service module."""

import asyncio
from collections.abc import AsyncGenerator

import cv2
from cv2 import CascadeClassifier, VideoCapture
from sqlalchemy.orm import Session

from src.entities.models import Camera, CameraStatus
from src.infra.config import (
    CAMERA_NOT_FOUND_IMAGE,
    CAMERA_OFF_IMAGE,
    CLASSIFIER_PATH,
    HAARCASCADE_PATH,
    USE_WEBCAM_FALLBACK,
    classifier_exists,
    get_ip_camera_capture,
    get_webcam_capture,
)
from src.repositories.camera_repository import CameraNotFound, get_camera_by_id
from src.repositories.person_repository import get_all_persons

# Parameters for facial recognition
faceDetector: CascadeClassifier = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
recognizer = cv2.face.LBPHFaceRecognizer_create()

if classifier_exists():
    recognizer.read(str(CLASSIFIER_PATH))

font: int = cv2.FONT_HERSHEY_COMPLEX_SMALL
width: int = 220
height: int = 220


def verifyPerson(session: Session, person_id: int) -> str | None:
    """Verify person name by ID."""
    persons = get_all_persons(session=session)
    for p in persons:
        if person_id == p.person_id:
            return p.name
    return None


def load_persons_cache() -> dict[int, str]:
    """Load all persons from database into a cache dict."""
    from src.infra.database import SessionLocal

    db: Session | None = None
    try:
        db = SessionLocal()
        persons = get_all_persons(session=db)
        cache: dict[int, str] = {p.person_id: p.name for p in persons}
        return cache
    except Exception as e:
        print(f"Error loading persons cache: {e}")
        return {}
    finally:
        if db is not None:
            db.close()


async def stream_recognition_only() -> AsyncGenerator[bytes, None]:
    """Stream facial recognition with person name lookup."""
    if not classifier_exists():
        image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
        cv2.putText(
            image,
            "Modelo nao treinado! Execute /treinamento primeiro.",
            (10, 30),
            font,
            1,
            (0, 0, 255),
            2,
        )
        _, encodedImage = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    recognizer.read(str(CLASSIFIER_PATH))
    persons_cache: dict[int, str] = load_persons_cache()

    cameraIP: VideoCapture = get_webcam_capture()
    if cameraIP is None or not cameraIP.isOpened():
        image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
        _, encodedImage = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    frame_count: int = 0
    try:
        while True:
            connected, frame = cameraIP.read()
            if not connected:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            if frame_count % 100 == 0:
                persons_cache = load_persons_cache()

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = faceDetector.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                cv2.putText(
                    frame,
                    "MODO: RECONHECIMENTO",
                    (10, 30),
                    font,
                    1,
                    (0, 255, 255),
                    2,
                )

                for x, y, w, h in detected_faces:
                    face_image = cv2.resize(
                        gray_image[y : y + h, x : x + w], (width, height)
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    try:
                        person_id, trust = recognizer.predict(face_image)
                        name: str = (
                            persons_cache.get(person_id, "Desconhecido")
                            if trust < 100
                            else "Desconhecido"
                        )

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
                        cv2.putText(frame, "?", (x, y - 10), font, 1, (0, 0, 255), 2)

                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Recognition error: {e}")
                await asyncio.sleep(0.01)

    finally:
        cameraIP.release()


async def stream_facial_recognition(
    session: Session, id_camera: int
) -> AsyncGenerator[bytes, None]:
    """Stream facial recognition from camera."""
    if not classifier_exists():
        image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
        cv2.putText(
            image,
            "Modelo nao treinado! Execute /treinamento primeiro.",
            (10, 30),
            font,
            1,
            (0, 0, 255),
            2,
        )
        _, encodedImage = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    recognizer.read(str(CLASSIFIER_PATH))

    camera: Camera | None = None
    use_webcam: bool = False

    try:
        camera = get_camera_by_id(session=session, _id=id_camera)
    except CameraNotFound:
        camera = None

    cameraIP: VideoCapture | None = None

    if camera is not None:
        cameraIP = get_ip_camera_capture(camera.user, camera.password, camera.camera_ip)
        if not cameraIP.isOpened():
            cameraIP.release()
            cameraIP = None

    if cameraIP is None and USE_WEBCAM_FALLBACK:
        cameraIP = get_webcam_capture()
        use_webcam = True
        if not cameraIP.isOpened():
            image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
            _, encodedImage = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )
            return
    elif cameraIP is None:
        image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
        _, encodedImage = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    should_run: bool = True
    while should_run:
        connected, frame = cameraIP.read()
        if connected:
            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = faceDetector.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                for x, y, w, h in detected_faces:
                    face_image = cv2.resize(
                        gray_image[y : y + h, x : x + w], (width, height)
                    )
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    person_id, trust = recognizer.predict(face_image)

                    name = verifyPerson(session, person_id)
                    if name is None:
                        name = "Desconhecido"

                    cv2.putText(frame, name, (x, y + (h + 30)), font, 2, (0, 0, 255))
                    cv2.putText(
                        frame,
                        str(f"Confianca: {round(trust, 2)}%"),
                        (x, y + (h + 50)),
                        font,
                        1,
                        (0, 0, 255),
                    )

                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                await asyncio.sleep(0.01)

            except Exception as e:
                print(e)

        if not use_webcam and camera:
            session.commit()
            session.refresh(camera)
            camera = get_camera_by_id(session=session, _id=id_camera)
            if camera and camera.status != CameraStatus.on:
                should_run = False

    cameraIP.release()
    cv2.destroyAllWindows()
    try:
        image = cv2.imread(str(CAMERA_OFF_IMAGE))
        _, encodedImage = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    except Exception as e:
        print(e)
