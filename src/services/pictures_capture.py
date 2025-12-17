"""Pictures capture service module."""

import asyncio
import time
from collections.abc import AsyncGenerator
from typing import Any

import cv2
import numpy as np
from cv2 import CascadeClassifier, VideoCapture
from sqlalchemy.orm import Session

from src.entities.models import Camera
from src.entities.schemas import CreateAndUpdatePerson
from src.infra.config import (
    CAMERA_NOT_FOUND_IMAGE,
    CAMERA_OFF_IMAGE,
    HAARCASCADE_PATH,
    PICTURES_DIR,
    USE_WEBCAM_FALLBACK,
    get_ip_camera_capture,
    get_webcam_capture,
)
from src.repositories.camera_repository import CameraNotFound, get_camera_by_id
from src.repositories.controller_repository import (
    get_controller_by_id,
    reset_capture_flag,
)
from src.repositories.person_repository import create_person, get_all_persons

# Face detection parameters
SCALE_FACTOR: float = 1.1
MIN_NEIGHBORS: int = 5
MIN_SIZE: tuple[int, int] = (60, 60)

# Classifier setup
classifier: CascadeClassifier = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
if classifier.empty():
    print(f"Warning: Could not load cascade classifier from {HAARCASCADE_PATH}")

font: int = cv2.FONT_HERSHEY_COMPLEX_SMALL
width: int = 220
height: int = 220


def getNextID(session: Session) -> int:
    """Get next available person ID."""
    persons = get_all_persons(session)
    if not persons:
        return 1
    return max(p.person_id for p in persons) + 1


def _get_camera_capture(
    session: Session, camera_id: int
) -> tuple[VideoCapture | None, bool, Camera | None]:
    """Get camera capture object."""
    camera: Camera | None = None
    use_webcam: bool = False
    cameraIP: VideoCapture | None = None

    try:
        camera = get_camera_by_id(session=session, _id=camera_id)
    except CameraNotFound:
        camera = None

    if camera is not None:
        cameraIP = get_ip_camera_capture(camera.user, camera.password, camera.camera_ip)
        if not cameraIP.isOpened():
            cameraIP.release()
            cameraIP = None

    if cameraIP is None and USE_WEBCAM_FALLBACK:
        cameraIP = get_webcam_capture()
        use_webcam = True

    return cameraIP, use_webcam, camera


def _yield_error_image(image_path: Any) -> bytes | None:
    """Generate error image frame."""
    try:
        image = cv2.imread(str(image_path))
        if image is not None:
            _, encodedImage = cv2.imencode(".jpg", image)
            return (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
    return None


# Global capture state for manual capture mode
capture_state: dict[str, Any] = {
    "should_capture": False,
    "person_id": None,
    "person_name": None,
    "samples_captured": 0,
    "max_samples": 20,
    "is_active": False,
}


def start_capture_session(
    person_id: int, person_name: str, max_samples: int = 20
) -> None:
    """Start a new capture session."""
    global capture_state
    capture_state["person_id"] = person_id
    capture_state["person_name"] = person_name
    capture_state["samples_captured"] = 0
    capture_state["max_samples"] = max_samples
    capture_state["is_active"] = True
    capture_state["should_capture"] = False


def trigger_capture() -> None:
    """Trigger a photo capture."""
    global capture_state
    capture_state["should_capture"] = True


def get_capture_state() -> dict[str, Any]:
    """Get current capture state."""
    return capture_state.copy()


def reset_capture_state() -> None:
    """Reset capture state."""
    global capture_state
    capture_state = {
        "should_capture": False,
        "person_id": None,
        "person_name": None,
        "samples_captured": 0,
        "max_samples": 20,
        "is_active": False,
    }


async def stream_video_only(camera_id: int = 0) -> AsyncGenerator[bytes, None]:
    """Stream video with face detection without database dependency."""
    cameraIP: VideoCapture = get_webcam_capture()

    if camera_id > 0:
        pass

    if cameraIP is None or not cameraIP.isOpened():
        error_frame = _yield_error_image(CAMERA_NOT_FOUND_IMAGE)
        if error_frame:
            yield error_frame
        return

    try:
        while True:
            connected, frame = cameraIP.read()
            if not connected:
                await asyncio.sleep(0.01)
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_SIZE,
                )

                luminosity: int = int(np.average(gray_image))
                num_faces: int = len(detected_faces)

                person_name: str = capture_state.get("person_name") or "---"
                samples: int = capture_state.get("samples_captured", 0)
                max_samples: int = capture_state.get("max_samples", 20)
                person_id: int = capture_state.get("person_id", 0)
                is_active: bool = capture_state.get("is_active", False)

                status_text = f"Fotos: {samples}/{max_samples} | Lum: {luminosity} | Faces: {num_faces}"
                cv2.putText(frame, status_text, (10, 30), font, 1, (0, 255, 0), 2)

                if is_active and person_name:
                    cv2.putText(
                        frame,
                        f"Pessoa: {person_name}",
                        (10, 60),
                        font,
                        1,
                        (255, 255, 0),
                        2,
                    )

                if not is_active:
                    cv2.putText(
                        frame,
                        "Inicie uma sessao em /captura/iniciar/{nome}",
                        (10, 60),
                        font,
                        0.8,
                        (0, 255, 255),
                        1,
                    )

                for x, y, w, h in detected_faces:
                    color = (0, 255, 0) if is_active else (255, 165, 0)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                    if person_name and person_name != "---":
                        cv2.putText(frame, person_name, (x, y - 10), font, 1, color, 2)

                    if (
                        capture_state["should_capture"]
                        and is_active
                        and samples < max_samples
                    ):
                        if luminosity >= 60:
                            face_image = cv2.resize(
                                gray_image[y : y + h, x : x + w], (width, height)
                            )
                            image_path = (
                                PICTURES_DIR / f"person.{person_id}.{samples + 1}.jpg"
                            )
                            cv2.imwrite(str(image_path), face_image)
                            capture_state["samples_captured"] += 1
                            capture_state["should_capture"] = False

                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (255, 255, 255), 4
                            )
                            print(f"Captured: {image_path}")
                        else:
                            capture_state["should_capture"] = False

                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Error in video stream: {e}")

    finally:
        cameraIP.release()


async def stream_pictures_capture_auto(
    session: Session,
    camera_id: int,
    person_name: str,
    samples_number: int = 20,
    capture_interval: float = 0.5,
    min_luminosity: int = 80,
) -> AsyncGenerator[bytes, None]:
    """Auto-capture mode: captures photos automatically when face is detected."""
    samples: int = 0
    last_capture_time: float = 0

    person_id: int = getNextID(session)

    cameraIP, use_webcam, camera = _get_camera_capture(session, camera_id)

    if cameraIP is None or not cameraIP.isOpened():
        error_frame = _yield_error_image(CAMERA_NOT_FOUND_IMAGE)
        if error_frame:
            yield error_frame
        return

    try:
        while samples < samples_number:
            connected, frame = cameraIP.read()
            if not connected:
                await asyncio.sleep(0.01)
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_SIZE,
                )

                current_time: float = time.time()
                luminosity: int = int(np.average(gray_image))
                num_faces: int = len(detected_faces)

                cv2.putText(
                    frame,
                    f"Capturadas: {samples}/{samples_number} | Lum: {luminosity} | Faces: {num_faces}",
                    (10, 30),
                    font,
                    1,
                    (0, 255, 0),
                    2,
                )

                for x, y, w, h in detected_faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        person_name,
                        (x, y - 10),
                        font,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    if (current_time - last_capture_time) >= capture_interval:
                        if luminosity >= min_luminosity:
                            face_image = cv2.resize(
                                gray_image[y : y + h, x : x + w], (width, height)
                            )
                            image_path = (
                                PICTURES_DIR / f"person.{person_id}.{samples + 1}.jpg"
                            )
                            cv2.imwrite(str(image_path), face_image)
                            samples += 1
                            last_capture_time = current_time

                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (255, 255, 255), 4
                            )

                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Error in capture loop: {e}")

    finally:
        cameraIP.release()

    # Show completion message
    try:
        completion_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            completion_frame,
            f"Captura concluida! {samples} fotos salvas.",
            (50, 200),
            font,
            1.5,
            (0, 255, 0),
            2,
        )
        cv2.putText(
            completion_frame,
            f"Pessoa: {person_name} (ID: {person_id})",
            (50, 250),
            font,
            1,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            completion_frame,
            "Execute /treinamento para treinar o modelo",
            (50, 300),
            font,
            1,
            (0, 255, 255),
            1,
        )

        _, encodedImage = cv2.imencode(".jpg", completion_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    except Exception as e:
        print(f"Error showing completion: {e}")

    # Register person in database
    if samples > 0:
        person = CreateAndUpdatePerson(person_id=person_id, name=person_name)
        create_person(session=session, person_info=person)


async def stream_pictures_capture(
    session: Session, person_name: str, camera_id: int
) -> AsyncGenerator[bytes, None]:
    """Legacy capture mode using controller flag."""
    samples: int = 1
    samples_number: int = 20
    controller = None

    person_id: int = getNextID(session)

    try:
        controller = get_controller_by_id(session=session, _id=1)
    except Exception as e:
        print(f"Controller not found: {e}")

    cameraIP, use_webcam, camera = _get_camera_capture(session, camera_id)

    if cameraIP is None or not cameraIP.isOpened():
        error_frame = _yield_error_image(CAMERA_NOT_FOUND_IMAGE)
        if error_frame:
            yield error_frame
        return

    try:
        while samples <= samples_number:
            connected, frame = cameraIP.read()
            if not connected:
                await asyncio.sleep(0.01)
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_SIZE,
                )

                luminosity: int = int(np.average(gray_image))
                num_faces: int = len(detected_faces)

                cv2.putText(
                    frame,
                    f"Fotos: {samples - 1}/{samples_number} | Lum: {luminosity} | Faces: {num_faces}",
                    (10, 30),
                    font,
                    1,
                    (0, 255, 0),
                    2,
                )

                for x, y, w, h in detected_faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{person_name}",
                        (x, y - 10),
                        font,
                        1,
                        (0, 0, 255),
                    )

                    if controller and controller.save_picture == 1:
                        if luminosity > 80:
                            face_image = cv2.resize(
                                gray_image[y : y + h, x : x + w], (width, height)
                            )
                            image_path = (
                                PICTURES_DIR / f"person.{person_id}.{samples}.jpg"
                            )
                            cv2.imwrite(str(image_path), face_image)
                            samples += 1
                            reset_capture_flag(session, 1)
                            cv2.rectangle(
                                frame, (x, y), (x + w, y + h), (255, 255, 255), 4
                            )

                _, encodedImage = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Error in capture: {e}")

            if controller:
                try:
                    session.refresh(controller)
                    controller = get_controller_by_id(session=session, _id=1)
                except Exception:
                    pass

    finally:
        cameraIP.release()

    completion_frame = _yield_error_image(CAMERA_OFF_IMAGE)
    if completion_frame:
        yield completion_frame

    if samples > 1:
        person = CreateAndUpdatePerson(person_id=person_id, name=person_name)
        create_person(session=session, person_info=person)
