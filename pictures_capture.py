import cv2
import numpy as np
import time
import asyncio
from sqlalchemy.orm import Session
from crud import (
    get_camera_by_id,
    CameraNotFound,
    create_person,
    get_all_persons,
    get_controller_by_id,
    reset_capture_flag,
)
from schema import CreateAndUpdatePerson
from config import (
    HAARCASCADE_PATH,
    PICTURES_DIR,
    CAMERA_NOT_FOUND_IMAGE,
    CAMERA_OFF_IMAGE,
    get_webcam_capture,
    get_ip_camera_capture,
    USE_WEBCAM_FALLBACK,
)

# Parameters for facial recognition
classifier = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
if classifier.empty():
    print(f"ERRO: Não foi possível carregar o classificador de {HAARCASCADE_PATH}")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
width, height = 220, 220

# Face detection parameters - more sensitive for better detection
SCALE_FACTOR = 1.1  # Lower = more sensitive (was 1.5, too aggressive)
MIN_NEIGHBORS = 5  # Higher = less false positives
MIN_SIZE = (60, 60)  # Minimum face size in pixels


def getNextID(session: Session):
    nextID = len(get_all_persons(session=session)) + 1
    return nextID


def _get_camera_capture(session: Session, camera_id: int):
    """Helper to get camera capture (IP or webcam fallback)."""
    camera = None
    use_webcam = False
    cameraIP = None

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


def _yield_error_image(image_path):
    """Helper to yield an error image."""
    image = cv2.imread(str(image_path))
    if image is not None:
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        return (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    return None


# Global state for manual capture mode
capture_state = {
    "should_capture": False,
    "person_id": None,
    "person_name": None,
    "samples_captured": 0,
    "max_samples": 20,
    "is_active": False,
}


def trigger_capture():
    """Trigger a manual capture."""
    capture_state["should_capture"] = True


def get_capture_state():
    """Get current capture state."""
    return capture_state.copy()


def reset_capture_state():
    """Reset capture state."""
    capture_state["should_capture"] = False
    capture_state["samples_captured"] = 0
    capture_state["is_active"] = False


def start_capture_session(person_id: int, person_name: str, max_samples: int = 20):
    """Start a new capture session."""
    capture_state["person_id"] = person_id
    capture_state["person_name"] = person_name
    capture_state["samples_captured"] = 0
    capture_state["max_samples"] = max_samples
    capture_state["is_active"] = True
    capture_state["should_capture"] = False


async def stream_video_only(camera_id: int = 0):
    """
    Stream video with face detection overlay.
    Used for the HTML interface - doesn't capture, just shows video with detection.
    Captures are triggered via the capture_state global.
    No database session needed - uses webcam directly.
    """
    # Direct webcam capture without database
    cameraIP = get_webcam_capture()
    if camera_id > 0:
        # For IP cameras, we'd need database but skip for now
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
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_SIZE,
                )

                luminosity = int(np.average(gray_image))
                num_faces = len(detected_faces)

                # Get current state
                person_name = capture_state.get("person_name", "---")
                samples = capture_state.get("samples_captured", 0)
                max_samples = capture_state.get("max_samples", 20)
                person_id = capture_state.get("person_id", 0)
                is_active = capture_state.get("is_active", False)

                # Draw status on frame
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

                for x, y, l, a in detected_faces:
                    # Draw face rectangle
                    color = (0, 255, 0) if is_active else (255, 165, 0)
                    cv2.rectangle(frame, (x, y), (x + l, y + a), color, 2)

                    if person_name and person_name != "---":
                        cv2.putText(frame, person_name, (x, y - 10), font, 1, color, 2)

                    # Check if capture was triggered
                    if (
                        capture_state["should_capture"]
                        and is_active
                        and samples < max_samples
                    ):
                        if luminosity >= 60:  # Minimum luminosity
                            face_image = cv2.resize(
                                gray_image[y : y + a, x : x + l], (width, height)
                            )
                            image_path = (
                                PICTURES_DIR / f"person.{person_id}.{samples + 1}.jpg"
                            )
                            cv2.imwrite(str(image_path), face_image)
                            capture_state["samples_captured"] += 1
                            capture_state["should_capture"] = False

                            # Flash effect
                            cv2.rectangle(
                                frame, (x, y), (x + l, y + a), (255, 255, 255), 4
                            )
                            print(f"Captured: {image_path}")
                        else:
                            capture_state["should_capture"] = False

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                # CRITICAL: Yield control to event loop so other requests can be processed
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
):
    """
    Auto-capture mode: automatically captures photos when a face is detected.
    No need to call /capturar endpoint - captures automatically with interval.

    Args:
        session: Database session
        camera_id: Camera ID (0 for webcam)
        person_name: Name of the person being captured
        samples_number: Number of photos to capture (default 20)
        capture_interval: Seconds between captures (default 0.5)
        min_luminosity: Minimum luminosity for capture (default 80)
    """
    samples = 0
    last_capture_time = 0

    person_id = getNextID(session)

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
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_SIZE,
                )

                current_time = time.time()
                luminosity = int(np.average(gray_image))
                num_faces = len(detected_faces)

                # Draw info on frame
                cv2.putText(
                    frame,
                    f"Capturadas: {samples}/{samples_number} | Lum: {luminosity} | Faces: {num_faces}",
                    (10, 30),
                    font,
                    1,
                    (0, 255, 0),
                    2,
                )

                for x, y, l, a in detected_faces:
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        person_name,
                        (x, y - 10),
                        font,
                        1,
                        (0, 255, 0),
                        2,
                    )

                    # Auto-capture with interval
                    if (current_time - last_capture_time) >= capture_interval:
                        if luminosity >= min_luminosity:
                            face_image = cv2.resize(
                                gray_image[y : y + a, x : x + l], (width, height)
                            )
                            image_path = (
                                PICTURES_DIR / f"person.{person_id}.{samples + 1}.jpg"
                            )
                            cv2.imwrite(str(image_path), face_image)
                            samples += 1
                            last_capture_time = current_time

                            # Flash effect to indicate capture
                            cv2.rectangle(
                                frame, (x, y), (x + l, y + a), (255, 255, 255), 4
                            )

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                # Yield control to event loop
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
            "Execute /treinamento para treinar o modelo.",
            (50, 300),
            font,
            1,
            (255, 255, 0),
            1,
        )
        (flag, encodedImage) = cv2.imencode(".jpg", completion_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    except Exception as e:
        print(f"Error showing completion: {e}")

    # Register person in database
    if samples > 0:
        person = CreateAndUpdatePerson(person_id=person_id, name=person_name)
        create_person(person_info=person, session=session)


async def stream_pictures_capture(session: Session, camera_id: int, person_name: str):
    """
    Manual capture mode: requires calling /capturar endpoint to capture each photo.
    Shows video stream with face detection overlay.
    """
    samples = 1
    samples_number = 20
    controller = None

    person_id = getNextID(session)

    # Get controller
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
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = classifier.detectMultiScale(
                    gray_image,
                    scaleFactor=SCALE_FACTOR,
                    minNeighbors=MIN_NEIGHBORS,
                    minSize=MIN_SIZE,
                )

                luminosity = int(np.average(gray_image))
                num_faces = len(detected_faces)

                # Draw status on frame
                cv2.putText(
                    frame,
                    f"Fotos: {samples - 1}/{samples_number} | Lum: {luminosity} | Faces: {num_faces}",
                    (10, 30),
                    font,
                    1,
                    (0, 255, 0),
                    2,
                )

                for x, y, l, a in detected_faces:
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                    cv2.putText(
                        frame,
                        f"{person_name}",
                        (x, y - 10),
                        font,
                        1,
                        (0, 0, 255),
                    )

                    # Check if capture was requested
                    if controller and controller.save_picture == 1:
                        if luminosity > 80:
                            face_image = cv2.resize(
                                gray_image[y : y + a, x : x + l], (width, height)
                            )
                            image_path = (
                                PICTURES_DIR / f"person.{person_id}.{samples}.jpg"
                            )
                            cv2.imwrite(str(image_path), face_image)
                            samples += 1
                            reset_capture_flag(session, 1)
                            # Flash effect
                            cv2.rectangle(
                                frame, (x, y), (x + l, y + a), (255, 255, 255), 4
                            )

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + bytearray(encodedImage)
                    + b"\r\n"
                )

                # Yield control to event loop
                await asyncio.sleep(0.01)

            except Exception as e:
                print(f"Error in capture: {e}")

            # Refresh controller state
            if controller:
                try:
                    session.refresh(controller)
                    controller = get_controller_by_id(session=session, _id=1)
                except Exception:
                    pass

    finally:
        cameraIP.release()

    # Show completion
    completion_frame = _yield_error_image(CAMERA_OFF_IMAGE)
    if completion_frame:
        yield completion_frame

    # Register person
    if samples > 1:
        person = CreateAndUpdatePerson(person_id=person_id, name=person_name)
        create_person(person_info=person, session=session)
