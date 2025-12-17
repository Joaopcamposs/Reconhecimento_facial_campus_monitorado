import cv2
import asyncio
from crud import get_camera_by_id, get_all_persons
from models import CameraStatus
from sqlalchemy.orm import Session
from crud import CameraNotFound
from config import (
    HAARCASCADE_PATH,
    CLASSIFIER_PATH,
    CAMERA_NOT_FOUND_IMAGE,
    CAMERA_OFF_IMAGE,
    get_webcam_capture,
    get_ip_camera_capture,
    USE_WEBCAM_FALLBACK,
    classifier_exists,
)

# Parameters for facial recognition
faceDetector = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
recognizer = cv2.face.LBPHFaceRecognizer_create()
# Only load classifier if it exists (after training)
if classifier_exists():
    recognizer.read(str(CLASSIFIER_PATH))
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
width, height = 220, 220


def verifyPerson(session: Session, id: int):
    persons = get_all_persons(session=session)
    for p in persons:
        if id == p.person_id:
            return p.name


def load_persons_cache():
    """Load all persons from database into a cache dict."""
    from database import SessionLocal

    try:
        db = SessionLocal()
        persons = get_all_persons(session=db)
        cache = {p.person_id: p.name for p in persons}
        db.close()
        return cache
    except Exception as e:
        print(f"Error loading persons cache: {e}")
        return {}


async def stream_recognition_only():
    """
    Stream facial recognition with person name lookup.
    Uses webcam directly - ideal for HTML interface.
    """
    # Check if classifier is trained
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
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    # Reload classifier
    recognizer.read(str(CLASSIFIER_PATH))

    # Load persons cache for name lookup
    persons_cache = load_persons_cache()

    # Use webcam directly
    cameraIP = get_webcam_capture()
    if cameraIP is None or not cameraIP.isOpened():
        image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    frame_count = 0
    try:
        while True:
            connected, frame = cameraIP.read()
            if not connected:
                await asyncio.sleep(0.01)
                continue

            frame_count += 1
            # Refresh persons cache every 100 frames
            if frame_count % 100 == 0:
                persons_cache = load_persons_cache()

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = faceDetector.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )

                # Draw mode indicator
                cv2.putText(
                    frame,
                    "MODO: RECONHECIMENTO",
                    (10, 30),
                    font,
                    1,
                    (0, 255, 255),
                    2,
                )

                for x, y, l, a in detected_faces:
                    face_image = cv2.resize(
                        gray_image[y : y + a, x : x + l], (width, height)
                    )
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)

                    try:
                        id, trust = recognizer.predict(face_image)
                        # Lookup person name from cache
                        name = (
                            persons_cache.get(id, "Desconhecido")
                            if trust < 100
                            else "Desconhecido"
                        )

                        cv2.putText(frame, name, (x, y - 10), font, 1, (0, 255, 0), 2)
                        cv2.putText(
                            frame,
                            f"Conf: {round(trust, 1)}",
                            (x, y + a + 20),
                            font,
                            1,
                            (0, 255, 0),
                            1,
                        )
                    except Exception:
                        cv2.putText(frame, "?", (x, y - 10), font, 1, (0, 0, 255), 2)

                (flag, encodedImage) = cv2.imencode(".jpg", frame)
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


async def stream_facial_recognition(session: Session, id_camera: int):
    # Check if classifier is trained
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
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    # Reload classifier to ensure latest trained model is used
    recognizer.read(str(CLASSIFIER_PATH))

    image = None
    camera = None
    use_webcam = False

    try:
        camera = get_camera_by_id(session=session, _id=id_camera)
    except CameraNotFound:
        camera = None

    # Initialize camera capture
    cameraIP = None

    if camera is not None:
        # Try to connect to IP camera
        cameraIP = get_ip_camera_capture(camera.user, camera.password, camera.camera_ip)
        if not cameraIP.isOpened():
            cameraIP.release()
            cameraIP = None

    # Fallback to webcam if IP camera not available
    if cameraIP is None and USE_WEBCAM_FALLBACK:
        cameraIP = get_webcam_capture()
        use_webcam = True
        if not cameraIP.isOpened():
            image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
            (flag, encodedImage) = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )
            return
    elif cameraIP is None:
        image = cv2.imread(str(CAMERA_NOT_FOUND_IMAGE))
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    # For webcam mode, we run continuously until stopped
    # For IP camera mode, we check the camera status
    should_run = True
    while should_run:
        connected, frame = cameraIP.read()
        if connected:
            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = faceDetector.detectMultiScale(
                    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
                )
                for x, y, l, a in detected_faces:
                    face_image = cv2.resize(
                        gray_image[y : y + a, x : x + l], (width, height)
                    )
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                    id, trust = recognizer.predict(face_image)

                    name = verifyPerson(session, id)
                    if name is None:
                        name = "Desconhecido"

                    cv2.putText(frame, name, (x, y + (a + 30)), font, 2, (0, 0, 255))
                    cv2.putText(
                        frame,
                        str(f"Confianca: {round(trust, 2)}%"),
                        (x, y + (a + 50)),
                        font,
                        1,
                        (0, 0, 255),
                    )

                # resize image (optional)
                frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

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
                print(e)

        # Check if we should continue running
        if not use_webcam and camera:
            session.commit()
            session.refresh(camera)
            camera = get_camera_by_id(session=session, _id=id_camera)
            if camera.status != CameraStatus.on:
                should_run = False

    cameraIP.release()
    cv2.destroyAllWindows()
    try:
        image = cv2.imread(str(CAMERA_OFF_IMAGE))
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    except Exception as e:
        print(e)
