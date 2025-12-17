"""
Video file analysis for facial recognition.
Allows analyzing pre-recorded video files instead of live streams.
"""

import cv2
import numpy as np
from pathlib import Path
from sqlalchemy.orm import Session
from crud import get_all_persons
from config import (
    HAARCASCADE_PATH,
    CLASSIFIER_PATH,
    classifier_exists,
)

# Parameters for facial recognition
faceDetector = cv2.CascadeClassifier(str(HAARCASCADE_PATH))
recognizer = cv2.face.LBPHFaceRecognizer_create()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
width, height = 220, 220


def verifyPerson(session: Session, id: int):
    """Get person name by ID."""
    persons = get_all_persons(session=session)
    for p in persons:
        if id == p.person_id:
            return p.name
    return "Desconhecido"


async def analyze_video_file(
    session: Session, video_path: str, output_path: str = None, skip_frames: int = 2
):
    """
    Analyze a video file and perform facial recognition on it.

    Args:
        session: Database session
        video_path: Path to the input video file
        output_path: Optional path to save the processed video
        skip_frames: Process every Nth frame (default 2 for performance)

    Yields:
        MJPEG frames for streaming
    """
    if not classifier_exists():
        # Return error frame
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
            "Execute GET /treinamento primeiro.",
            (50, 250),
            font,
            1,
            (255, 255, 255),
            1,
        )
        (flag, encodedImage) = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    # Load classifier
    recognizer.read(str(CLASSIFIER_PATH))

    # Open video file
    video_path = Path(video_path)
    if not video_path.exists():
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(
            error_frame,
            "Erro: Video nao encontrado",
            (50, 200),
            font,
            1,
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
        (flag, encodedImage) = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    cap = cv2.VideoCapture(str(video_path))

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
        (flag, encodedImage) = cv2.imencode(".jpg", error_frame)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
        return

    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Setup video writer if output path specified
    video_writer = None
    if output_path:
        output_path = Path(output_path)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        video_writer = cv2.VideoWriter(
            str(output_path), fourcc, fps, (frame_width, frame_height)
        )

    frame_count = 0
    faces_detected = 0
    recognized_persons = {}

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip frames for performance
            if frame_count % skip_frames != 0:
                continue

            try:
                gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detected_faces = faceDetector.detectMultiScale(
                    gray_image, scaleFactor=1.5, minSize=(30, 30)
                )

                for x, y, l, a in detected_faces:
                    faces_detected += 1
                    face_image = cv2.resize(
                        gray_image[y : y + a, x : x + l], (width, height)
                    )
                    cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 255, 0), 2)

                    try:
                        id, trust = recognizer.predict(face_image)
                        name = verifyPerson(session, id)

                        # Track recognized persons
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
                            (x, y + a + 20),
                            font,
                            0.8,
                            (0, 255, 0),
                            1,
                        )
                    except Exception:
                        cv2.putText(frame, "Erro", (x, y - 10), font, 1, (0, 0, 255), 2)

                # Draw progress info
                progress = int((frame_count / total_frames) * 100)
                cv2.putText(
                    frame,
                    f"Frame: {frame_count}/{total_frames} ({progress}%) | Faces: {faces_detected}",
                    (10, 30),
                    font,
                    0.8,
                    (255, 255, 0),
                    1,
                )

                # Write to output video if specified
                if video_writer:
                    video_writer.write(frame)

                # Encode and yield frame
                (flag, encodedImage) = cv2.imencode(".jpg", frame)
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
        f"Total de frames: {frame_count}",
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

    y_pos = 180
    cv2.putText(
        summary_frame,
        "Pessoas reconhecidas:",
        (50, y_pos),
        font,
        1,
        (255, 255, 0),
        1,
    )
    y_pos += 30

    for name, data in recognized_persons.items():
        cv2.putText(
            summary_frame,
            f"  - {name}: {data['count']} deteccoes",
            (50, y_pos),
            font,
            0.8,
            (255, 255, 255),
            1,
        )
        y_pos += 25

    (flag, encodedImage) = cv2.imencode(".jpg", summary_frame)
    yield (
        b"--frame\r\n"
        b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
    )


def analyze_video_file_sync(
    session: Session, video_path: str, output_path: str = None
) -> dict:
    """
    Synchronous version that returns a summary instead of streaming.

    Returns:
        Dictionary with analysis results
    """
    if not classifier_exists():
        return {"status": "error", "message": "Modelo não treinado"}

    recognizer.read(str(CLASSIFIER_PATH))

    video_path = Path(video_path)
    if not video_path.exists():
        return {"status": "error", "message": f"Vídeo não encontrado: {video_path}"}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"status": "error", "message": "Não foi possível abrir o vídeo"}

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    frame_count = 0
    faces_detected = 0
    recognized_persons = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # Process every 3rd frame for speed
        if frame_count % 3 != 0:
            continue

        try:
            gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detected_faces = faceDetector.detectMultiScale(
                gray_image, scaleFactor=1.5, minSize=(30, 30)
            )

            for x, y, l, a in detected_faces:
                faces_detected += 1
                face_image = cv2.resize(
                    gray_image[y : y + a, x : x + l], (width, height)
                )

                try:
                    id, trust = recognizer.predict(face_image)
                    name = verifyPerson(session, id)

                    if name not in recognized_persons:
                        recognized_persons[name] = {
                            "count": 0,
                            "best_confidence": float("inf"),
                            "person_id": id,
                        }
                    recognized_persons[name]["count"] += 1
                    if trust < recognized_persons[name]["best_confidence"]:
                        recognized_persons[name]["best_confidence"] = trust
                except Exception:
                    pass

        except Exception as e:
            print(f"Error processing frame: {e}")

    cap.release()

    # Format results
    persons_list = []
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
