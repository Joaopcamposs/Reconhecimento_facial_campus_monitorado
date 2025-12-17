import cv2
from crud import get_camera_by_id, get_all_persons
from models import CameraStatus
from sqlalchemy.orm import Session
from crud import CameraNotFound

# Parameters for facial recognition
faceDetector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("classifierLBPH.yml")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
width, height = 220, 220


def verifyPerson(session: Session, id: int):
    persons = get_all_persons(session=session)
    for p in persons:
        if id == p.person_id:
            return p.name


async def stream_facial_recognition(session: Session, id_camera: int):
    image = None
    camera = None
    try:
        camera = get_camera_by_id(session=session, _id=id_camera)
    except CameraNotFound:
        image = cv2.imread("camera_nao_encontrada.jpg")
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    cameraIP = cv2.VideoCapture(
        f"rtsp://{camera.user}:{camera.password}@{camera.camera_ip}/"
    )
    # cameraIP = cv2.VideoCapture(0)  #Hardcoded WebCam
    if camera:
        while camera.status == CameraStatus.on:
            connected, frame = cameraIP.read()
            if connected:
                try:
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = faceDetector.detectMultiScale(
                        gray_image, scaleFactor=1.5, minSize=(30, 30)
                    )
                    for x, y, l, a in detected_faces:
                        face_image = cv2.resize(
                            gray_image[y : y + a, x : x + l], (width, height)
                        )
                        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                        id, trust = recognizer.predict(face_image)

                        name = verifyPerson(session, id)

                        cv2.putText(
                            frame, name, (x, y + (a + 30)), font, 2, (0, 0, 255)
                        )
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
                except Exception as e:
                    print(e)
            session.commit()
            session.refresh(camera)
            camera = get_camera_by_id(session=session, _id=id_camera)
        cv2.destroyAllWindows()
        try:
            image = cv2.imread("camera_desligada.jpg")
            (flag, encodedImage) = cv2.imencode(".jpg", image)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
            )
        except Exception as e:
            print(e)
