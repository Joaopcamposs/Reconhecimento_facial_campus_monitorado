import cv2
import numpy as np
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

# Parameters for facial recognition
classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
width, height = 220, 220


def getNextID(session: Session):
    nextID = len(get_all_persons(session=session)) + 1
    return nextID


async def stream_pictures_capture(session: Session, camera_id: int, person_name: str):
    samples = 1
    samples_number = 20
    image = None
    camera = None

    id = getNextID(session)
    name = person_name

    try:
        camera = get_camera_by_id(session=session, _id=camera_id)
        controller = get_controller_by_id(session=session, _id=1)
    except CameraNotFound:
        image = cv2.imread("camera_nao_encontrada.jpg")
        (flag, encodedImage) = cv2.imencode(".jpg", image)
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encodedImage) + b"\r\n"
        )
    cameraIP = cv2.VideoCapture(0)  #Hardcoded WebCam
    if camera:
        cameraIP = cv2.VideoCapture(
            f"rtsp://{camera.user}:{camera.password}@{camera.camera_ip}/"
        )
    if camera:
        while samples <= samples_number:
            connected, frame = cameraIP.read()
            if connected:
                try:
                    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    detected_faces = classifier.detectMultiScale(
                        gray_image, scaleFactor=1.5, minSize=(150, 150)
                    )

                    for x, y, l, a in detected_faces:
                        cv2.rectangle(frame, (x, y), (x + l, y + a), (0, 0, 255), 2)
                        cv2.putText(
                            frame,
                            f"Luminosidade (min:110): {str(int(np.average(gray_image)))}",
                            (x, y + (a + 30)),
                            font,
                            1,
                            (0, 0, 255),
                        )
                        # if cv2.waitKey(1) & 0xFF == ord('q'):  # tecla 'q' captura as fotos
                        if controller.save_picture == 1:
                            if (
                                np.average(gray_image) > 110
                            ):  # captura apenas se a media de luminosidade for maior que 110
                                face_image = cv2.resize(
                                    gray_image[y : y + a, x : x + l], (width, height)
                                )
                                cv2.imwrite(
                                    "pictures/person."
                                    + str(id)
                                    + "."
                                    + str(samples)
                                    + ".jpg",
                                    face_image,
                                )
                                samples += 1
                                reset_capture_flag(session, 1)

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
            session.refresh(controller)
            camera = get_camera_by_id(session=session, _id=camera_id)
            controller = get_controller_by_id(session=session, _id=1)
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

        # Add name and id of the person in the database after capturing the images
        person = CreateAndUpdatePerson(person_id=id, name=name)
        person = create_person(person_info=person, session=session)
