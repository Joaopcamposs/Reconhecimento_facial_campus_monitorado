from time import sleep
import cv2
from fastapi import APIRouter, Depends, BackgroundTasks
from starlette.responses import StreamingResponse
from database import get_db
from sqlalchemy.orm import Session
from pictures_capture import stream_pictures_capture
from schema import CreateAndUpdateCamera, CreateAndUpdatePerson
from crud import (
    create_camera,
    get_camera_by_id,
    get_all_cameras,
    update_camera,
    remove_camera,
    get_all_persons,
    get_person_by_id,
    update_person,
    create_person,
    remove_person,
    set_capture_flag,
)
from training import trainLBPH
from facial_recognition import stream_facial_recognition

app = APIRouter()


# API endpoint to get info of a particular camera
@app.get("/camera/{camera_id}")
def pegar_info_camera(camera_id: int, session: Session = Depends(get_db)):
    try:
        camera_info = get_camera_by_id(session, camera_id)
        return camera_info
    except Exception as e:
        raise e


# API endpoint to update a existing camera info
@app.put("/camera/{camera_id}")
def atualizar_info_camera(
    camera_id: int,
    new_info: CreateAndUpdateCamera,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(update_camera, session, camera_id, new_info)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to get the list of cameras
@app.get("/cameras")
def listar_cameras(session: Session = Depends(get_db)):
    cameras = get_all_cameras(session=session)

    return cameras


# API endpoint to add a camera to the database
@app.post("/camera")
def cadastrar_camera(
    background_tasks: BackgroundTasks,
    new_camera: CreateAndUpdateCamera,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(create_camera, session, new_camera)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to delete a camera from the database
@app.delete("/camera/{camera_id}")
def deletar_camera(
    background_tasks: BackgroundTasks,
    camera_id: int,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(remove_camera, session, camera_id)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to get info of a particular pessoa
@app.get("/pessoa/{person_id}")
def pegar_info_pessoa(person_id: int, session: Session = Depends(get_db)):
    try:
        person_info = get_person_by_id(session, person_id)
        return person_info
    except Exception as e:
        raise e


# API endpoint to update a existing pessoa info
@app.put("/pessoa/{person_id}")
def atualizar_info_pessoa(
    person_id: int,
    new_info: CreateAndUpdatePerson,
    background_tasks: BackgroundTasks,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(update_person, session, person_id, new_info)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to get the list of pessoas
@app.get("/pessoas")
def listar_pessoas(session: Session = Depends(get_db)):
    persons = get_all_persons(session=session)

    return persons


# API endpoint to add a pessoa to the database
@app.post("/pessoa")
def cadastrar_pessoa(
    background_tasks: BackgroundTasks,
    new_person: CreateAndUpdatePerson,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(create_person, session, new_person)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to delete a car info from the data base
@app.delete("/pessoa/{person_id}")
def deletar_pessoa(
    background_tasks: BackgroundTasks,
    person_id: int,
    session: Session = Depends(get_db),
):
    try:
        background_tasks.add_task(remove_person, session, person_id)
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to train a new file of facial recognition
@app.get("/treinamento")
def treinar_reconhecimento():
    try:
        trainLBPH()
        return 200, "Requisição recebida"
    except Exception as e:
        raise e


# API endpoint to facial recognition stream
@app.get("/video/{camera_id}")
def reconhecimento_facial(camera_id: int, session: Session = Depends(get_db)):
    return StreamingResponse(
        stream_facial_recognition(session=session, id_camera=camera_id),
        media_type="multipart/x-mixed-replace;boundary=frame",
    )


# API endpoint to stream and catch pictures
@app.get("/fotos/{camera_id}&{nome_pessoa}")
def capturar_fotos(
    camera_id: int, nome_pessoa: str, session: Session = Depends(get_db)
):
    try:
        return StreamingResponse(
            stream_pictures_capture(
                session=session, camera_id=camera_id, person_name=nome_pessoa
            ),
            media_type="multipart/x-mixed-replace;boundary=frame",
        )
    except Exception as e:
        raise e


# API endpoint to catch the atual image to a picture
@app.post("/capturar")
def capturar(session: Session = Depends(get_db)):
    try:
        set_capture_flag(session, 1)
        return 200, "Requisicao recebida"
    except Exception as e:
        raise e


# API endpoint to start background cameras
@app.get("/background_cameras")
def iniciar_cameras_background(session: Session = Depends(get_db)):
    camera = get_camera_by_id(session, 1)
    camera_ip = cv2.VideoCapture(
        f"rtsp://{camera.user}:{camera.password}@{camera.camera_ip}/"
    )
    while True:
        image_ok, frame = camera_ip.read()
        if image_ok:
            print("running on background")
        sleep(30)
