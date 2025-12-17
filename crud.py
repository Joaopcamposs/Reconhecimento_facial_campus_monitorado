from typing import List
from sqlalchemy.orm import Session
from models import Camera, Person, Controller
from schema import CreateAndUpdateCamera, CreateAndUpdatePerson


# Function to get list of cameras
def get_all_cameras(session: Session) -> List[Camera]:
    return session.query(Camera).all()


class CameraNotFound(Exception):
    pass


# Function to get info of a particular camera
def get_camera_by_id(session: Session, _id: int) -> Camera:
    camera = session.query(Camera).get(_id)

    if camera is None:
        return None

    return camera


# Function to add a new camera to the database
def create_camera(session: Session, camera_info: CreateAndUpdateCamera) -> Camera:
    new_camera = Camera(**camera_info.dict())
    session.add(new_camera)
    session.commit()
    session.refresh(new_camera)
    return new_camera


# Function to update details of the camera
def update_camera(
    session: Session, _id: int, info_update: CreateAndUpdateCamera
) -> Camera:
    camera = get_camera_by_id(session, _id)

    if camera is None:
        raise Exception

    camera.camera_ip = info_update.camera_ip
    camera.user = info_update.user
    camera.status = info_update.status
    camera.password = info_update.password
    session.commit()
    session.refresh(camera)

    return camera


# Function to delete a camera from the db
def remove_camera(session: Session, _id: int):
    camera_info = get_camera_by_id(session, _id)

    if camera_info is None:
        raise Exception

    session.delete(camera_info)
    session.commit()

    return


# Function to get list of persons
def get_all_persons(session: Session) -> List[Person]:
    return session.query(Person).all()


class PersonNotFound(Exception):
    pass


# Function to get info of a particular person
def get_person_by_id(session: Session, _id: int) -> Person:
    pessoa = session.query(Person).get(_id)

    if pessoa is None:
        raise PersonNotFound

    return pessoa


# Function to add a new person to the database
def create_person(session: Session, person_info: CreateAndUpdatePerson) -> Person:
    new_person = Person(**person_info.dict())
    session.add(new_person)
    session.commit()
    session.refresh(new_person)
    return new_person


# Function to update details of the person
def update_person(
    session: Session, _id: int, info_update: CreateAndUpdatePerson
) -> Person:
    person = get_person_by_id(session, _id)

    if person is None:
        raise Exception

    person.person_id = info_update.person_id
    person.name = info_update.name
    session.commit()
    session.refresh(person)

    return person


# Function to delete a person from the db
def remove_person(session: Session, _id: int):
    person_info = get_person_by_id(session, _id)

    if person_info is None:
        raise Exception

    session.delete(person_info)
    session.commit()

    return


# Function to get info of controller of captures
def get_controller_by_id(session: Session, _id: int) -> Controller:
    controller = session.query(Controller).get(_id)

    if controller is None:
        raise Exception

    return controller


# Function to set captura flag
def set_capture_flag(session: Session, _id: int):
    controller = get_controller_by_id(session, _id)
    controller.save_picture = 1

    session.commit()
    session.refresh(controller)

    return controller


# Function to reset captura flag
def reset_capture_flag(session: Session, _id: int):
    controller = get_controller_by_id(session, _id)
    controller.save_picture = 0

    session.commit()
    session.refresh(controller)

    return controller


# Function to create database and tables
def create_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from database import obter_uri_do_banco_de_dados

    url_banco = obter_uri_do_banco_de_dados()
    url_banco = url_banco.replace("iftm", "")
    engine = create_engine(url_banco)
    Session = sessionmaker(engine)
    try:
        with Session.begin() as session:
            session.execute("CREATE DATABASE iftm;")
            session.execute("use iftm;")
            session.execute("""create table camera(
                            camera_id int auto_increment primary key,
                            user varchar(50),
                            camera_ip varchar(50),
                            password varchar(50),
                            status varchar(50)
                            );""")
            session.execute("""create table person(
                            person_id int auto_increment primary key,
                            name varchar(50)
                            );""")
            session.execute("""create table controller(
                            capture_id int primary key,
                            save_picture int
                            );""")
            session.execute("""insert into controller(capture_id, save_picture)
                            values (1, 0);""")
            session.commit()
    except:
        return "Something went wrong"

    return "created database"
