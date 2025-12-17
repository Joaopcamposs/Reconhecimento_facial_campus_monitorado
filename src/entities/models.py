import enum

from sqlalchemy.schema import Column
from sqlalchemy.types import Enum, Integer, String

from src.infra.database import Base


class CameraStatus(enum.Enum):
    on = enum.auto()
    off = enum.auto()


class Camera(Base):
    __tablename__ = "camera"
    camera_id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user: str = Column(String(50))
    camera_ip: str = Column(String(50))
    password: str = Column(String(50))
    status: CameraStatus = Column(Enum(CameraStatus))


class Person(Base):
    __tablename__ = "person"
    person_id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    name: str = Column(String(50))


class Controller(Base):
    __tablename__ = "controller"
    capture_id: int = Column(Integer, primary_key=True, index=True, autoincrement=True)
    save_picture: int = Column(Integer)
