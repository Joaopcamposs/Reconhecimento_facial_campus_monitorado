from pydantic import BaseModel

from src.entities.models import CameraStatus


class CreateAndUpdateCamera(BaseModel):
    user: str
    camera_ip: str
    password: str
    status: CameraStatus


class Cameras(CreateAndUpdateCamera):
    id: int

    class Config:
        orm_mode = True


class CreateAndUpdatePerson(BaseModel):
    person_id: int
    name: str


class Persons(CreateAndUpdatePerson):
    class Config:
        orm_mode = True
