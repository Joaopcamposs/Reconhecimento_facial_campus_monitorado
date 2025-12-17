from pydantic import BaseModel
from models import CameraStatus


# TO support creation and update APIs
class CreateAndUpdateCamera(BaseModel):
    user: str
    camera_ip: str
    password: str
    status: CameraStatus


# TO support list and get APIs
class Cameras(CreateAndUpdateCamera):
    id: int

    class Config:
        orm_mode = True


# TO support creation and update APIs
class CreateAndUpdatePerson(BaseModel):
    person_id: int
    name: str


# TO support list and get APIs
class Persons(CreateAndUpdatePerson):
    class Config:
        orm_mode = True
