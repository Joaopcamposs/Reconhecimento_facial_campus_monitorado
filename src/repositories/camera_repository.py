from sqlalchemy.orm import Session

from src.entities.models import Camera
from src.entities.schemas import CreateAndUpdateCamera


class CameraNotFound(Exception):
    pass


def get_all_cameras(session: Session) -> list[Camera]:
    """Get list of all cameras."""
    return session.query(Camera).all()


def get_camera_by_id(session: Session, _id: int) -> Camera | None:
    """Get camera by ID."""
    camera: Camera | None = session.query(Camera).get(_id)
    return camera


def create_camera(session: Session, camera_info: CreateAndUpdateCamera) -> Camera:
    """Add a new camera to the database."""
    new_camera: Camera = Camera(**camera_info.dict())
    session.add(new_camera)
    session.commit()
    session.refresh(new_camera)
    return new_camera


def update_camera(
    session: Session, _id: int, info_update: CreateAndUpdateCamera
) -> Camera:
    """Update camera details."""
    camera: Camera | None = get_camera_by_id(session, _id)

    if camera is None:
        raise CameraNotFound(f"Camera with id {_id} not found")

    camera.camera_ip = info_update.camera_ip
    camera.user = info_update.user
    camera.status = info_update.status
    camera.password = info_update.password
    session.commit()
    session.refresh(camera)

    return camera


def remove_camera(session: Session, _id: int) -> None:
    """Delete a camera from the database."""
    camera_info: Camera | None = get_camera_by_id(session, _id)

    if camera_info is None:
        raise CameraNotFound(f"Camera with id {_id} not found")

    session.delete(camera_info)
    session.commit()
