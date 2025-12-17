from sqlalchemy.orm import Session

from src.entities.models import Controller


class ControllerNotFound(Exception):
    pass


def get_controller_by_id(session: Session, _id: int) -> Controller:
    """Get controller by ID."""
    controller: Controller | None = session.query(Controller).get(_id)

    if controller is None:
        raise ControllerNotFound(f"Controller with id {_id} not found")

    return controller


def set_capture_flag(session: Session, _id: int) -> Controller:
    """Set capture flag to 1."""
    controller: Controller = get_controller_by_id(session, _id)
    controller.save_picture = 1

    session.commit()
    session.refresh(controller)

    return controller


def reset_capture_flag(session: Session, _id: int) -> Controller:
    """Reset capture flag to 0."""
    controller: Controller = get_controller_by_id(session, _id)
    controller.save_picture = 0

    session.commit()
    session.refresh(controller)

    return controller
