from sqlalchemy.orm import Session

from src.entities.models import Person
from src.entities.schemas import CreateAndUpdatePerson


class PersonNotFound(Exception):
    pass


def get_all_persons(session: Session) -> list[Person]:
    """Get list of all persons."""
    return session.query(Person).all()


def get_person_by_id(session: Session, _id: int) -> Person:
    """Get person by ID."""
    pessoa: Person | None = session.query(Person).get(_id)

    if pessoa is None:
        raise PersonNotFound(f"Person with id {_id} not found")

    return pessoa


def create_person(session: Session, person_info: CreateAndUpdatePerson) -> Person:
    """Add a new person to the database."""
    new_person: Person = Person(**person_info.dict())
    session.add(new_person)
    session.commit()
    session.refresh(new_person)
    return new_person


def update_person(
    session: Session, _id: int, info_update: CreateAndUpdatePerson
) -> Person:
    """Update person details."""
    person: Person = get_person_by_id(session, _id)

    person.person_id = info_update.person_id
    person.name = info_update.name
    session.commit()
    session.refresh(person)

    return person


def remove_person(session: Session, _id: int) -> None:
    """Delete a person from the database."""
    person_info: Person = get_person_by_id(session, _id)
    session.delete(person_info)
    session.commit()
