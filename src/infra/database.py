import os
from collections.abc import Generator
from typing import Any

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

Base = declarative_base()


def obter_uri_do_banco_de_dados(eh_teste: bool = False) -> str:
    """Build database URI based on environment."""
    ambiente_de_teste: bool = eh_teste or bool(os.getenv("TEST_ENV", False))
    em_docker: bool = os.getenv("IN_DOCKER", "false").lower() == "true"

    host: str = "reconhecimento_facial_db" if em_docker else "localhost"
    port: int = 3306 if any([ambiente_de_teste, em_docker]) else 33061
    password: str = "password"
    user: str = "root"
    schema: str = "iftm"
    database_uri: str = f"mysql+pymysql://{user}:{password}@{host}:{port}/{schema}"

    return database_uri


DATABASE_URL: str = obter_uri_do_banco_de_dados()

db_engine: Engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_recycle=3600)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)


def init_db() -> None:
    """Initialize database tables using SQLAlchemy models."""
    from src.entities.models import Camera, Controller, Person  # noqa: F401

    Base.metadata.create_all(bind=db_engine)

    # Create default controller if not exists
    db: Session = SessionLocal()
    try:
        from src.entities.models import Controller

        controller: Controller | None = (
            db.query(Controller).filter(Controller.capture_id == 1).first()
        )
        if not controller:
            controller = Controller(capture_id=1, save_picture=0)
            db.add(controller)
            db.commit()
    except Exception as e:
        print(f"Error initializing controller: {e}")
    finally:
        db.close()


def get_db() -> Generator[Session, Any, None]:
    """Generate database session."""
    db: Session = SessionLocal()
    try:
        yield db
    finally:
        db.close()
