import os

from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker


def obter_uri_do_banco_de_dados(eh_teste: bool = False) -> str:
    ambiente_de_teste = eh_teste or os.getenv("TEST_ENV", False)
    em_docker = os.getenv("IN_DOCKER", "false").lower() == "true"

    print("ambiente_de_teste", ambiente_de_teste)
    print("em_docker", em_docker)

    host = "reconhecimento_facial_db" if em_docker else "localhost"
    port = 3306 if any([ambiente_de_teste, em_docker]) else 33061
    password = "password"
    user = "root"
    schema = "iftm"
    database_uri = f"mysql+pymysql://{user}:{password}@{host}:{port}/{schema}"

    return database_uri


DATABASE_URL = obter_uri_do_banco_de_dados()

db_engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=db_engine)

Base = declarative_base()


def get_db():
    """
    Function to generate db session
    :return: Session
    """
    db = None
    try:
        db = SessionLocal()
        yield db
    finally:
        db.close()
