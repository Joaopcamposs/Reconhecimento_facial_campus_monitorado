from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# DATABASE_URL = "mysql+pymysql://root:@localhost:3307/iftm"  # local
DATABASE_URL = "mysql+pymysql://root:password@reconhecimento_facial_db/iftm"  # docker

# root is the mysql user
# password is mysql password
# reconhecimento_facial_db is the connection
# iftm is the database (schema)

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
