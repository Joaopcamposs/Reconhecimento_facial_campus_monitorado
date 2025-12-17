from fastapi import FastAPI

from src.infra.database import init_db
from src.api import routes

app = FastAPI()

# Initialize database tables using SQLAlchemy
init_db()

# include routes from api
app.include_router(routes.app)
