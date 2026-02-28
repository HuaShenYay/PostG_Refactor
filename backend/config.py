import os

from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Config:
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL", "mysql+pymysql://root:123456@localhost/poetry_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    JSON_AS_ASCII = False
