"""
production/backend/app/db/connection.py
SQLAlchemy engine, session factory, and table-creation helper.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from ..core.config import DATABASE_URL
from .models import Base

engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=10,
    max_overflow=20,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def create_tables() -> None:
    """Create all tables if they don't exist (idempotent)."""
    Base.metadata.create_all(bind=engine)
