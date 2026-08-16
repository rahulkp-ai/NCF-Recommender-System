"""
production/backend/app/db/session.py
FastAPI dependency that yields a DB session and closes it after request.
"""

from collections.abc import Generator

from .connection import SessionLocal


def get_db() -> Generator:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
