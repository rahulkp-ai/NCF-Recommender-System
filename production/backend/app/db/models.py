"""
production/backend/app/db/models.py
SQLAlchemy ORM models — single source of truth for the DB schema.
"""
from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, func
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    username      = Column(String(50), unique=True, nullable=False)
    email         = Column(String(100), unique=True, nullable=False)
    password_hash = Column(String(200), nullable=False)
    age_group     = Column(String(10))
    gender        = Column(String(1))
    occupation    = Column(Integer)
    created_at    = Column(DateTime, server_default=func.now())

    interactions  = relationship("Interaction", back_populates="user")


class Movie(Base):
    __tablename__ = "movies"

    id         = Column(Integer, primary_key=True, index=True)
    title      = Column(String(200), nullable=False)
    genres     = Column(String(200))
    tmdb_id    = Column(Integer, nullable=True)
    poster_url = Column(String(500), nullable=True)
    year       = Column(Integer, nullable=True)

    interactions = relationship("Interaction", back_populates="movie")


class Interaction(Base):
    __tablename__ = "interactions"

    id         = Column(Integer, primary_key=True, index=True)
    user_id    = Column(Integer, ForeignKey("users.id"), nullable=False, index=True)
    movie_id   = Column(Integer, ForeignKey("movies.id"), nullable=False, index=True)
    event_type = Column(String(20), nullable=False)
    rating     = Column(Float, nullable=True)
    created_at = Column(DateTime, server_default=func.now())

    user  = relationship("User",  back_populates="interactions")
    movie = relationship("Movie", back_populates="interactions")
