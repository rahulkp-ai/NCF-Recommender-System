"""
production/backend/app/db/repository.py
Data-access layer — all raw DB queries live here.
Service layer calls these; routes never touch DB directly.
"""
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import or_, func

from .models import User, Movie, Interaction


# ── User ──────────────────────────────────────────────────────────────────────

class UserRepository:
    @staticmethod
    def get_by_id(db: Session, user_id: int) -> Optional[User]:
        return db.query(User).filter(User.id == user_id).first()

    @staticmethod
    def get_by_username(db: Session, username: str) -> Optional[User]:
        return db.query(User).filter(User.username == username).first()

    @staticmethod
    def get_by_email(db: Session, email: str) -> Optional[User]:
        return db.query(User).filter(User.email == email).first()

    @staticmethod
    def create(db: Session, **kwargs) -> User:
        user = User(**kwargs)
        db.add(user)
        db.commit()
        db.refresh(user)
        return user


# ── Movie ─────────────────────────────────────────────────────────────────────

class MovieRepository:
    @staticmethod
    def get_by_id(db: Session, movie_id: int) -> Optional[Movie]:
        return db.query(Movie).filter(Movie.id == movie_id).first()

    @staticmethod
    def search(db: Session, query: str, limit: int = 20) -> List[Movie]:
        q = f"%{query}%"
        return (
            db.query(Movie)
            .filter(
                or_(
                    Movie.title.ilike(q),
                    Movie.genres.ilike(q),
                )
            )
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_popular(db: Session, limit: int = 20) -> List[Movie]:
        """Movies with the most interactions."""
        return (
            db.query(Movie)
            .join(Interaction, Movie.id == Interaction.movie_id, isouter=True)
            .group_by(Movie.id)
            .order_by(func.count(Interaction.id).desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_many(db: Session, ids: List[int]) -> List[Movie]:
        return db.query(Movie).filter(Movie.id.in_(ids)).all()

    @staticmethod
    def get_trending(db: Session, limit: int = 20) -> List[Movie]:
        """Recently-seeded movies with posters (used for hero/trending rows)."""
        return (
            db.query(Movie)
            .filter(Movie.poster_url != None)
            .order_by(Movie.year.desc().nullslast())
            .limit(limit)
            .all()
        )


# ── Interaction ───────────────────────────────────────────────────────────────

class InteractionRepository:
    @staticmethod
    def create(db: Session, **kwargs) -> Interaction:
        obj = Interaction(**kwargs)
        db.add(obj)
        db.commit()
        db.refresh(obj)
        return obj

    @staticmethod
    def get_user_movie_ids(db: Session, user_id: int) -> List[int]:
        rows = (
            db.query(Interaction.movie_id)
            .filter(Interaction.user_id == user_id)
            .all()
        )
        return [r.movie_id for r in rows]

    @staticmethod
    def count_for_user(db: Session, user_id: int) -> int:
        return (
            db.query(Interaction)
            .filter(Interaction.user_id == user_id)
            .count()
        )
