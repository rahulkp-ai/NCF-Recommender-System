"""backend/app/services/interaction_service.py"""

from fastapi import HTTPException, status
from sqlalchemy.orm import Session

from ..db.repository import InteractionRepository, MovieRepository
from ..schemas.movie import InteractionCreate, InteractionOut


class InteractionService:
    @staticmethod
    def record(db: Session, user_id: int, payload: InteractionCreate) -> InteractionOut:
        movie = MovieRepository.get_by_id(db, payload.movie_id)
        if not movie:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Movie not found")
        obj = InteractionRepository.create(
            db,
            user_id=user_id,
            movie_id=payload.movie_id,
            event_type=payload.event_type,
            rating=payload.rating,
        )
        return InteractionOut.model_validate(obj)
