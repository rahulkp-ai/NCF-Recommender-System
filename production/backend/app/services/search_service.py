"""backend/app/services/search_service.py"""

from sqlalchemy.orm import Session

from production.observability.metrics import RECOMMENDATION_STRATEGY

from ..db.repository import MovieRepository
from ..schemas.movie import MovieOut, SearchResponse


class SearchService:
    @staticmethod
    def search(db: Session, query: str, limit: int = 20) -> SearchResponse:
        movies = MovieRepository.search(db, query, limit=limit)
        RECOMMENDATION_STRATEGY.labels("backend", "search").inc()
        return SearchResponse(
            query=query,
            results=[MovieOut.model_validate(m) for m in movies],
            total=len(movies),
        )
