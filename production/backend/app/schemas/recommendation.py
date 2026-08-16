"""backend/app/schemas/recommendation.py"""

from pydantic import BaseModel


class RecommendedMovie(BaseModel):
    item_id: int
    score: float
    source: str
    alpha: float
    title: str | None = None
    genres: str | None = None
    poster_url: str | None = None
    year: int | None = None


class RecommendationResponse(BaseModel):
    user_id: int
    n_interactions: int
    alpha: float
    strategy: str
    recommendations: list[RecommendedMovie]


class HomepageResponse(BaseModel):
    strategy: str
    recommendations: list[RecommendedMovie]
