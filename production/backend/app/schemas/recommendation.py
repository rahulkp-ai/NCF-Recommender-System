"""backend/app/schemas/recommendation.py"""
from typing import Optional, List
from pydantic import BaseModel


class RecommendedMovie(BaseModel):
    item_id: int
    score: float
    source: str
    alpha: float
    title: Optional[str] = None
    genres: Optional[str] = None
    poster_url: Optional[str] = None
    year: Optional[int] = None


class RecommendationResponse(BaseModel):
    user_id: int
    n_interactions: int
    alpha: float
    strategy: str
    recommendations: List[RecommendedMovie]


class HomepageResponse(BaseModel):
    strategy: str
    recommendations: List[RecommendedMovie]
