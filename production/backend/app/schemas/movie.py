"""backend/app/schemas/movie.py"""
from typing import Optional, List
from pydantic import BaseModel


class MovieOut(BaseModel):
    id: int
    title: str
    genres: Optional[str] = None
    poster_url: Optional[str] = None
    year: Optional[int] = None

    model_config = {"from_attributes": True}


class InteractionCreate(BaseModel):
    movie_id: int
    event_type: str   # "like" | "click" | "rate"
    rating: Optional[float] = None


class InteractionOut(BaseModel):
    id: int
    user_id: int
    movie_id: int
    event_type: str
    rating: Optional[float] = None

    model_config = {"from_attributes": True}


class SearchResponse(BaseModel):
    query: str
    results: List[MovieOut]
    total: int
