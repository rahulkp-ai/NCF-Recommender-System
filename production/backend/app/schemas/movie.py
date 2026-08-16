"""backend/app/schemas/movie.py"""

from pydantic import BaseModel


class MovieOut(BaseModel):
    id: int
    title: str
    genres: str | None = None
    poster_url: str | None = None
    year: int | None = None

    model_config = {"from_attributes": True}


class InteractionCreate(BaseModel):
    movie_id: int
    event_type: str  # "like" | "click" | "rate"
    rating: float | None = None


class InteractionOut(BaseModel):
    id: int
    user_id: int
    movie_id: int
    event_type: str
    rating: float | None = None

    model_config = {"from_attributes": True}


class SearchResponse(BaseModel):
    query: str
    results: list[MovieOut]
    total: int
