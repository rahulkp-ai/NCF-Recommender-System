"""
production/backend/app/api/v1/recommend.py
GET /api/v1/recommend/{user_id}   — personalised recommendations
GET /api/v1/recommend/homepage    — cold-start / anonymous
GET /api/v1/recommend/trending    — trending movies (with posters)
GET /api/v1/recommend/popular     — most-interacted movies
"""

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from production.shared.exceptions.errors import NotFoundError

from ...db.models import User
from ...db.repository import MovieRepository
from ...db.session import get_db
from ...schemas.movie import MovieOut
from ...schemas.recommendation import HomepageResponse, RecommendationResponse
from ..deps import get_current_user

router = APIRouter(prefix="/api/v1/recommend", tags=["recommend"])


def _rec_service(request: Request):
    return request.app.state.rec_service


@router.get("/homepage", response_model=HomepageResponse)
def recommend_homepage(
    k: int = Query(20, ge=1, le=50),
    request: Request = ...,
    db: Session = Depends(get_db),
):
    return _rec_service(request).recommend_homepage(db, k=k)


@router.get("/trending")
def trending(db: Session = Depends(get_db)):
    return [MovieOut.model_validate(m) for m in MovieRepository.get_trending(db, limit=20)]


@router.get("/popular")
def popular(db: Session = Depends(get_db)):
    return [MovieOut.model_validate(m) for m in MovieRepository.get_popular(db, limit=20)]


@router.get("/movie/{movie_id}", response_model=MovieOut)
def get_movie(movie_id: int, db: Session = Depends(get_db)):
    movie = MovieRepository.get_by_id(db, movie_id)
    if not movie:
        raise NotFoundError("Movie", movie_id)
    return movie


@router.get("/similar/{movie_id}")
def similar_movies(
    movie_id: int,
    limit: int = Query(12, ge=1, le=30),
    db: Session = Depends(get_db),
):
    """Return movies sharing at least one genre with the given movie."""
    movie = MovieRepository.get_by_id(db, movie_id)
    if not movie or not movie.genres:
        return []
    # Extract genres and search for each one, collect unique results
    genres = [g.strip() for g in movie.genres.split("|")]
    seen_ids = {movie_id}
    results = []
    for genre in genres:
        if len(results) >= limit:
            break
        matches = MovieRepository.search(db, genre, limit=limit)
        for m in matches:
            if m.id not in seen_ids and m.poster_url:
                seen_ids.add(m.id)
                results.append(MovieOut.model_validate(m))
    return results[:limit]


@router.get("/{user_id}", response_model=RecommendationResponse)
def recommend_for_user(
    user_id: int,
    k: int = Query(10, ge=1, le=50),
    request: Request = ...,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return _rec_service(request).recommend_for_user(user_id, db, k=k)
