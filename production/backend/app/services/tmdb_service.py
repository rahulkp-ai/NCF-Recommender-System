"""
production/backend/app/services/tmdb_service.py
Fetches TMDB poster URLs for movies and caches them in the database.
Run once after seeding: python -m backend.app.services.tmdb_service
"""

import re
import time

import requests
from sqlalchemy.orm import Session

from ..core.config import TMDB_API_KEY
from ..db.connection import SessionLocal
from ..db.models import Movie

TMDB_SEARCH = "https://api.themoviedb.org/3/search/movie"
TMDB_IMG = "https://image.tmdb.org/t/p/w500"


def clean_title(title: str) -> str:
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def fetch_poster(title: str, year: int | None) -> str | None:
    if not TMDB_API_KEY:
        return None
    try:
        r = requests.get(
            TMDB_SEARCH,
            params={
                "api_key": TMDB_API_KEY,
                "query": clean_title(title),
                "year": year or "",
            },
            timeout=5,
        )
        if r.status_code == 401:
            return None
        r.raise_for_status()
        results = r.json().get("results", [])
        if results and results[0].get("poster_path"):
            return TMDB_IMG + results[0]["poster_path"]
    except Exception:
        pass
    return None


def populate_posters(limit: int = 4000, sleep: float = 0.25):
    db: Session = SessionLocal()
    try:
        movies = (
            db.query(Movie)
            .filter((Movie.poster_url.is_(None)) | (Movie.poster_url == ""))
            .limit(limit)
            .all()
        )
        if not movies:
            print("All posters already populated.")
            return

        print(f"Fetching posters for {len(movies)} movies...")
        for i, movie in enumerate(movies):
            url = fetch_poster(movie.title, movie.year)
            if url:
                movie.poster_url = url
            if (i + 1) % 25 == 0:
                db.commit()
                print(f"  {i + 1}/{len(movies)} done")
            time.sleep(sleep)
        db.commit()
        print("Poster fetch complete.")
    finally:
        db.close()


if __name__ == "__main__":
    populate_posters()
