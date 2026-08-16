"""
research/hybrid/search.py

Title search with hybrid recommendation fallback.
"""

from sqlalchemy.orm import Session

from .hybrid_engine import HybridEngine


def search_and_recommend(
    query: str,
    user_id: int,
    seen_items: list[int],
    engine: HybridEngine,
    db: Session,
    k: int = 10,
) -> dict:
    """
    Search for movies by title. Returns matched movies plus recommendations.

    Branch A — title found:
      Return matched movies + hybrid recommendations ("you may also like")

    Branch B — no title match:
      Extract keywords, attempt genre-tag match, return content-based fallback
    """
    from backend.app.db.models import Movie

    # ── Search DB ─────────────────────────────────────────────────────────────
    matches = db.query(Movie).filter(Movie.title.ilike(f"%{query}%")).limit(5).all()

    if matches:
        # Branch A: found — add hybrid "you may also like"
        match_ids = [m.id for m in matches]
        recommendations = engine.recommend(user_id=user_id, seen_items=seen_items + match_ids, k=k)
        return {
            "branch": "found",
            "query": query,
            "matched_movies": [{"id": m.id, "title": m.title, "genres": m.genres} for m in matches],
            "recommendations": recommendations,
        }
    else:
        # Branch B: not found — content-based fallback
        # Attempt to infer a relevant item from query keywords
        all_movies = db.query(Movie).all()
        # Score every movie by how many query words appear in its title/genres
        tokens = set(query.lower().split())
        scored = []
        for movie in all_movies:
            text = (movie.title + " " + (movie.genres or "")).lower()
            overlap = sum(1 for t in tokens if t in text)
            if overlap > 0:
                scored.append((movie.id, overlap))
        scored.sort(key=lambda x: -x[1])

        fallback_recs = []
        if scored:
            seed_item_id = scored[0][0]
            fallback_recs = engine.recommend_similar(seed_item_id, k=k)

        if not fallback_recs:
            # Last resort: pure popularity
            fallback_recs = [
                {"item_id": iid, "score": engine.pop.score(iid), "source": "popularity_fallback"}
                for iid in engine.pop.top_k(k)
            ]

        return {
            "branch": "not_found",
            "query": query,
            "matched_movies": [],
            "recommendations": fallback_recs,
        }
