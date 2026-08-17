"""
production/backend/app/services/recommendation_service.py
Wraps HybridEngine; bridges ML model and the database.
"""

from sqlalchemy.orm import Session

from production.observability.metrics import RECOMMENDATION_STRATEGY

from ..core.logging import logger
from ..db.repository import InteractionRepository, MovieRepository


class RecommendationService:
    """Loaded once at startup — thread-safe for inference."""

    def __init__(self, engine):
        self.engine = engine

    @classmethod
    def load(cls) -> "RecommendationService":
        from production.recommenders.hybrid.engine import HybridEngine

        try:
            engine = HybridEngine.load()
        except Exception:
            # Fail fast and loud at startup — a backend that comes up
            # without a working model is worse than one that crashes
            # immediately and gets restarted by the orchestrator.
            logger.exception("Failed to load HybridEngine — see traceback above")
            raise
        logger.info("HybridEngine loaded — n_items: %d", engine.n_items)
        return cls(engine)

    # ── public API ────────────────────────────────────────────────────────────

    def recommend_for_user(self, user_id: int, db: Session, k: int = 10) -> dict:
        seen_items = InteractionRepository.get_user_movie_ids(db, user_id)
        n_seen = len(seen_items)
        alpha = self.engine.alpha(n_seen)

        recs = self.engine.recommend(user_id=user_id, seen_items=seen_items, k=k)
        enriched = self._enrich(recs, db)

        strategy = "ncf" if alpha >= 0.5 else ("blend" if alpha > 0 else "cold_start")
        RECOMMENDATION_STRATEGY.labels("backend", strategy).inc()
        return {
            "user_id": user_id,
            "n_interactions": n_seen,
            "alpha": round(alpha, 3),
            "strategy": strategy,
            "recommendations": enriched,
        }

    def recommend_homepage(self, db: Session, k: int = 20) -> dict:
        recs = self.engine.recommend(user_id=0, seen_items=[], k=k)
        RECOMMENDATION_STRATEGY.labels("backend", "cold_start").inc()
        return {"strategy": "cold_start", "recommendations": self._enrich(recs, db)}

    def search_and_recommend(self, query: str, user_id: int, db: Session, k: int = 10) -> dict:
        movies = MovieRepository.search(db, query, limit=k)
        results = [
            {
                "item_id": m.id,
                "score": 1.0,
                "source": "search",
                "alpha": 0.0,
                "title": m.title,
                "genres": m.genres,
                "poster_url": m.poster_url,
                "year": m.year,
            }
            for m in movies
        ]
        RECOMMENDATION_STRATEGY.labels("backend", "search").inc()
        return {"query": query, "results": results, "total": len(results)}

    # ── enrichment ────────────────────────────────────────────────────────────

    def _enrich(self, recs: list, db: Session) -> list:
        if not recs:
            return recs
        item_ids = [r["item_id"] for r in recs]
        movies = MovieRepository.get_many(db, item_ids)
        movie_map = {m.id: m for m in movies}
        enriched = []
        for r in recs:
            movie = movie_map.get(r["item_id"])
            enriched.append(
                {
                    **r,
                    "title": movie.title if movie else "Unknown",
                    "genres": movie.genres if movie else "",
                    "poster_url": movie.poster_url if movie else None,
                    "year": movie.year if movie else None,
                }
            )
        return enriched
