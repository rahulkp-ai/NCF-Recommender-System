"""
production/tests/integration/test_error_handling.py

New in Phase 5. Exercises the AppError -> JSONResponse handler registered
in production/backend/app/main.py end-to-end, using the same DB-backed
TestClient convention as production/tests/unit/test_api.py (this suite
also expects a live Postgres via docker-compose — it is not a pure unit
test despite living alongside one in the moved tests/ directory; noted
here since Phase 4's move didn't reclassify pre-existing tests).

Run: pytest production/tests/integration/test_error_handling.py -v
(requires `docker compose up db` or an equivalent local Postgres —
same requirement as production/tests/unit/test_api.py)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client():
    with (
        patch("production.backend.app.db.seed.run_seed", return_value=None),
        patch(
            "production.backend.app.services.recommendation_service.RecommendationService.load"
        ) as mock_load,
    ):
        mock_load.return_value = MagicMock()
        from production.backend.app.main import app

        with TestClient(app) as c:
            yield c


def test_movie_not_found_returns_structured_error(client):
    """GET a movie ID that can't exist (negative) and confirm the
    NotFoundError -> AppError handler produces the expected JSON shape,
    not FastAPI's default {"detail": "..."} shape."""
    r = client.get("/api/v1/recommend/movie/-1")
    assert r.status_code == 404
    body = r.json()
    assert body["error"] == "not_found"
    assert "Movie" in body["message"]
    assert body["details"]["resource"] == "Movie"
    assert body["details"]["id"] == "-1"


def test_unhandled_validation_still_returns_422(client):
    """Confirm the new AppError handler doesn't interfere with FastAPI's
    own request-validation errors (a different code path entirely)."""
    r = client.get("/api/v1/recommend/homepage?k=not_a_number")
    assert r.status_code == 422
