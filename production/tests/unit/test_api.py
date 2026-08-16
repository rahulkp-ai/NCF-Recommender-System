"""
production/tests/unit/test_api.py
Integration tests for the FastAPI backend.
Run: pytest production/tests/unit/test_api.py -v

Moved from tests/test_api.py in Phase 4. This file is now one directory
deeper (production/tests/unit/ vs. tests/), so the sys.path insert below
was updated from parents[1] to parents[3] to still resolve to repo root
— same class of fix as the two flagged sys.path bugs in main.py
(see docs/decisions/0003-recommenders-serving-rename.md).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from production.backend.app.db.connection import create_tables

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def client():
    """Create test client with mocked DB and rec service."""
    # run_seed() is mocked below (slow synthetic data generation, not needed
    # for these tests), but that also skips its internal create_tables() call
    # — so we create the schema explicitly here against a real test database.
    create_tables()
    # Mock the heavy startup (DB seed + model load)
    with (
        patch("production.backend.app.db.seed.run_seed", return_value=None),
        patch(
            "production.backend.app.services.recommendation_service.RecommendationService.load"
        ) as mock_load,
    ):
        mock_svc = MagicMock()
        mock_svc.recommend_homepage.return_value = {
            "strategy": "cold_start",
            "recommendations": [
                {
                    "item_id": 1,
                    "score": 0.9,
                    "source": "cold_start",
                    "alpha": 0.0,
                    "title": "Toy Story (1995)",
                    "genres": "Animation|Comedy",
                    "poster_url": None,
                    "year": 1995,
                }
            ],
        }
        mock_svc.recommend_for_user.return_value = {
            "user_id": 1,
            "n_interactions": 5,
            "alpha": 0.25,
            "strategy": "blend",
            "recommendations": [
                {
                    "item_id": 2,
                    "score": 0.8,
                    "source": "blend",
                    "alpha": 0.25,
                    "title": "GoodFellas (1990)",
                    "genres": "Crime|Drama",
                    "poster_url": None,
                    "year": 1990,
                }
            ],
        }
        mock_load.return_value = mock_svc

        from production.backend.app.main import app

        with TestClient(app) as c:
            yield c


# ── Health ────────────────────────────────────────────────────────────────────


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"


def test_root(client):
    r = client.get("/")
    assert r.status_code == 200


# ── Auth ──────────────────────────────────────────────────────────────────────


def test_register_and_login(client):
    import random
    import string

    suffix = "".join(random.choices(string.ascii_lowercase, k=6))
    payload = {
        "username": f"test_{suffix}",
        "email": f"test_{suffix}@example.com",
        "password": "testpass123",
    }
    # Register
    r = client.post("/api/v1/auth/register", json=payload)
    assert r.status_code == 201, r.text
    data = r.json()
    assert "access_token" in data
    assert data["user"]["username"] == payload["username"]

    # Login
    r2 = client.post(
        "/api/v1/auth/login",
        json={
            "username": payload["username"],
            "password": payload["password"],
        },
    )
    assert r2.status_code == 200
    assert "access_token" in r2.json()


def test_login_wrong_password(client):
    r = client.post(
        "/api/v1/auth/login",
        json={
            "username": "user_0001",
            "password": "wrongpassword",
        },
    )
    assert r.status_code == 401


def test_profile_requires_auth(client):
    r = client.get("/api/v1/users/profile")
    assert r.status_code == 401


# ── Recommendations ───────────────────────────────────────────────────────────


def test_homepage_recommendations(client):
    r = client.get("/api/v1/recommend/homepage")
    assert r.status_code == 200
    data = r.json()
    assert "recommendations" in data
    assert len(data["recommendations"]) > 0


def test_search(client):
    r = client.get("/api/v1/search?q=toy")
    assert r.status_code == 200
    data = r.json()
    assert "results" in data
    assert "total" in data
    assert data["query"] == "toy"


def test_search_empty_query(client):
    r = client.get("/api/v1/search?q=")
    assert r.status_code == 422  # Validation error — q too short


# ── Interaction ───────────────────────────────────────────────────────────────


def test_interact_requires_auth(client):
    r = client.post("/api/v1/interact", json={"movie_id": 1, "event_type": "like"})
    assert r.status_code == 401
