"""
production/tests/unit/test_ml.py
Unit tests for the hybrid recommendation engine components.
Run: pytest production/tests/unit/test_ml.py -v

Moved from tests/test_ml.py in Phase 4. sys.path insert updated from
parents[1] to parents[3] for the new depth.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import pytest
import numpy as np
import pandas as pd


# ── PopularityEngine ──────────────────────────────────────────────────────────

class TestPopularityEngine:
    def setup_method(self):
        from production.recommenders.popularity.engine import PopularityEngine
        interactions = pd.DataFrame({
            "item_id": [1, 1, 1, 2, 2, 3, 4, 4, 4, 4],
        })
        self.engine = PopularityEngine().fit(interactions)

    def test_scores_in_range(self):
        for item_id in [1, 2, 3, 4]:
            s = self.engine.score(item_id)
            assert 0.0 <= s <= 1.0

    def test_most_popular_scores_highest(self):
        # item 4 has 4 interactions — should score highest
        assert self.engine.score(4) >= self.engine.score(1)
        assert self.engine.score(1) >= self.engine.score(2)

    def test_unseen_item_scores_zero(self):
        assert self.engine.score(9999) == 0.0

    def test_top_k_returns_k_items(self):
        top = self.engine.top_k(3)
        assert len(top) == 3

    def test_top_k_excludes_seen(self):
        top = self.engine.top_k(10, exclude={1, 2, 3, 4})
        assert len(top) == 0

    def test_top_k_respects_exclude(self):
        top = self.engine.top_k(10, exclude={4})
        assert 4 not in top


# ── ContentEngine ─────────────────────────────────────────────────────────────

class TestContentEngine:
    def setup_method(self):
        from production.recommenders.content.engine import ContentEngine
        movies = pd.DataFrame({
            "item_id": [1, 2, 3],
            "genres":  ["Action|Drama", "Comedy|Romance", "Action|Comedy"],
        })
        self.engine = ContentEngine().fit(movies)

    def test_user_profile_none_for_empty(self):
        profile = self.engine.user_profile([])
        assert profile is None

    def test_user_profile_none_for_unknown(self):
        profile = self.engine.user_profile([9999])
        assert profile is None

    def test_user_profile_shape(self):
        profile = self.engine.user_profile([1])
        assert profile is not None
        assert len(profile) == 18  # 18 genres

    def test_cosine_similarity_same_genre(self):
        # Items 1 and 3 both have Action — profile from item 1 should score item 3 higher than item 2
        profile = self.engine.user_profile([1])
        s1 = self.engine.score(profile, 1)
        s2 = self.engine.score(profile, 2)
        s3 = self.engine.score(profile, 3)
        assert s3 > s2  # item 3 shares Action with item 1

    def test_score_unknown_item(self):
        profile = self.engine.user_profile([1])
        assert self.engine.score(profile, 9999) == 0.0


# ── HybridEngine alpha schedule ───────────────────────────────────────────────

class TestAlphaSchedule:
    def test_zero_interactions(self):
        from production.recommenders.hybrid.engine import HybridEngine
        assert HybridEngine.alpha(0) == 0.0

    def test_half_warmup(self):
        from production.recommenders.hybrid.engine import HybridEngine
        assert HybridEngine.alpha(10) == pytest.approx(0.5)

    def test_full_warmup(self):
        from production.recommenders.hybrid.engine import HybridEngine
        assert HybridEngine.alpha(20) == pytest.approx(1.0)

    def test_capped_at_one(self):
        from production.recommenders.hybrid.engine import HybridEngine
        assert HybridEngine.alpha(999) == 1.0


# ── Security helpers ──────────────────────────────────────────────────────────

class TestSecurity:
    def test_hash_and_verify(self):
        from backend.app.core.security import hash_password, verify_password
        hashed = hash_password("mysecret")
        assert verify_password("mysecret", hashed)
        assert not verify_password("wrongpass", hashed)

    def test_jwt_roundtrip(self):
        from backend.app.core.security import create_access_token, decode_access_token
        token = create_access_token({"sub": "42"})
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "42"

    def test_invalid_jwt(self):
        from backend.app.core.security import decode_access_token
        assert decode_access_token("not.a.valid.token") is None
