"""
research/hybrid/cold_start.py

Cold-start and popularity-based recommendation strategies.
Used when a user has too few interactions for NCF to be reliable.

Two sub-strategies:
  1. Popularity-based  — recommend globally trending items
  2. Content-based     — recommend items similar to what the user has seen,
                         using genre vectors as a lightweight feature
"""

import numpy as np
import pandas as pd
from pathlib import Path
from functools import lru_cache


# ── Popularity engine ─────────────────────────────────────────────────────────

class PopularityEngine:
    """
    Scores items by their global interaction frequency.

    score_pop(i) = log(1 + count(i)) / log(1 + max_count)

    Log-scaling prevents blockbusters from completely dominating.
    All scores are normalised to [0, 1].
    """

    def __init__(self):
        self._scores: dict[int, float] = {}
        self._sorted: list[int] = []

    def fit(self, interactions_df: pd.DataFrame) -> "PopularityEngine":
        """
        Compute popularity scores from an interactions DataFrame.

        Parameters
        ----------
        interactions_df : DataFrame with column 'item_id' (or 'movie_id')
        """
        id_col = "item_id" if "item_id" in interactions_df.columns else "movie_id"
        counts = interactions_df[id_col].value_counts()

        max_count = float(counts.max())
        self._scores = {
            int(item_id): float(np.log1p(cnt) / np.log1p(max_count))
            for item_id, cnt in counts.items()
        }
        # Pre-sort for fast top-K queries
        self._sorted = sorted(self._scores, key=self._scores.get, reverse=True)
        print(f"PopularityEngine: scored {len(self._scores):,} items")
        return self

    def score(self, item_id: int) -> float:
        return self._scores.get(item_id, 0.0)

    def top_k(self, k: int, exclude: set[int] | None = None) -> list[int]:
        """Return top-K item IDs, excluding already-seen items."""
        exclude = exclude or set()
        result = []
        for item_id in self._sorted:
            if item_id not in exclude:
                result.append(item_id)
            if len(result) >= k:
                break
        return result


# ── Content / genre engine ────────────────────────────────────────────────────

class ContentEngine:
    """
    Content-based scoring using TF-IDF genre vectors.

    Each movie is represented as a binary genre vector.
    A user's taste profile is the mean of their watched movies' genre vectors.
    Similarity is cosine distance between user profile and candidate item.

    Genres in MovieLens 1M (18 genres + 'unknown'):
    Action, Adventure, Animation, Children's, Comedy, Crime, Documentary,
    Drama, Fantasy, Film-Noir, Horror, Musical, Mystery, Romance,
    Sci-Fi, Thriller, War, Western
    """

    GENRES = [
        "Action", "Adventure", "Animation", "Children's", "Comedy",
        "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
        "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
        "Thriller", "War", "Western",
    ]

    def __init__(self):
        self._item_vectors: dict[int, np.ndarray] = {}
        self._n_genres = len(self.GENRES)

    def fit(self, movies_df: pd.DataFrame) -> "ContentEngine":
        """
        Build genre vectors for all items.

        Parameters
        ----------
        movies_df : DataFrame with columns 'item_id' (or 'id') and 'genres'
                    genres is a pipe-separated string: "Action|Drama"
        """
        id_col = "item_id" if "item_id" in movies_df.columns else "id"
        for _, row in movies_df.iterrows():
            vec = self._parse_genres(str(row.genres))
            self._item_vectors[int(row[id_col])] = vec
        print(f"ContentEngine: built vectors for {len(self._item_vectors):,} items")
        return self

    def _parse_genres(self, genre_str: str) -> np.ndarray:
        """Convert 'Action|Drama|Thriller' → binary ndarray of length n_genres."""
        vec = np.zeros(self._n_genres, dtype=np.float32)
        for g in genre_str.split("|"):
            g = g.strip()
            if g in self.GENRES:
                vec[self.GENRES.index(g)] = 1.0
        return vec

    def user_profile(self, seen_item_ids: list[int]) -> np.ndarray | None:
        """
        Build a user taste profile as the mean genre vector of seen items.
        Returns None if no known items are in seen_item_ids.
        """
        vecs = [self._item_vectors[i] for i in seen_item_ids
                if i in self._item_vectors]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    def score(self, user_profile: np.ndarray, item_id: int) -> float:
        """Cosine similarity between user profile and item genre vector."""
        if item_id not in self._item_vectors:
            return 0.0
        item_vec = self._item_vectors[item_id]
        num = float(np.dot(user_profile, item_vec))
        denom = float(np.linalg.norm(user_profile) * np.linalg.norm(item_vec))
        return num / denom if denom > 1e-9 else 0.0

    def top_k(self, user_profile: np.ndarray, k: int,
              exclude: set[int] | None = None) -> list[int]:
        """Return top-K items by cosine similarity, excluding seen items."""
        exclude = exclude or set()
        scores = {
            iid: self.score(user_profile, iid)
            for iid in self._item_vectors
            if iid not in exclude
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]