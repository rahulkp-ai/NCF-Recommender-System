"""
production/recommenders/content/engine.py
Content-based cold-start engine.

Split out of ml/models/ncf/cold_start.py in Phase 4 (see
docs/decisions/0003-recommenders-serving-rename.md). ContentEngine
logic below is unchanged, verbatim from the original file.
"""

import numpy as np
import pandas as pd


class ContentEngine:
    """
    Content-based scoring via binary genre vectors + cosine similarity.
    """

    GENRES = [
        "Action",
        "Adventure",
        "Animation",
        "Children's",
        "Comedy",
        "Crime",
        "Documentary",
        "Drama",
        "Fantasy",
        "Film-Noir",
        "Horror",
        "Musical",
        "Mystery",
        "Romance",
        "Sci-Fi",
        "Thriller",
        "War",
        "Western",
    ]

    def __init__(self):
        self._item_vectors: dict[int, np.ndarray] = {}
        self._n_genres = len(self.GENRES)

    def fit(self, movies_df: pd.DataFrame) -> "ContentEngine":
        id_col = "item_id" if "item_id" in movies_df.columns else "id"
        for _, row in movies_df.iterrows():
            vec = self._parse_genres(str(row.genres))
            self._item_vectors[int(row[id_col])] = vec
        return self

    def _parse_genres(self, genre_str: str) -> np.ndarray:
        vec = np.zeros(self._n_genres, dtype=np.float32)
        for g in genre_str.split("|"):
            g = g.strip()
            if g in self.GENRES:
                vec[self.GENRES.index(g)] = 1.0
        return vec

    def user_profile(self, seen_item_ids: list[int]) -> np.ndarray | None:
        vecs = [self._item_vectors[i] for i in seen_item_ids if i in self._item_vectors]
        if not vecs:
            return None
        return np.mean(vecs, axis=0)

    def score(self, user_profile: np.ndarray, item_id: int) -> float:
        if item_id not in self._item_vectors:
            return 0.0
        iv = self._item_vectors[item_id]
        num = float(np.dot(user_profile, iv))
        denom = float(np.linalg.norm(user_profile) * np.linalg.norm(iv))
        return num / denom if denom > 1e-9 else 0.0

    def top_k(self, user_profile: np.ndarray, k: int, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        scores = {
            iid: self.score(user_profile, iid) for iid in self._item_vectors if iid not in exclude
        }
        return sorted(scores, key=scores.get, reverse=True)[:k]
