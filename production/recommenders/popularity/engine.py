"""
production/recommenders/popularity/engine.py
Popularity-based cold-start engine.

Split out of ml/models/ncf/cold_start.py in Phase 4 (see
docs/decisions/0003-recommenders-serving-rename.md). PopularityEngine
logic below is unchanged, verbatim from the original file.
"""
import numpy as np
import pandas as pd


class PopularityEngine:
    """
    Scores items by log-normalised global interaction frequency.
    score(i) = log(1 + count(i)) / log(1 + max_count)
    """

    def __init__(self):
        self._scores: dict[int, float] = {}
        self._sorted: list[int] = []

    def fit(self, interactions_df: pd.DataFrame) -> "PopularityEngine":
        id_col = "item_id" if "item_id" in interactions_df.columns else "movie_id"
        counts = interactions_df[id_col].value_counts()
        max_count = float(counts.max())
        self._scores = {
            int(iid): float(np.log1p(cnt) / np.log1p(max_count))
            for iid, cnt in counts.items()
        }
        self._sorted = sorted(self._scores, key=self._scores.get, reverse=True)
        return self

    def score(self, item_id: int) -> float:
        return self._scores.get(item_id, 0.0)

    def top_k(self, k: int, exclude: set[int] | None = None) -> list[int]:
        exclude = exclude or set()
        result = []
        for iid in self._sorted:
            if iid not in exclude:
                result.append(iid)
            if len(result) >= k:
                break
        return result

