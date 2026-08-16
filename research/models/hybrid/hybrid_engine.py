"""
research/hybrid/hybrid_engine.py

The hybrid recommendation engine.

Combines:
  - NCF model (PyTorch) for warm users
  - Popularity engine for cold users
  - Content engine for genre-aware cold-start

Scoring formula:
  alpha(n) = min(1.0, n / N_WARMUP)
  score    = alpha * ncf_score + (1 - alpha) * cold_score
"""

# Example usage (run from repo root):
#   python -c "from research.hybrid.popularity import build_and_save_engines; build_and_save_engines()"

import numpy as np
import torch

from models.pytorch_ncf.model import PyTorchNCF, get_device

from .cold_start import ContentEngine, PopularityEngine
from .popularity import load_engines

# Tunable constants — put these in your thesis as hyperparameters
N_WARMUP = 20  # interactions before NCF fully trusted
TOP_K_DEFAULT = 10  # default recommendation list length
W_POPULARITY = 0.6  # weight of popularity within cold-start
W_CONTENT = 0.4  # weight of content similarity within cold-start


class HybridEngine:
    """
    Unified recommendation interface used by the FastAPI backend.

    Usage
    -----
    engine = HybridEngine.load()
    recs   = engine.recommend(user_id=42, seen_items=[1, 5, 23], k=10)
    """

    def __init__(
        self,
        ncf_model: PyTorchNCF,
        pop_engine: PopularityEngine,
        content_engine: ContentEngine,
        n_items: int,
    ):
        self.ncf = ncf_model
        self.pop = pop_engine
        self.content = content_engine
        self.n_items = n_items
        self.device = get_device()
        self.ncf.to(self.device)
        self.ncf.eval()

    # ── Factory ───────────────────────────────────────────────────────────────

    @classmethod
    def load(
        cls,
        weights_path: str = "research/evaluation/pytorch_ncf_weights.pt",
        meta_path: str = "research/datasets/processed/meta.json",
    ) -> "HybridEngine":
        """Load all components from disk and return a ready engine."""
        import json

        with open(meta_path) as f:
            meta = json.load(f)
        n_users = meta["n_users"]
        n_items = meta["n_items"]

        device = get_device()
        model = PyTorchNCF(n_users, n_items, emb_dim=32, hidden=[64, 32])
        model.load_state_dict(torch.load(weights_path, map_location=device))

        pop_engine, content_engine = load_engines()
        return cls(model, pop_engine, content_engine, n_items)

    # ── Alpha schedule ────────────────────────────────────────────────────────

    @staticmethod
    def alpha(n_interactions: int) -> float:
        """Linear warm-up: alpha = min(1, n / N_WARMUP)."""
        return min(1.0, n_interactions / N_WARMUP)

    # ── NCF scoring ───────────────────────────────────────────────────────────

    def _ncf_scores(self, user_id: int, candidate_items: list[int]) -> np.ndarray:
        """
        Score all candidate items for user_id using the NCF model.
        Returns ndarray of shape (len(candidates),) with values in [0,1].
        """
        with torch.no_grad():
            u = torch.full((len(candidate_items),), user_id, dtype=torch.long, device=self.device)
            i = torch.tensor(candidate_items, dtype=torch.long, device=self.device)
            scores = torch.sigmoid(self.ncf(u, i)).cpu().numpy()
        return scores

    # ── Cold-start scoring ────────────────────────────────────────────────────

    def _cold_scores(self, seen_items: list[int], candidate_items: list[int]) -> np.ndarray:
        """
        Score candidates using popularity + content blend.
        Returns ndarray of shape (len(candidates),) with values in [0,1].
        """
        # User taste profile from seen items
        user_profile = self.content.user_profile(seen_items)

        scores = np.zeros(len(candidate_items))
        for idx, item_id in enumerate(candidate_items):
            pop_score = self.pop.score(item_id)
            content_score = (
                self.content.score(user_profile, item_id) if user_profile is not None else 0.0
            )
            scores[idx] = W_POPULARITY * pop_score + W_CONTENT * content_score
        return scores

    # ── Main recommendation entry point ───────────────────────────────────────

    def recommend(
        self,
        user_id: int,
        seen_items: list[int],
        k: int = TOP_K_DEFAULT,
        n_candidates: int = 200,
    ) -> list[dict]:
        """
        Generate top-K recommendations for user_id.

        Parameters
        ----------
        user_id      : int — re-indexed user ID (0-based)
        seen_items   : list[int] — item IDs the user has already interacted with
        k            : int — number of recommendations to return
        n_candidates : int — candidate pool size before final ranking

        Returns
        -------
        list of dicts: [{"item_id": int, "score": float, "source": str}, ...]
        """
        n = len(seen_items)
        a = self.alpha(n)
        b = 1.0 - a
        seen = set(seen_items)

        # ── Build candidate pool ──────────────────────────────────────────────
        # Cold candidates: top popularity items not yet seen
        cold_candidates = self.pop.top_k(n_candidates, exclude=seen)

        # For warm users, add random items to ensure NCF diversity
        if a > 0 and len(cold_candidates) < n_candidates:
            rng = np.random.default_rng(42)
            extra = [
                i
                for i in rng.integers(0, self.n_items, n_candidates * 2)
                if i not in seen and i not in cold_candidates
            ]
            cold_candidates += extra[: n_candidates - len(cold_candidates)]

        candidates = cold_candidates[:n_candidates]

        # ── Score candidates ──────────────────────────────────────────────────
        ncf_scores = self._ncf_scores(user_id, candidates) if a > 0 else np.zeros(len(candidates))
        cold_scores = (
            self._cold_scores(seen_items, candidates) if b > 0 else np.zeros(len(candidates))
        )

        # Normalise each signal independently to [0, 1] before blending
        def norm(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 1e-9 else x

        final = a * norm(ncf_scores) + b * norm(cold_scores)

        # ── Rank and return top-K ─────────────────────────────────────────────
        top_idx = np.argsort(-final)[:k]
        source = "ncf" if a >= 0.5 else ("blend" if a > 0 else "cold_start")

        return [
            {
                "item_id": candidates[idx],
                "score": round(float(final[idx]), 4),
                "source": source,
                "ncf_score": round(float(ncf_scores[idx]), 4),
                "cold_score": round(float(cold_scores[idx]), 4),
                "alpha": round(a, 2),
            }
            for idx in top_idx
        ]

    # ── Search-mode recommendation ────────────────────────────────────────────

    def recommend_similar(self, item_id: int, k: int = 10) -> list[dict]:
        """
        Content-based: find items most similar to a given item.
        Used by the search fallback path when a query matches no exact title.
        """
        if item_id not in self.content._item_vectors:
            return []
        item_profile = self.content._item_vectors[item_id]
        similar = self.content.top_k(item_profile, k + 1, exclude={item_id})[:k]
        return [
            {
                "item_id": iid,
                "score": round(self.content.score(item_profile, iid), 4),
                "source": "content_similar",
            }
            for iid in similar
        ]
