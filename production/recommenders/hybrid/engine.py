"""
production/recommenders/hybrid/engine.py
Hybrid Recommendation Engine: NCF + Popularity + Content-Based.

Moved from ml/models/ncf/hybrid_engine.py in Phase 4 (see
docs/decisions/0003-recommenders-serving-rename.md). PopularityEngine and
ContentEngine were split into sibling packages (production/recommenders/
popularity/ and production/recommenders/content/) — see
production/recommenders/hybrid/_pickle_compat.py for how old pickled
artifacts (saved against the historical `hybrid.cold_start` module path)
still deserialize correctly after that split. All algorithm logic below
is unchanged from the original file.
"""

import json
import pickle
import sys
import types
from pathlib import Path

import numpy as np
import torch

from ..ncf.model import PyTorchNCF, get_device

N_WARMUP = 20
TOP_K_DEFAULT = 10
W_POPULARITY = 0.6
W_CONTENT = 0.4


def _register_pickle_shim():
    """
    The .pkl files were saved when classes lived at `hybrid.cold_start`.
    Register sys.modules aliases so pickle.load can find them at the new path.
    Called right before any pickle.load to guarantee correct ordering.
    """
    from . import _pickle_compat as _cs

    if "hybrid" not in sys.modules:
        _h = types.ModuleType("hybrid")
        _h.cold_start = _cs
        sys.modules["hybrid"] = _h
        sys.modules["hybrid.cold_start"] = _cs


class HybridEngine:
    def __init__(self, ncf_model, pop_engine, content_engine, n_items):
        self.ncf = ncf_model
        self.pop = pop_engine
        self.content = content_engine
        self.n_items = n_items
        self.device = get_device()
        self.ncf.to(self.device)
        self.ncf.eval()

    @classmethod
    def load(cls, weights_path=None, meta_path=None):
        root = Path(__file__).resolve().parents[3]
        # NOTE: parents[3] is still repo root after the Phase 4 move —
        # production/recommenders/hybrid/engine.py is the same depth from
        # root (3 levels) as the original ml/models/ncf/hybrid_engine.py
        # was. Verified, not assumed — see docs/decisions/0003.

        weights_path = (
            Path(weights_path)
            if weights_path
            else root / "production" / "artifacts" / "production_model.pt"
        )
        meta_path = (
            Path(meta_path) if meta_path else root / "production" / "artifacts" / "meta.json"
        )
        pop_path = root / "production" / "artifacts" / "pop_engine.pkl"
        content_path = root / "production" / "artifacts" / "content_engine.pkl"

        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata not found: {meta_path}")
        if not weights_path.exists():
            raise FileNotFoundError(f"Weights not found: {weights_path}")

        with open(meta_path) as f:
            meta = json.load(f)
        n_users, n_items = meta["n_users"], meta["n_items"]

        device = get_device()
        model = PyTorchNCF(n_users, n_items, emb_dim=32, hidden=[64, 32])
        model.load_state_dict(torch.load(weights_path, map_location=device))

        # Register shim immediately before unpickling
        _register_pickle_shim()

        with open(pop_path, "rb") as f:
            pop = pickle.load(f)
        with open(content_path, "rb") as f:
            content = pickle.load(f)

        return cls(model, pop, content, n_items)

    @staticmethod
    def alpha(n_interactions):
        return min(1.0, n_interactions / N_WARMUP)

    def _ncf_scores(self, user_id, candidates):
        with torch.no_grad():
            u = torch.full((len(candidates),), user_id, dtype=torch.long, device=self.device)
            i = torch.tensor(candidates, dtype=torch.long, device=self.device)
            return torch.sigmoid(self.ncf(u, i)).cpu().numpy()

    def _cold_scores(self, seen_items, candidates):
        profile = self.content.user_profile(seen_items)
        scores = np.zeros(len(candidates))
        for idx, item_id in enumerate(candidates):
            pop_s = self.pop.score(item_id)
            cont_s = self.content.score(profile, item_id) if profile is not None else 0.0
            scores[idx] = W_POPULARITY * pop_s + W_CONTENT * cont_s
        return scores

    def recommend(self, user_id, seen_items, k=TOP_K_DEFAULT, n_candidates=200):
        seen = set(seen_items)
        a = self.alpha(len(seen_items))
        b = 1.0 - a

        candidates = self.pop.top_k(n_candidates, exclude=seen)
        ncf_scores = self._ncf_scores(user_id, candidates) if a > 0 else np.zeros(len(candidates))
        cold_scores = (
            self._cold_scores(seen_items, candidates) if b > 0 else np.zeros(len(candidates))
        )

        def norm(x):
            rng = x.max() - x.min()
            return (x - x.min()) / rng if rng > 1e-9 else x

        final_scores = a * norm(ncf_scores) + b * norm(cold_scores)
        top_idx = np.argsort(-final_scores)[:k]
        source = "ncf" if a >= 0.8 else ("blend" if a > 0.2 else "cold_start")

        return [
            {
                "item_id": candidates[idx],
                "score": round(float(final_scores[idx]), 4),
                "source": source,
                "alpha": round(a, 2),
            }
            for idx in top_idx
        ]
