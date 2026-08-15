"""
research/evaluation/metrics.py

All evaluation metrics for the NCF recommendation system.
Implements the exact protocol from He et al. (2017):
  - Leave-one-out split (Phase 2)
  - 99 random negatives per test user
  - Hit@K and NDCG@K

Also implements Precision@K and Recall@K for thesis completeness.
"""

import numpy as np
import pandas as pd
from typing import Callable


# ── Core ranking metrics ──────────────────────────────────────────────────────

def hit_at_k(ranked_list: list, relevant_item: int, k: int) -> float:
    """
    Hit@K: 1 if relevant_item appears in the top-K of ranked_list.
    """
    return 1.0 if relevant_item in ranked_list[:k] else 0.0


def ndcg_at_k(ranked_list: list, relevant_item: int, k: int) -> float:
    """
    NDCG@K: 1/log2(rank+1) if relevant_item is in top-K, else 0.
    Maximum value = 1.0 (item at rank 1).
    """
    if relevant_item not in ranked_list[:k]:
        return 0.0
    rank = ranked_list.index(relevant_item) + 1   # 1-indexed
    return 1.0 / np.log2(rank + 1)


def precision_at_k(ranked_list: list, relevant_items: set, k: int) -> float:
    """Fraction of top-K that are relevant."""
    top_k = ranked_list[:k]
    hits  = sum(1 for item in top_k if item in relevant_items)
    return hits / k


def recall_at_k(ranked_list: list, relevant_items: set, k: int) -> float:
    """Fraction of all relevant items that appear in top-K."""
    if not relevant_items:
        return 0.0
    top_k = ranked_list[:k]
    hits  = sum(1 for item in top_k if item in relevant_items)
    return hits / len(relevant_items)


# ── Full evaluation runner ────────────────────────────────────────────────────

def evaluate_model(
    score_fn:   Callable[[int, list[int]], np.ndarray],
    test_df:    pd.DataFrame,
    n_items:    int,
    k:          int   = 10,
    n_neg:      int   = 99,
    seed:       int   = 0,
    verbose:    bool  = True,
) -> dict:
    """
    Run full leave-one-out evaluation.
    """
    rng      = np.random.default_rng(seed)
    hits     = []
    ndcgs    = []
    precs    = []
    recs     = []
    per_user = []

    for _, row in test_df.iterrows():
        uid       = int(row.user_id)
        pos_item  = int(row.item_id)

        # Sample n_neg negatives
        neg_items  = rng.integers(0, n_items, size=n_neg).tolist()
        candidates = [pos_item] + neg_items    # positive always at index 0

        # Score all candidates
        scores = score_fn(uid, candidates)     # shape (100,)

        # Rank descending
        ranked_idx  = np.argsort(-scores)
        ranked_list = [candidates[i] for i in ranked_idx]

        # Compute metrics
        h = hit_at_k(ranked_list, pos_item, k)
        n = ndcg_at_k(ranked_list, pos_item, k)
        p = precision_at_k(ranked_list, {pos_item}, k)
        r = recall_at_k(ranked_list, {pos_item}, k)

        hits.append(h); ndcgs.append(n); precs.append(p); recs.append(r)
        per_user.append({"user_id": uid, "hit": h, "ndcg": n,
                         "pos_rank": ranked_list.index(pos_item) + 1})

    results = {
        "hit_at_k":       round(float(np.mean(hits)),  4),
        "ndcg_at_k":      round(float(np.mean(ndcgs)), 4),
        "precision_at_k": round(float(np.mean(precs)), 4),
        "recall_at_k":    round(float(np.mean(recs)),  4),
        "k":              k,
        "n_users":        len(test_df),
        "per_user":       per_user,
    }

    if verbose:
        print(f"Evaluation @K={k} over {len(test_df):,} users:")
        print(f"  Hit@{k}       : {results['hit_at_k']:.4f}")
        print(f"  NDCG@{k}      : {results['ndcg_at_k']:.4f}")
        print(f"  Precision@{k} : {results['precision_at_k']:.4f}")
        print(f"  Recall@{k}    : {results['recall_at_k']:.4f}")

    return results

# ── Main Execution Block ──────────────────────────────────────────────────────
"""
if __name__ == "__main__":
    # This part runs ONLY when you execute this file directly.
    
    print("--- Initializing Mock Evaluation ---")
    
    # 1. Create a dummy test set (5 users)
    dummy_data = {
        'user_id': [101, 102, 103, 104, 105],
        'item_id': [500, 600, 700, 800, 900]
    }
    df = pd.DataFrame(dummy_data)

    # 2. Create a mock scoring function
    # In a real scenario, this would be your Neural CF model's .predict()
    def mock_score_fn(user_id, candidates):
        # Returns random scores for the sake of demonstration
        return np.random.uniform(0, 1, size=len(candidates))

    # 3. Run the evaluation
    results = evaluate_model(
        score_fn=mock_score_fn,
        test_df=df,
        n_items=1000,
        k=10,
        n_neg=99
    )
    
    print("\nScript executed successfully.")
    """