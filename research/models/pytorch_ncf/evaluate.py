"""
models/pytorch_ncf/evaluate.py

Load a saved PyTorch NCF model and evaluate on the test set.
This is the script referenced in your thesis results table.
"""

# python -m research.models.pytorch_ncf.evaluate

import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from .model import PyTorchNCF, get_device
from .train import hit_at_k

PROC_DIR = Path("research/datasets/processed")
LOG_DIR  = Path("research/evaluation")


def ndcg_at_k(model, val_df, n_items, k=10,
              n_neg=99, seed=0, device=None):
    """
    NDCG@K — Normalised Discounted Cumulative Gain.
    Measures not just whether the positive is in top-K,
    but how highly it is ranked (higher = better).

    NDCG@K = 1/log2(rank+2) if positive is in top-K, else 0
    (Ideal DCG for a single relevant item = 1/log2(2) = 1.0)
    """
    if device is None:
        device = get_device()
    model.eval()
    rng   = np.random.default_rng(seed)
    ndcgs = []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            uid      = int(row.user_id)
            pos_item = int(row.item_id)
            neg_items  = rng.integers(0, n_items, size=n_neg).tolist()
            candidates = [pos_item] + neg_items

            u_t = torch.full((len(candidates),), uid,
                             dtype=torch.long, device=device)
            i_t = torch.tensor(candidates, dtype=torch.long, device=device)
            scores   = model.predict(u_t, i_t).cpu().numpy()
            ranked   = np.argsort(-scores)
            pos_rank = int(np.where(ranked == 0)[0][0])

            ndcg = 1.0 / np.log2(pos_rank + 2) if pos_rank < k else 0.0
            ndcgs.append(ndcg)

    return float(np.mean(ndcgs))


def full_evaluation(weights_path: str = None) -> dict:
    """Run Hit@10 and NDCG@10 on the full test set."""
    meta    = json.load(open(PROC_DIR / "meta.json"))
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    test_df = pd.read_csv(PROC_DIR / "test.csv")
    device  = get_device()

    model = PyTorchNCF(n_users, n_items, emb_dim=32, hidden=[64, 32])
    if weights_path is None:
        weights_path = LOG_DIR / "pytorch_ncf_weights.pt"
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)

    print(f"Evaluating on {len(test_df):,} test users...")
    hit  = hit_at_k(model, test_df, n_items, k=10, device=device)
    ndcg = ndcg_at_k(model, test_df, n_items, k=10, device=device)

    results = {"hit_at_10": round(hit, 4), "ndcg_at_10": round(ndcg, 4)}
    print(f"Test  Hit@10  : {hit:.4f}")
    print(f"Test  NDCG@10 : {ndcg:.4f}")

    with open(LOG_DIR / "pytorch_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


if __name__ == "__main__":
    full_evaluation()