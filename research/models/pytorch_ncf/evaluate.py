"""
research/models/pytorch_ncf/evaluate.py

Load a saved PyTorch NCF model and evaluate on the test set.
This is the script referenced in your thesis results table.
"""

import json
import pandas as pd
import torch
from pathlib import Path
from .model import PyTorchNCF, get_device
from .train import hit_at_k, ndcg_at_k

RESEARCH_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = RESEARCH_DIR / "datasets" / "processed"
LOG_DIR  = RESEARCH_DIR / "evaluation"


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