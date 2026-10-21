"""
research/models/pytorch_ncf/train.py

Training loop for PyTorch NCF with full evaluation metric logging
(Hit@10, NDCG@10, Precision@10, Recall@10).

Mirrors scratch train.py exactly — same hyperparameter names,
same evaluation protocol, same log format — so the comparison
dashboard can overlay results directly.

Key differences from scratch:
- Processes full batches at once (no Python loop over samples)
- Uses MPS backend on Apple Silicon
- loss.backward() replaces manual backward()
"""
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

from research.mlops.experiment_tracker import ExperimentTracker
from .model import PyTorchNCF, get_device

RESEARCH_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = RESEARCH_DIR / "datasets" / "processed"
LOG_DIR = RESEARCH_DIR / "evaluation"
LOG_DIR.mkdir(exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────────────────

def build_dataloader(batch_size: int, device: torch.device) -> DataLoader:
    """Load processed numpy arrays and wrap in a PyTorch DataLoader."""
    users  = torch.from_numpy(np.load(PROC_DIR / "train_users.npy")).long()
    items  = torch.from_numpy(np.load(PROC_DIR / "train_items.npy")).long()
    labels = torch.from_numpy(
        np.load(PROC_DIR / "train_labels.npy").astype(np.float32)
    )
    dataset = TensorDataset(users, items, labels)
    return DataLoader(
        dataset,
        batch_size  = batch_size,
        shuffle     = True,
        num_workers = 0,         # 0 is safest for MPS
        pin_memory  = False,
    )


# ── Unified Evaluation: Hit@K, NDCG@K, Precision@K, Recall@K ─────────────────

def evaluate(model: PyTorchNCF, val_df: pd.DataFrame,
             n_items: int, k: int = 10,
             n_neg: int = 99, seed: int = 0,
             device: torch.device = torch.device("cpu")) -> dict:
    """
    Evaluates Hit@K, NDCG@K, Precision@K, and Recall@K simultaneously.
    Leave-One-Out Protocol: 1 positive item + n_neg negative samples per user.
    """
    model.eval()
    rng = np.random.default_rng(seed)

    hits, ndcgs, precisions, recalls = [], [], [], []

    with torch.no_grad():
        for _, row in val_df.iterrows():
            uid      = int(row.user_id)
            pos_item = int(row.item_id)

            neg_items  = rng.integers(0, n_items, size=n_neg).tolist()
            candidates = [pos_item] + neg_items       # pos item is always index 0

            u_tensor = torch.full((len(candidates),), uid,
                                  dtype=torch.long, device=device)
            i_tensor = torch.tensor(candidates,
                                    dtype=torch.long, device=device)

            scores   = model.predict(u_tensor, i_tensor).cpu().numpy()
            ranked   = np.argsort(-scores)            # descending order
            pos_rank = int(np.where(ranked == 0)[0][0])

            # Hits & Rank position logic
            hit  = 1.0 if pos_rank < k else 0.0
            ndcg = (1.0 / np.log2(pos_rank + 2)) if pos_rank < k else 0.0
            
            # Leave-One-Out metric equivalence:
            rec  = hit                       # Recall@10 = Hits / 1
            prec = hit / k                   # Precision@10 = Hits / K

            hits.append(hit)
            ndcgs.append(ndcg)
            precisions.append(prec)
            recalls.append(rec)

    model.train()
    
    return {
        "hit_at_k":       float(np.mean(hits)),
        "ndcg_at_k":      float(np.mean(ndcgs)),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k":    float(np.mean(recalls)),
    }
    
def hit_at_k(model: PyTorchNCF, df: pd.DataFrame, n_items: int,
             k: int = 10, n_neg: int = 99, seed: int = 0,
             device: torch.device = torch.device("cpu")) -> float:
    """Compute overall Hit@K score across dataframe."""
    res = evaluate(model, df, n_items, k=k, n_neg=n_neg, seed=seed, device=device)
    return res["hit_at_k"]


def ndcg_at_k(model: PyTorchNCF, df: pd.DataFrame, n_items: int,
              k: int = 10, n_neg: int = 99, seed: int = 0,
              device: torch.device = torch.device("cpu")) -> float:
    """Compute overall NDCG@K score across dataframe."""
    res = evaluate(model, df, n_items, k=k, n_neg=n_neg, seed=seed, device=device)
    return res["ndcg_at_k"]

# ── Training Loop ─────────────────────────────────────────────────────────────

def train(
    n_epochs:   int   = 20,
    emb_dim:    int   = 32,
    hidden:     list  = [64, 32],
    lr:         float = 1e-3,
    batch_size: int   = 1024,
    eval_every: int   = 1,
    seed:       int   = 42,
) -> PyTorchNCF:
    """Train PyTorch NCF and log complete metric evaluation data."""
    tracker = ExperimentTracker()
    run_id  = tracker.start_run("pytorch_ncf", {"emb_dim": emb_dim,
                                                 "hidden": hidden, "lr": lr})

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print("=" * 60)
    print(f"PyTorch NCF — Training  [device: {device}]")
    print("=" * 60)

    # Data
    loader  = build_dataloader(batch_size, device)
    meta    = json.load(open(PROC_DIR / "meta.json"))
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    val_df  = pd.read_csv(PROC_DIR / "val.csv")

    # Model
    model = PyTorchNCF(n_users, n_items, emb_dim=emb_dim,
                       hidden=hidden, dropout=0.0).to(device)
    print(model)
    print(f"Parameters: emb_dim={emb_dim}, hidden={hidden}, lr={lr}")
    print(f"Training batches per epoch: {len(loader):,}\n")

    # Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 betas=(0.9, 0.999), eps=1e-8)

    log = {
        "type": "pytorch",
        "device": str(device),
        "config": {"emb_dim": emb_dim, "hidden": hidden,
                   "lr": lr, "batch_size": batch_size},
        "epochs": [],
        "best_hit_at_10": 0.0,
        "best_ndcg_at_10": 0.0,
        "best_precision_at_10": 0.0,
        "best_recall_at_10": 0.0,
    }

    best_hit   = 0.0
    best_ndcg  = 0.0
    best_prec  = 0.0
    best_rec   = 0.0
    t_start    = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss  = 0.0
        epoch_start = time.time()

        for u_batch, i_batch, l_batch in loader:
            u_batch = u_batch.to(device)
            i_batch = i_batch.to(device)
            l_batch = l_batch.to(device)

            logits = model(u_batch, i_batch)
            loss   = criterion(logits, l_batch)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss   = epoch_loss / len(loader)
        epoch_time = time.time() - epoch_start

        metrics = None
        if epoch % eval_every == 0:
            eval_sub = val_df.head(200)
            metrics  = evaluate(model, eval_sub, n_items, k=10, device=device)
            
            best_hit  = max(best_hit, metrics["hit_at_k"])
            best_ndcg = max(best_ndcg, metrics["ndcg_at_k"])
            best_prec = max(best_prec, metrics["precision_at_k"])
            best_rec  = max(best_rec, metrics["recall_at_k"])

            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Hit@10: {metrics['hit_at_k']:.4f} | "
                  f"NDCG@10: {metrics['ndcg_at_k']:.4f} | "
                  f"Prec@10: {metrics['precision_at_k']:.4f} | "
                  f"Rec@10: {metrics['recall_at_k']:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        hit_k = metrics["hit_at_k"] if metrics else 0.0
        tracker.log_epoch(run_id, epoch=epoch, loss=avg_loss, hit_at_10=hit_k)

        log["epochs"].append({
            "epoch":           epoch,
            "loss":            round(avg_loss, 6),
            "hit_at_10":       round(metrics["hit_at_k"], 4) if metrics else None,
            "ndcg_at_10":      round(metrics["ndcg_at_k"], 4) if metrics else None,
            "precision_at_10": round(metrics["precision_at_k"], 4) if metrics else None,
            "recall_at_10":    round(metrics["recall_at_k"], 4) if metrics else None,
            "time_sec":        round(epoch_time, 2),
        })

    total_time = time.time() - t_start
    log["total_time_sec"]       = round(total_time, 1)
    log["best_hit_at_10"]       = round(best_hit, 4)
    log["best_ndcg_at_10"]      = round(best_ndcg, 4)
    log["best_precision_at_10"] = round(best_prec, 4)
    log["best_recall_at_10"]    = round(best_rec, 4)

    tracker.finish_run(run_id, best_hit_at_10=best_hit, n_epochs=n_epochs,
                       total_time_sec=total_time)

    with open(LOG_DIR / "pytorch_ncf_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete in {total_time:.1f}s | Best Hit@10: {best_hit:.4f}")
    print(f"Log saved to evaluation/pytorch_ncf_log.json")

    torch.save(model.state_dict(), LOG_DIR / "pytorch_ncf_weights.pt")
    print("Weights saved to evaluation/pytorch_ncf_weights.pt")

    return model


if __name__ == "__main__":
    train(n_epochs=20, emb_dim=32, hidden=[64, 32], lr=1e-3, batch_size=1024, eval_every=1)