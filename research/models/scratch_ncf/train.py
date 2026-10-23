"""
models/scratch_ncf/train.py

Full training loop for scratch NCF (NumPy).
Logs loss per epoch and evaluates Hit@10, NDCG@10, Precision@10, and Recall@10.
"""

import json
import time
import numpy as np
import pandas as pd
from pathlib import Path


from research.mlops.experiment_tracker import ExperimentTracker
from .forward import Scratch_NCF
from .loss import BCELoss
from .optimizer import Adam

RESEARCH_DIR = Path(__file__).resolve().parents[2]
PROC_DIR = RESEARCH_DIR / "datasets" / "processed"
LOG_DIR = RESEARCH_DIR / "evaluation"
LOG_DIR.mkdir(exist_ok=True)


# ── Unified Evaluation: Hit@K, NDCG@K, Precision@K, Recall@K ─────────────────

def evaluate(model: Scratch_NCF, val_df: pd.DataFrame, n_items: int, 
             k: int = 10, n_neg: int = 99, seed: int = 0) -> dict:
    """
    Evaluates Hit@K, NDCG@K, Precision@K, and Recall@K simultaneously for Scratch NCF.
    Leave-One-Out Protocol: 1 positive item + n_neg negative samples per user.
    """
    rng = np.random.default_rng(seed)
    hits, ndcgs, precisions, recalls = [], [], [], []

    for _, row in val_df.iterrows():
        uid      = int(row.user_id)
        pos_item = int(row.item_id)

        # Sample 99 negative items
        neg_items  = rng.integers(0, n_items, size=n_neg).tolist()
        candidates = [pos_item] + neg_items  # Positive always at index 0

        # Compute scores for all candidates
        scores = [model.predict(uid, iid) for iid in candidates]

        # Rank position of positive item (0-indexed)
        ranked   = sorted(range(len(candidates)), key=lambda x: -scores[x])
        pos_rank = ranked.index(0)

        # Hit & Rank metrics
        hit  = 1.0 if pos_rank < k else 0.0
        ndcg = (1.0 / np.log2(pos_rank + 2)) if pos_rank < k else 0.0

        # Leave-One-Out metric equivalence:
        rec  = hit          # Recall@10 = Hits / 1
        prec = hit / k      # Precision@10 = Hits / K

        hits.append(hit)
        ndcgs.append(ndcg)
        precisions.append(prec)
        recalls.append(rec)

    return {
        "hit_at_k":       float(np.mean(hits)),
        "ndcg_at_k":      float(np.mean(ndcgs)),
        "precision_at_k": float(np.mean(precisions)),
        "recall_at_k":    float(np.mean(recalls)),
    }


# ── Mini-batch data loader ────────────────────────────────────────────────────

def batch_iter(user: np.ndarray, items: np.ndarray, 
               labels: np.ndarray, batch_size: int,
               rng: np.random.Generator):
    """Yield shuffled mini-batches of (user_ids, item_ids, labels)."""
    n   = len(user)
    idx = rng.permutation(n)

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        s1  = idx[start:end]
        yield user[s1], items[s1], labels[s1]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    n_epochs:   int   = 20,
    emb_dim:    int   = 32,
    hidden:     list  = [64, 32],
    lr:         float = 1e-3,
    batch_size: int   = 512,
    eval_every: int   = 1,
    seed:       int   = 42,
) -> Scratch_NCF:
    """
    Train scratch NCF on MovieLens 1M processed data.

    Logs all metrics to evaluation/scratch_ncf_log.json
    so the comparison dashboard can overlay results directly.
    """
    tracker = ExperimentTracker()
    run_id  = tracker.start_run("scratch_ncf", {"emb_dim": emb_dim,
                                                 "hidden": hidden, "lr": lr})

    print("=" * 60)
    print("Scratch NCF (NumPy) — Training")
    print("=" * 60)

    # Load data
    meta    = json.load(open(PROC_DIR / "meta.json"))
    n_users = meta["n_users"]
    n_items = meta["n_items"]

    users  = np.load(PROC_DIR / "train_users.npy")
    items  = np.load(PROC_DIR / "train_items.npy")
    labels = np.load(PROC_DIR / "train_labels.npy").astype(float)

    val_df = pd.read_csv(PROC_DIR / "val.csv")

    # Build model
    model   = Scratch_NCF(n_users, n_items, emb_dim=emb_dim, hidden=hidden, seed=seed)
    loss_fn = BCELoss()
    optim   = Adam(model.get_all_params(), lr=lr)

    print(f"Model: {model}")
    print(f"Parameters: emb_dim={emb_dim}, hidden={hidden}, lr={lr}")
    print(f"Training samples: {len(users):,}\n")

    rng = np.random.default_rng(seed)
    log = {
        "type": "scratch",
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
        epoch_loss  = 0.0
        n_batches   = 0
        epoch_start = time.time()

        for u_batch, i_batch, l_batch in batch_iter(users, items, labels, batch_size, rng):
            batch_loss = 0.0
            model.zero_grad()

            for u, i, y in zip(u_batch, i_batch, l_batch):
                # Forward
                logit = model.forward(int(u), int(i))
                loss  = loss_fn.forward(logit, float(y))
                batch_loss += loss

                # Backward
                grad = loss_fn.backward()   # = ŷ - y
                model.backward(grad)
                loss_fn.zero_grad()

            # Update once per batch (mini-batch SGD)
            optim.step()
            model.zero_grad()

            epoch_loss += batch_loss / len(u_batch)
            n_batches  += 1

        avg_loss   = epoch_loss / n_batches
        epoch_time = time.time() - epoch_start

        metrics = None
        if epoch % eval_every == 0:
            eval_sub = val_df.head(200)
            metrics  = evaluate(model, eval_sub, n_items, k=10, seed=seed)

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

    # Save log for comparison dashboard
    with open(LOG_DIR / "scratch_ncf_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete in {total_time:.1f}s | Best Hit@10: {best_hit:.4f}")
    print(f"Log saved to evaluation/scratch_ncf_log.json")

    # Save model weights
    np.savez(LOG_DIR / "scratch_ncf_weights.npz",
             user_emb=model.user_emb.W,
             item_emb=model.item_emb.W,
             **{f"W{i}": l.W for i, l in enumerate(model.linears)},
             **{f"b{i}": l.b for i, l in enumerate(model.linears)})
    print("Weights saved to evaluation/scratch_ncf_weights.npz")

    return model


if __name__ == "__main__":
    train(n_epochs=20, emb_dim=32, hidden=[64, 32], lr=1e-3, batch_size=512, eval_every=1)