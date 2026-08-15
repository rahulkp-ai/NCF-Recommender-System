"""
research/models/scratch_ncf/train.py

Full training loop for scratch NCF.
Logs loss per epoch and evaluates Hit@10 on validation set.
"""

# NCF-Recommender-System % python -m research.models.scratch_ncf.train

import json
import time
import numpy as np
from pathlib import Path
from tqdm import tqdm

from .forward   import ScratchNCF
from .loss      import BCELoss
from .optimizer import Adam

PROC_DIR = Path("research/datasets/processed")
LOG_DIR  = Path("research/evaluation")
LOG_DIR.mkdir(exist_ok=True)


# ── Evaluation: Hit@K ────────────────────────────────────────────────────────

def hit_at_k(model: ScratchNCF, val_df, n_items: int,
             k: int = 10, n_neg: int = 99, seed: int = 0) -> float:
    """
    Leave-one-out Hit@K evaluation.

    For each user, score their held-out positive + 99 random negatives.
    Hit@K = 1 if the positive is in the top-K ranked items.

    Returns mean Hit@K across all users.
    """
    rng = np.random.default_rng(seed)
    hits = 0

    for _, row in val_df.iterrows():
        uid      = int(row.user_id)
        pos_item = int(row.item_id)

        # Sample 99 negative items (random, may include some positives — acceptable)
        neg_items = rng.integers(0, n_items, size=n_neg).tolist()
        candidates = [pos_item] + neg_items   # positive always first

        # Score all candidates
        scores = [model.predict(uid, iid) for iid in candidates]

        # Rank: position of positive item (0-indexed)
        ranked = sorted(range(len(scores)), key=lambda x: -scores[x])
        pos_rank = ranked.index(0)            # index 0 = positive item

        if pos_rank < k:
            hits += 1

    return hits / len(val_df)


# ── Mini-batch data loader ────────────────────────────────────────────────────

def batch_iter(users: np.ndarray, items: np.ndarray,
               labels: np.ndarray, batch_size: int,
               rng: np.random.Generator):
    """Yield shuffled mini-batches of (user_ids, item_ids, labels)."""
    n = len(users)
    idx = rng.permutation(n)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        sl  = idx[start:end]
        yield users[sl], items[sl], labels[sl]


# ── Training loop ─────────────────────────────────────────────────────────────

def train(
    n_epochs:   int   = 20,
    emb_dim:    int   = 32,
    hidden:     list  = [64, 32],
    lr:         float = 1e-3,
    batch_size: int   = 512,
    eval_every: int   = 2,
    seed:       int   = 42,
) -> ScratchNCF:
    """
    Train scratch NCF on MovieLens 1M processed data.

    All hyperparameters are logged to evaluation/scratch_ncf_log.json
    so the comparison dashboard can read them.
    """
    print("=" * 60)
    print("Scratch NCF (NumPy) — Training")
    print("=" * 60)

    # Load data
    meta   = json.load(open(PROC_DIR / "meta.json"))
    n_users = meta["n_users"]
    n_items = meta["n_items"]

    users  = np.load(PROC_DIR / "train_users.npy")
    items  = np.load(PROC_DIR / "train_items.npy")
    labels = np.load(PROC_DIR / "train_labels.npy").astype(float)

    import pandas as pd
    val_df = pd.read_csv(PROC_DIR / "val.csv")

    # Build model
    model = ScratchNCF(n_users, n_items, emb_dim=emb_dim,
                       hidden=hidden, seed=seed)
    loss_fn = BCELoss()
    optim   = Adam(model.get_all_params(), lr=lr)

    print(f"Model: {model}")
    print(f"Parameters: emb_dim={emb_dim}, hidden={hidden}, lr={lr}")
    print(f"Training samples: {len(users):,}")
    print()

    rng = np.random.default_rng(seed)
    log = {
        "type": "scratch",
        "config": {"emb_dim": emb_dim, "hidden": hidden,
                   "lr": lr, "batch_size": batch_size},
        "epochs": [],
    }

    best_hit = 0.0
    t_start  = time.time()

    for epoch in range(1, n_epochs + 1):
        epoch_loss   = 0.0
        n_batches    = 0
        epoch_start  = time.time()

        for u_batch, i_batch, l_batch in batch_iter(
                users, items, labels, batch_size, rng):

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

        avg_loss    = epoch_loss / n_batches
        epoch_time  = time.time() - epoch_start

        # Evaluate every eval_every epochs
        hit_k = None
        if epoch % eval_every == 0:
            hit_k = hit_at_k(model, val_df.head(200),   # 200 users for speed
                             n_items, k=10)
            best_hit = max(best_hit, hit_k)
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Hit@10: {hit_k:.4f} | "
                  f"Time: {epoch_time:.1f}s")
        else:
            print(f"Epoch {epoch:3d}/{n_epochs} | "
                  f"Loss: {avg_loss:.4f} | "
                  f"Time: {epoch_time:.1f}s")

        log["epochs"].append({
            "epoch":      epoch,
            "loss":       round(avg_loss, 6),
            "hit_at_10":  round(hit_k, 4) if hit_k else None,
            "time_sec":   round(epoch_time, 2),
        })

    total_time = time.time() - t_start
    log["total_time_sec"] = round(total_time, 1)
    log["best_hit_at_10"] = round(best_hit, 4)

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
    train(n_epochs=20, emb_dim=32, hidden=[64, 32], lr=1e-3, batch_size=512)