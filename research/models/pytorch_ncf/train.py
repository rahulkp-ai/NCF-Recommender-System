"""
research/models/pytorch_ncf/train.py

Training loop for PyTorch NCF.
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
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import PyTorchNCF, get_device

PROC_DIR = Path("research/datasets/processed")
LOG_DIR = Path("research/evaluation")
LOG_DIR.mkdir(exist_ok=True)


# ── Dataset ──────────────────────────────────────────────────────────────────


def build_dataloader(batch_size: int, device: torch.device) -> DataLoader:
    """Load processed numpy arrays and wrap in a PyTorch DataLoader."""
    users = torch.from_numpy(np.load(PROC_DIR / "train_users.npy")).long()
    items = torch.from_numpy(np.load(PROC_DIR / "train_items.npy")).long()
    labels = torch.from_numpy(np.load(PROC_DIR / "train_labels.npy").astype(np.float32))
    dataset = TensorDataset(users, items, labels)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # 0 is safest for MPS
        pin_memory=False,
    )


# ── Evaluation: Hit@K ────────────────────────────────────────────────────────


def hit_at_k(
    model: PyTorchNCF,
    val_df: pd.DataFrame,
    n_items: int,
    k: int = 10,
    n_neg: int = 99,
    seed: int = 0,
    device: torch.device | None = None,
) -> float:
    """
    Leave-one-out Hit@K — identical protocol to scratch implementation.
    Vectorised: scores all candidates for all users in one batch.
    """
    if device is None:
        device = torch.device("cpu")
    model.eval()
    rng = np.random.default_rng(seed)
    hits = 0

    with torch.no_grad():
        for _, row in val_df.iterrows():
            uid = int(row.user_id)
            pos_item = int(row.item_id)

            neg_items = rng.integers(0, n_items, size=n_neg).tolist()
            candidates = [pos_item] + neg_items  # pos always index 0

            u_tensor = torch.full((len(candidates),), uid, dtype=torch.long, device=device)
            i_tensor = torch.tensor(candidates, dtype=torch.long, device=device)

            scores = model.predict(u_tensor, i_tensor).cpu().numpy()
            ranked = np.argsort(-scores)  # descending
            pos_rank = int(np.where(ranked == 0)[0][0])

            if pos_rank < k:
                hits += 1

    model.train()
    return hits / len(val_df)


# ── Training loop ─────────────────────────────────────────────────────────────


def train(
    n_epochs: int = 20,
    emb_dim: int = 32,
    hidden: list | None = None,
    lr: float = 1e-3,
    batch_size: int = 1024,
    eval_every: int = 2,
    seed: int = 42,
) -> PyTorchNCF:
    """
    Train PyTorch NCF and log results in the same format as scratch trainer.

    Benchmark comparison note:
      - batch_size=1024 is larger than scratch (512) to exploit GPU batching
      - For exact apples-to-apples timing, also run with batch_size=512
    """
    if hidden is None:
        hidden = [64, 32]
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = get_device()
    print("=" * 60)
    print(f"PyTorch NCF — Training  [device: {device}]")
    print("=" * 60)

    # Data
    loader = build_dataloader(batch_size, device)
    with open(PROC_DIR / "meta.json") as f:
        meta = json.load(f)
    n_users = meta["n_users"]
    n_items = meta["n_items"]
    val_df = pd.read_csv(PROC_DIR / "val.csv")

    # Model
    model = PyTorchNCF(n_users, n_items, emb_dim=emb_dim, hidden=hidden, dropout=0.0).to(device)
    print(model)
    print(f"Parameters: emb_dim={emb_dim}, hidden={hidden}, lr={lr}")
    print(f"Training batches per epoch: {len(loader):,}\n")

    # Loss and optimiser — identical hyperparameters to scratch
    criterion = nn.BCEWithLogitsLoss()  # numerically stable sigmoid+BCE
    optimiser = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)

    log = {
        "type": "pytorch",
        "device": str(device),
        "config": {"emb_dim": emb_dim, "hidden": hidden, "lr": lr, "batch_size": batch_size},
        "epochs": [],
    }

    best_hit = 0.0
    t_start = time.time()

    for epoch in range(1, n_epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for u_batch, i_batch, l_batch in loader:
            u_batch = u_batch.to(device)
            i_batch = i_batch.to(device)
            l_batch = l_batch.to(device)

            # ── Forward ──────────────────────────────────────────────
            logits = model(u_batch, i_batch)  # (B,)
            loss = criterion(logits, l_batch)

            # ── Backward ─────────────────────────────────────────────
            optimiser.zero_grad()
            loss.backward()  # autograd traverses graph
            optimiser.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)
        epoch_time = time.time() - epoch_start

        hit_k = None
        if epoch % eval_every == 0:
            hit_k = hit_at_k(model, val_df.head(200), n_items, k=10, device=device)
            best_hit = max(best_hit, hit_k)
            print(
                f"Epoch {epoch:3d}/{n_epochs} | "
                f"Loss: {avg_loss:.4f} | "
                f"Hit@10: {hit_k:.4f} | "
                f"Time: {epoch_time:.1f}s"
            )
        else:
            print(f"Epoch {epoch:3d}/{n_epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")

        log["epochs"].append(
            {
                "epoch": epoch,
                "loss": round(avg_loss, 6),
                "hit_at_10": round(hit_k, 4) if hit_k else None,
                "time_sec": round(epoch_time, 2),
            }
        )

    total_time = time.time() - t_start
    log["total_time_sec"] = round(total_time, 1)
    log["best_hit_at_10"] = round(best_hit, 4)

    with open(LOG_DIR / "pytorch_ncf_log.json", "w") as f:
        json.dump(log, f, indent=2)

    print(f"\nTraining complete in {total_time:.1f}s | Best Hit@10: {best_hit:.4f}")
    print("Log saved to evaluation/pytorch_ncf_log.json")

    # Save weights
    torch.save(model.state_dict(), LOG_DIR / "pytorch_ncf_weights.pt")
    print("Weights saved to research/evaluation/pytorch_ncf_weights.pt")

    return model


if __name__ == "__main__":
    train(n_epochs=20, emb_dim=32, hidden=[64, 32], lr=1e-3, batch_size=1024)
