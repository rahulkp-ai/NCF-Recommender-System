"""
research/datasets/movielens_loader.py

Loads, validates, and preprocesses MovieLens 1M into NCF-ready tensors.
Implements leave-one-out evaluation split (He et al. 2017).
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent

# 2. Define RAW_DIR relative to the script location
RAW_DIR   = BASE_DIR / "raw" / "movielens" / "ml-1m"

# 3. Define PROC_DIR relative to the script location
PROC_DIR  = BASE_DIR / "processed"

PROC_DIR.mkdir(parents=True, exist_ok=True)


# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_ratings(path: Path = RAW_DIR / "ratings.dat") -> pd.DataFrame:
    """Load ratings.dat and return a clean DataFrame."""
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",          # required for multi-char sep
        names=["user_id", "item_id", "rating", "timestamp"],
        dtype={"user_id": int, "item_id": int,
               "rating": float,  "timestamp": int},
    )
    print(f"Loaded {len(df):,} ratings | "
          f"{df.user_id.nunique():,} users | "
          f"{df.item_id.nunique():,} items")
    return df


def load_movies(path: Path = RAW_DIR / "movies.dat") -> pd.DataFrame:
    """Load movies.dat — note latin-1 encoding for special characters."""
    df = pd.read_csv(
        path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )
    return df


# ── 2. Validate ──────────────────────────────────────────────────────────────

def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Assert invariants and drop bad rows. Raises on critical failures."""
    assert df.isnull().sum().sum() == 0,  "Null values detected — check raw file"
    assert (df.rating >= 1).all(),        "Rating below 1 detected"
    assert (df.rating <= 5).all(),        "Rating above 5 detected"

    before = len(df)
    df = df.drop_duplicates(subset=["user_id", "item_id"], keep="last")
    print(f"Dropped {before - len(df)} duplicate (user, item) pairs")
    return df


# ── 3. Binarise implicit feedback ────────────────────────────────────────────

def binarise(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """
    Convert explicit ratings to implicit feedback.
    threshold=0 keeps ALL interactions (any rating = watched).
    For stricter implicit: threshold=3.5 keeps only positive ratings.

    Thesis note: We treat any interaction as positive signal (threshold=0)
    following standard implicit CF practice.
    """
    df = df[df.rating > threshold].copy()
    df["label"] = 1
    print(f"After binarisation: {len(df):,} positive interactions")
    return df[["user_id", "item_id", "label", "timestamp"]]


# ── 4. Re-index IDs to contiguous integers ──────────────────────────────────

def reindex(df: pd.DataFrame):
    """
    Map original IDs → 0-based contiguous integers.

    This is CRITICAL for embedding lookups:
      P[user_id, :] requires user_id in [0, M-1]
      Q[item_id, :] requires item_id in [0, N-1]

    Returns:
        df          - DataFrame with new integer IDs
        user_map    - dict {original_id → new_id}
        item_map    - dict {original_id → new_id}
        n_users     - total number of unique users
        n_items     - total number of unique items
    """
    user_ids = sorted(df.user_id.unique())
    item_ids = sorted(df.item_id.unique())

    user_map = {old: new for new, old in enumerate(user_ids)}
    item_map = {old: new for new, old in enumerate(item_ids)}

    df = df.copy()
    df["user_id"] = df.user_id.map(user_map)
    df["item_id"] = df.item_id.map(item_map)

    n_users, n_items = len(user_map), len(item_map)
    print(f"Re-indexed: {n_users:,} users (0..{n_users-1}) | "
          f"{n_items:,} items (0..{n_items-1})")
    return df, user_map, item_map, n_users, n_items


# ── 5. Leave-one-out split ───────────────────────────────────────────────────

def leave_one_out_split(df: pd.DataFrame):
    """
    Standard NCF evaluation protocol (He et al. 2017):
      - For each user, hold out their LAST interaction (by timestamp) as test
      - Hold out second-to-last as validation
      - Everything else is training data

    Returns: train_df, val_df, test_df
    """
    df = df.sort_values(["user_id", "timestamp"])

    # Last interaction per user → test
    test_idx = df.groupby("user_id").tail(1).index
    remaining = df.drop(test_idx)

    # Second-to-last → val
    val_idx = remaining.groupby("user_id").tail(1).index
    train_df = remaining.drop(val_idx)
    val_df   = remaining.loc[val_idx]
    test_df  = df.loc[test_idx]

    print(f"Split — train: {len(train_df):,} | "
          f"val: {len(val_df):,} | test: {len(test_df):,}")
    return train_df.reset_index(drop=True), \
           val_df.reset_index(drop=True),   \
           test_df.reset_index(drop=True)


# ── 6. Negative sampling ────────────────────────────────────────────────────

def sample_negatives(
    df: pd.DataFrame,
    n_items: int,
    n_neg: int = 4,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each positive (user, item) pair, sample n_neg items the user
    has NOT interacted with.

    This produces the training set used to optimise BCE loss:
      L = -log(ŷ_pos) - Σ log(1 - ŷ_neg)

    Returns DataFrame with columns: user_id, item_id, label
    """
    rng = np.random.default_rng(seed)

    # Build set of observed items per user for fast lookup
    user_pos = df.groupby("user_id")["item_id"].apply(set).to_dict()

    records = []
    for uid, pos_items in user_pos.items():
        for pos_item in pos_items:
            records.append((uid, pos_item, 1))      # positive
            # sample negatives, rejecting known positives
            neg_pool = list(set(range(n_items)) - pos_items)
            negs = rng.choice(neg_pool, size=n_neg, replace=False)
            for neg in negs:
                records.append((uid, int(neg), 0))  # negative

    result = pd.DataFrame(records, columns=["user_id", "item_id", "label"])
    print(f"Sampled negatives — total rows: {len(result):,} "
          f"(ratio 1:{n_neg})")
    return result.sample(frac=1, random_state=seed).reset_index(drop=True)


# ── 7. Save processed data ──────────────────────────────────────────────────

def save_processed(train, val, test, user_map, item_map, n_users, n_items):
    train.to_csv(PROC_DIR / "train.csv", index=False)
    val.to_csv(  PROC_DIR / "val.csv",   index=False)
    test.to_csv( PROC_DIR / "test.csv",  index=False)

    # Save as numpy arrays for fast loading during training
    np.save(PROC_DIR / "train_users.npy",  train.user_id.values)
    np.save(PROC_DIR / "train_items.npy",  train.item_id.values)
    np.save(PROC_DIR / "train_labels.npy", train.label.values)

    # Save metadata
    meta = {"n_users": n_users, "n_items": n_items,
            "n_train": len(train), "n_val": len(val), "n_test": len(test)}
    import json
    with open(PROC_DIR / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved processed data to {PROC_DIR}/")
    print(f"meta.json: {meta}")


# ── 8. Main pipeline ─────────────────────────────────────────────────────────

def run_pipeline():
    print("=" * 60)
    print("MovieLens 1M preprocessing pipeline")
    print("=" * 60)

    ratings = load_ratings()
    ratings = validate(ratings)
    ratings = binarise(ratings, threshold=0.0)

    ratings, user_map, item_map, n_users, n_items = reindex(ratings)

    train_raw, val_df, test_df = leave_one_out_split(ratings)

    # Apply negative sampling only to training set
    train_df = sample_negatives(train_raw, n_items, n_neg=4)

    save_processed(train_df, val_df, test_df, user_map, item_map,
                   n_users, n_items)

    print("\nPipeline complete.")
    return train_df, val_df, test_df, n_users, n_items


if __name__ == "__main__":
    run_pipeline()