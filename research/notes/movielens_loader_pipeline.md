# MovieLens 1M Preprocessing Pipeline for Neural Collaborative Filtering (NCF)

This pipeline converts raw explicit movie ratings from the MovieLens 1M dataset into binarized, reindexed, and negative-sampled data ready for deep learning via Neural Collaborative Filtering (NCF). It strictly follows the standard empirical evaluation methodologies introduced by He et al. (2017).

---

## Pipeline Architecture Overview

The ETL workflow processes raw files through 6 distinct stages before exporting compressed binary payloads and metadata to disk:

```text
[Raw .dat Files] ──> 1. Load & Stream ──> 2. Clean & Validate ──> 3. Implicit Binarization
                                                                            │
[Saved Payloads] <── 6. Serialize Disk <── 5. Vectorized Negatives <── 4. 0-Based Reindexing
```

---

## Input Data Specifications

The pipeline expects the unzipped MovieLens 1M dataset structured under your project directory as follows:

- **Directory Root:** `data/raw/movielens/ml-1m/`
- **Target Files:**
  - `ratings.dat`: User ratings formatted as `UserID::MovieID::Rating::Timestamp`
  - `movies.dat`: Movie metadata formatted as `MovieID::Title::Genres`

---

## Detailed Phase Breakdown

### 1. Load & Enforce Types (`load_ratings`, `load_movies`)

- Parses multi-character double-colon separators (`::`) using the fallback Python parsing engine.
- Enforces explicit native memory allocations: `int32` for structural IDs and `float64` for rating weights.
- Restricts text parsing overhead by utilizing `latin-1` byte decoding on structural metadata strings.

### 2. Guardrail Validation (`validate`)

- **Null Check:** Asserts complete data density across all columns, halting execution if corrupted rows leak in.
- **Range Bounds:** Guarantees implicit scalar integrity by asserting:

[
1.0 \le \text{rating} \le 5.0
]

- **Deduplication:** Strips overlapping `(user_id, item_id)` timeline artifacts, retaining only the chronological tail interaction.

### 3. Implicit Feedback Conversion (`binarise`)

- Recommendation engines perform best when converting explicit numbers into implicit signals.
- By default, a tracking `threshold=0.0` maps all observed records to a hard tracking `label: 1`.
- This assumes that if a user chose to interact with a movie, it represents a positive implicit preference signal.

### 4. Dense Coordinate Reindexing (`reindex`)

- Deep learning frameworks use sparse lookup matrices (`nn.Embedding`).
- These matrices require contiguous integer coordinates starting precisely at index `0`.
- This phase reads unique keys, maps them to an unbroken sequence `[0, M-1]` for users and `[0, N-1]` for items, and records translation dictionaries (`user_map`, `item_map`) to safely decode outputs later.

### 5. Leave-One-Out Evaluation Split (`leave_one_out_split`)

Rather than a random shuffle split, this module implements a **chronological time-split** to mimic production behavior:

- **`test_df`**: Contains the absolute newest interaction (`tail(1)`) for every distinct user.
- **`val_df`**: Contains the second-to-last chronological interaction for every user.
- **`train_df`**: Contains all remaining early history used to fit model parameters.

### 6. High-Speed Vectorized Negative Sampling (`sample_negatives_optimized`)

- Models optimizing Binary Cross-Entropy (BCE) loss need contrast examples (`label: 0`) to prevent gradient collapse.
- This module matches every positive training item with `4` unobserved negative items (`4:1` ratio).

**Optimization Note:** Instead of relying on slow nested Python iteration loops, this uses **Vectorized Rejection Sampling**. It samples massive blocks of item indices natively using NumPy and resolves rare history collisions using highly optimized vectorized bitmasks.

---

## Output Storage Target Layout

Processed outputs are automatically structured inside the `data/processed/` folder across three distinct functional formats:

```text
data/processed/
├── meta.json         # Global model-dimension parameters (JSON config)
├── train.csv         # Full training dataset framework (Human-readable check)
├── val.csv           # Holdout tuning anchors containing real timelines
├── test.csv          # Evaluation benchmarks containing real timelines
├── train_users.npy   # Raw high-speed binary array for training user indices
├── train_items.npy   # Raw high-speed binary array for training item indices
└── train_labels.npy  # Raw high-speed binary targets (mixed 0s and 1s)
```

### Generated Schema Payloads

#### `meta.json` (System Architecture Dimensions)

```json
{
  "n_users": 6040,
  "n_items": 3706,
  "n_train": 4940645,
  "n_val": 6040,
  "n_test": 6040
}
```

#### `train.csv` / `_users.npy` Tensor Framework

| user_id (Dense) | item_id (Dense) | label | timestamp |
| --------------- | --------------- | ----- | --------- |
| 142             | 982             | 0     | None      |
| 12              | 42              | 1     | 978220114 |
| 0               | 2401            | 0     | None      |

> **Note:** `val.csv` and `test.csv` preserve original sequence timestamps to anchor historical tracking evaluations.

---

## Execution Commands

To execute the data preparation workflow inside your active virtual environment container, run:

```bash
python data/movielens_loader.py
```
