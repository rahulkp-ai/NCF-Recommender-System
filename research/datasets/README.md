# `research/datasets/`

Renamed from `research/data/` in Phase 4 (to avoid a naming collision
with `production/artifacts/` — see `docs/decisions/0003`).

- `raw/movielens/ml-1m/` — the original MovieLens-1M files (`movies.dat`,
  `ratings.dat`, `users.dat`), the original `.zip`, and its `README`.
  6,040 users, 3,706 movies (see `production/artifacts/meta.json` for
  the exact split sizes used downstream).
- `processed/` — train/val/test CSV splits, `.npy` tensors for fast
  loading, and the pickled `pop_engine.pkl`/`content_engine.pkl`
  cold-start engines (the same files promoted to
  `production/artifacts/` — see `ARCHITECTURE.md`'s data-flow diagram).

Regenerate `processed/` from `raw/` via `models/*/train.py`'s data
loading path, or `datasets/movielens_loader.py` directly.
