# `research/models/`

Three genuinely different implementations, not variations of one thing —
this project's central experiment is comparing the first two:

- **`scratch_ncf/`** — Neural Collaborative Filtering built from first
  principles in NumPy: custom `embeddings.py`, `forward.py`, `layers.py`,
  `loss.py`, `optimizer.py`, and `train.py`. No autograd framework — 
  gradients are hand-derived and checked in `research/tests/test_gradients.py`.
- **`pytorch_ncf/`** — the same architecture, in PyTorch, trained on
  identical data/hyperparameters for a fair speed and quality comparison
  (see `research/evaluation/compare.py` and `benchmark_summary.ipynb`).
- **`hybrid/`** — blends the trained NCF model's collaborative-filtering
  score with content-based and popularity signals, plus cold-start
  handling for users with few/no interactions.
  `production/recommenders/` has its own hardened, since-split copy of
  this logic — see `docs/decisions/0002` and `0003` for why they're
  intentionally two separate implementations, not one shared module.

Run any model's `train.py` from the repo root (each sets up its own
`PROC_DIR` relative to `research/datasets/processed/`).
