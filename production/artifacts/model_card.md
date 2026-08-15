# Model Card: production_model.pt

## Model Details

- **Architecture**: MLP-based Neural Collaborative Filtering (NCF)
  — `production/recommenders/ncf/model.py:PyTorchNCF`
- **Embedding dim**: 32 (user + item)
- **Hidden layers**: [64, 32], ReLU activations, single-unit sigmoid output
- **Framework**: PyTorch
- **Serving wrapper**: `production/recommenders/hybrid/engine.py:HybridEngine`
  (blends this model's scores with `PopularityEngine` and `ContentEngine`
  for cold-start users — see `docs/architecture/overview.md`)

## Training Data

- MovieLens-1M (`research/datasets/raw/movielens/ml-1m/`)
- Train/val/test split produced in `research/` (see
  `research/datasets/processed/`)

## Evaluation Results

(as reported in this repository's `README.md` "Model Performance" table,
research/evaluation/, and the associated IEEE paper)

| Metric   | Scratch NCF (NumPy) | PyTorch NCF (this artifact) |
| -------- | ------------------- | --------------------------- |
| BCE Loss | 0.6133              | 0.6111                      |
| Hit@10   | 0.6279              | 0.6293                      |
| NDCG@10  | —                   | 0.3541                      |

## Intended Use

Movie recommendation for the demo application in `production/frontend/` +
`production/backend/`. Not intended for use outside this demo context —
trained on MovieLens-1M's fixed user/item population, not designed to
generalize to unseen users/items without retraining
(`scripts/train.py` in `research/`).

## Limitations

- Cold-start users (< 20 interactions) are served primarily by
  `PopularityEngine`/`ContentEngine` blending rather than this model —
  see `HybridEngine.alpha()`.
- Trained on a closed, fixed item catalog (MovieLens-1M movies only) —
  cannot score items outside that catalog without retraining and
  re-exporting `meta.json`.

## Provenance

Promoted from a `research/experiments/` training run. Currently a manual
copy — `scripts/export_model.py` (planned, see `ROADMAP.md`) will
formalize this hand-off and write this file automatically alongside the
weights in future promotions.
