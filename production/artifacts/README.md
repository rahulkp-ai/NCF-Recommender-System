# `production/artifacts/`

Trained model weights and derived data only — no source code, no raw
datasets. Read-only at runtime (mounted `:ro` in `docker-compose.yml`).

| File | Used by |
|---|---|
| `production_model.pt` | `production/recommenders/ncf/model.py` (via `HybridEngine.load()`) |
| `pop_engine.pkl`, `content_engine.pkl` | `HybridEngine`'s cold-start blending |
| `meta.json` | `n_users`/`n_items`/split sizes, read by `HybridEngine.load()` |
| `model_card.md` | Human-readable model documentation |
| `training_config.json` | Hyperparameter snapshot for this promoted model |
| `version.txt` | Current promoted model version |
| `seed_data/movies.dat` | `production/backend/app/db/seed.py` — demo DB seeding (added in Phase 5, see `PHASE5_REPORT.md`) |

## Provenance
Currently populated by a manual copy from `research/evaluation/` +
`research/datasets/processed/`. `scripts/export_model.py` (planned —
see `ROADMAP.md`) will automate this hand-off.
