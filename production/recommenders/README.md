# `production/recommenders/`

Production model code: architecture definitions and inference-time logic
for the NCF model and the hybrid (collaborative + content + popularity)
recommendation engine. Moved from `ml/models/ncf/` in Phase 4 and split
into sibling packages — see
`docs/decisions/0003-recommenders-serving-rename.md`.

**Contains no training code and no raw datasets** — it only reads trained
weights from `../artifacts/`. Training happens in `research/`.

## Structure
- `ncf/model.py` — `PyTorchNCF`, the trained network architecture
- `popularity/engine.py` — `PopularityEngine`, cold-start fallback
- `content/engine.py` — `ContentEngine`, cold-start fallback
- `hybrid/engine.py` — `HybridEngine`, blends all three signal sources
- `hybrid/_pickle_compat.py` — compatibility shim so pre-split pickled
  artifacts still deserialize (see file docstring)

## Relationship to `research/models/hybrid/`
This is a deliberate hardened refactor, not a copy — see
`docs/decisions/0002-hybrid-engine-refactor.md`. Do not merge the two.

## Boundary rule
Never import from `research/`. See `ARCHITECTURE.md`.
