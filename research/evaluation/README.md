# `research/evaluation/`

Everything related to measuring and comparing the two NCF
implementations (`research/models/scratch_ncf/` vs. `pytorch_ncf/`).

- `metrics.py` — `hit_at_k`, `ndcg_at_k`, `precision_at_k` (also imported
  by `production/tests/unit/test_metrics.py` — a documented
  research→production exception, see `research/README.md`'s "Boundary
  note" and `docs/decisions/0001`).
- `plots.py` — generates `figures/paper/fig1`–`fig4` for the IEEE paper.
- `compare.py` — loads both models' training logs and builds
  `comparison.json` (the data `research/comparisons/`'s dashboard serves).
- `benchmark_summary.ipynb` — the headline results table (promoted from
  a stray `notes/` folder in Phase 6 — see repo-root `PHASE6_REPORT.md`).
- `*_log.json` — per-epoch training logs for each model.
- `*_weights.{npz,pt}` — the trained weights from these research runs
  (the PyTorch one is the same file promoted to
  `production/artifacts/production_model.pt`).
- `pytorch_test_results.json` — held-out test set evaluation results.
- `figures/paper/`, `figures/thesis/` — see `research/README.md` for why
  there are two sets, not one.

## Regenerating results
```bash
python -m research.models.scratch_ncf.train
python -m research.models.pytorch_ncf.train
python -m research.evaluation.compare      # writes comparison.json
python -m research.evaluation.plots        # writes figures/paper/*
```
