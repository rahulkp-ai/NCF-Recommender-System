# `research/comparisons/`

Serves the scratch-vs-PyTorch comparison data for interactive viewing —
distinct from `research/evaluation/compare.py`, which *generates* that
data (see that file's docstring and `research/README.md`).

- `api/compare.py` — a small FastAPI router (`GET /api/compare/data`,
  `GET /api/compare/summary`) serving `research/evaluation/comparison.json`.
- `dashboard/index.html` — a static dashboard consuming that API.

Run the generator first (`python -m research.evaluation.compare`) so
`comparison.json` exists before starting this API.
