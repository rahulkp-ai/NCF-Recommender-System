# `production/tests/unit/`

Moved from the original flat `tests/` directory in Phase 4. Despite the
name, `test_api.py` requires a live Postgres (it's DB-backed via
`TestClient`, only `run_seed`/`RecommendationService.load` are mocked)
— see `production/tests/integration/README.md` for the note on this
pre-existing classification, which Phase 4 didn't relitigate.

- `test_api.py` — backend endpoint tests
- `test_ml.py` — `PopularityEngine`/`ContentEngine`/`HybridEngine` unit tests
- `test_metrics.py` — metric function tests (imports `research.evaluation.metrics`
  — a documented research→production exception, see root `research/README.md`)

Run: `pytest production/tests/unit/ -v`
