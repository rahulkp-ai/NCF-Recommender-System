# `tests/`

Production test suite (backend API, metrics, ML inference logic).
Separate from `research/tests/`, which tests research-only code
(e.g. gradient checks on from-scratch implementations).

## Structure
- `test_api.py`, `test_metrics.py`, `test_ml.py` — existing unit-level tests
- `integration/` — cross-component tests (e.g. backend → model_server over
  HTTP), added in this hardening pass, currently empty pending Phase 5
- `e2e/` — full-stack tests (frontend → backend → model_server), added in
  this hardening pass, currently empty pending Phase 5

## Run
```bash
pytest                 # all tests, per pyproject.toml testpaths
pytest tests/test_ml.py -v
```
