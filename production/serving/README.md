# `production/serving/`

Standalone FastAPI service whose only job is to load the trained hybrid
model from `../artifacts/` and serve inference requests, independent of
the main backend API. Moved from `model_server/` in Phase 4 — see
`docs/decisions/0003-recommenders-serving-rename.md`.

## Why separate from backend
- Independent scaling/redeploy without touching auth/search logic.
- Model loading (weights into memory) doesn't block backend startup.
- See `docs/architecture/overview.md` for the full request-flow rationale.

## Run locally
```bash
uvicorn production.serving.app.main:app --reload --port 8001
```
Health check: `GET /health`.

## Boundary rule
Never import from `research/`. Only reads from `../artifacts/` and `../recommenders/`.
