# `production/backend/`

FastAPI service handling authentication, user data, search, and
orchestration of recommendation requests (delegating actual model
inference to `production/serving/`). Moved from `backend/` in Phase 4 —
internals unchanged.

## Responsibilities
- User auth (JWT)
- Movie search / metadata enrichment (TMDB)
- Calling `production/serving` for recommendations and enriching the response
- Persisting user interactions to Postgres

## Run locally
```bash
uvicorn production.backend.app.main:app --reload --port 8000
```
Docs at `/docs`. See `docs/developer-guide/getting-started.md` for full setup.

## Boundary rule
Never import from `research/`. See `ARCHITECTURE.md`.
