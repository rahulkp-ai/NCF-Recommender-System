# Architecture Overview

See `ARCHITECTURE.md` at the repo root for the authoritative summary. This
page expands on the request flow for a single recommendation request.

## Request flow: "get recommendations for user X"

1. **`production/frontend`** — user opens their recommendations page. Frontend calls
   `GET /api/recommendations/{user_id}` on `backend`.
2. **`backend`** — authenticates the request (JWT), validates `user_id`,
   fetches the user's interaction history from Postgres, and calls
   `model_server` with the feature payload.
3. **`model_server`** — loads (once, at startup) the trained hybrid engine
   from `artifacts/`, runs inference (`ml/models/ncf/hybrid_engine.py`),
   blends collaborative, content-based, and popularity signals (with
   cold-start fallback for new users via `cold_start.py`), and returns
   ranked movie IDs + scores.
4. **`backend`** — enriches ranked IDs with movie metadata (title, poster,
   genres — some sourced from TMDB), returns JSON to the frontend.
5. **`production/frontend`** — renders the recommendation carousel.

## Why inference is a separate service from the backend

`model_server` is intentionally its own FastAPI process rather than a
module imported into `backend`, so that:
- Model loading (weights into memory) doesn't block/slow backend startup.
- The model can be scaled or redeployed independently of the API layer.
- A future swap to a different serving stack (e.g. TorchServe, BentoML)
  only touches `model_server`, not `backend`.

## Data flow: research → production (target state)

```
research/experiments/expNNN_*/train.py
        │  (produces best checkpoint + metrics.json)
        ▼
scripts/export_model.py   ← promotes one experiment's artifact
        │
        ▼
artifacts/production_model.pt  (consumed by model_server, nothing else)
```

This bridge script does not exist yet in the current repo (flagged in the
Phase 1 audit and `ROADMAP.md`) — today the artifact is copied by hand.
