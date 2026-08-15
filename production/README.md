# `production/`

The deployed system — everything a user of the live app touches. Never
imports from `research/` — see `ARCHITECTURE.md` and
`docs/decisions/0001-research-production-separation.md`.

| Component | Purpose |
|---|---|
| `frontend/` | Next.js UI |
| `backend/` | Auth, search, user data, orchestrates recommendations |
| `serving/` | Standalone inference service (loads the trained model) |
| `recommenders/` | NCF + hybrid + content + popularity model code |
| `gateway-optional/` | Optional JWT proxy, not wired in by default |
| `artifacts/` | Trained weights + engines (git-ignored data, not code) |
| `shared/` | Cross-service code (currently: the `AppError` exception hierarchy) |
| `observability/` | Logging + metrics, shared by `backend` and `serving` |
| `validation/` | Currently empty by design — see its own README |
| `tests/` | `unit/`, `integration/`, `e2e/` |

See `docs/architecture/overview.md` for the full request-flow diagram.
