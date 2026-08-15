# `production/shared/`

Cross-service code used by more than one of `production/backend`,
`production/serving`, `production/recommenders`.

## `exceptions/` — populated in Phase 5
The `AppError` hierarchy (`NotFoundError`, `ValidationError`,
`ModelServerUnavailableError`, etc.) — see `exceptions/errors.py`. Used by
`production/backend`'s services/routes today; available to
`production/serving` if/when it needs the same error shape.

## `config/`, `constants/`, `types/` — checked in Phase 5, still empty
Looked for genuine cross-service duplication that would justify moving
code here (the usual trigger for a `shared/` module) and didn't find
any: `production/backend/app/core/constants.py` (blending weights,
search limits) is backend-specific with no `serving`-side duplicate;
config is similarly service-specific today (`backend/app/core/config.py`
vs. `serving`'s own env handling). Left empty with this note rather than
populated with speculative content — see
`docs/decisions/0004-deferred-enterprise-patterns.md` for the same
principle applied elsewhere.

**Trigger to revisit**: if `production/serving` grows its own
`config.py`/`constants.py` and any values genuinely duplicate the
backend's (e.g. the same `MODEL_SERVER_URL` default, or shared
enum-like types for recommendation "strategy" values currently
duplicated as string literals — `"ncf"`, `"blend"`, `"cold_start"` —
in `recommendation_service.py`), consolidate here.
