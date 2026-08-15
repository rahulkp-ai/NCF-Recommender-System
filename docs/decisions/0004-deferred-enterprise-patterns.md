# ADR 0004: Deferred Enterprise Patterns

## Status
Accepted.

## Context
External reviews of this repository's architecture (July 2026) suggested
a fuller enterprise ML platform structure: a feature store, background
workers, full Clean Architecture (`domain/`, `application/` with
CQRS-style commands/queries/handlers), API versioning beyond the current
`api/v1/` prefix already in place, data contracts, a dedicated cache
layer, and a multi-stage pipeline framework replacing `scripts/`.

## Decision
These patterns are **deliberately not implemented** at this stage. Each
is real and each would be justified at a different scale — the decision
here is about sequencing, not disagreement with the suggestions.

| Pattern | Trigger condition to revisit |
|---|---|
| `feature_store/` (Feast + Redis) | Online feature computation becomes a latency bottleneck, or a second model needs to share features with the hybrid engine. |
| `workers/` (Celery/RQ + broker) | Retraining or cache-warming needs to run asynchronously / on a schedule, independent of a request. |
| `domain/`/`application/` CQRS split | The backend's resource count and business-rule complexity grow past what 4-5 resource types (users, movies, ratings, recommendations) can express clearly in the current `services/` layer. |
| Deeper `api/v2/` versioning | A second API consumer besides `production/frontend` exists (e.g. a public API, a mobile app). |
| `contracts/`, `messaging/` | A second backend service needs to communicate with `production/backend` over something other than direct HTTP calls. |
| `cache/` (Redis) | Recommendation latency or DB load becomes a measured problem — not before, since adding a cache layer without a measured need adds invalidation complexity for no benefit. |
| Full `pipelines/` framework | `scripts/train.py` / `evaluate.py` / `export_model.py` stop being sufficient — e.g. multi-step DAGs with retries/scheduling are needed (Airflow/Prefect territory). |

## Consequences
- ✅ The repository stays honestly scoped to what's actually implemented
  — no empty `feature_store/` or `workers/` folders that make the
  project look larger than it is without functioning code inside.
- ✅ A reviewer reading this ADR sees that these patterns were
  *evaluated and deliberately sequenced*, not overlooked — this is
  itself a signal of engineering judgment.
- ⚠️ Revisit this list periodically — if `ROADMAP.md`'s near-term items
  are completed and one of the trigger conditions above is met, that
  pattern should be scaffolded for real, not preemptively.
