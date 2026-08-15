# Roadmap

This file exists to back up the several `(planned, see ROADMAP.md)` references
scattered through the codebase (`SECURITY.md`, `research/README.md`,
`production/artifacts/model_card.md`, `production/artifacts/README.md`,
`docs/architecture/overview.md`, `docs/decisions/0001-*.md`,
`docs/decisions/0004-*.md`) with one real, honest list — not a rewrite of
those documents' intent.

## Near-term (blocks manual toil, no new infra)

- [ ] `scripts/export_model.py` — automates the currently-manual promotion
      of a trained checkpoint from `research/evaluation/` +
      `research/datasets/processed/` into `production/artifacts/`. Referenced
      as planned in four places (see above); this is the single most-cited
      gap in the docs and the natural next thing to build once the
      deployment/monitoring pass in this session is done.

## Deferred by design (see `docs/decisions/0004-deferred-enterprise-patterns.md`)

Not missing — deliberately not built yet, with explicit trigger conditions:

- [ ] `cache/` (Redis) — only once recommendation latency or DB load is a
      _measured_ problem.
- [ ] `contracts/`, `messaging/` — only once a second backend service needs
      to talk to `production/backend` over something other than direct HTTP.
- [ ] Full `pipelines/` framework (Airflow/Prefect-style DAGs) — only once
      `scripts/train.py` / `evaluate.py` / `export_model.py` stop being
      sufficient on their own.

## Longer-term (from README "Future Improvements")

- [ ] Feature Store integration
- [ ] Real-time streaming (Kafka)
- [ ] Online learning
- [ ] Deployed, publicly-accessible instance (target: Render/Railway +
      Vercel free tiers — see `docs/deployment/README.md`)

## Already shipped (kept here only to show what moved off this list)

- [x] CI/CD pipeline (`.github/workflows/`)
- [x] Basic HTTP monitoring (`/metrics` endpoints, `prometheus-client`
      instrumentation on backend + model_server)
- [x] Local zero-cost Docker Compose deployment + Prometheus/Grafana wiring
      — done in this session, see `CHANGELOG.md`
