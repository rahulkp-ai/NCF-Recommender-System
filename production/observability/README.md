# `production/observability/`

Cross-service instrumentation shared by `production/backend` and
`production/serving` (added in Phase 5 — see `PHASE5_REPORT.md`).

- `logging/` — `configure_logging()`: one logging config for both
  services, human-readable text by default or JSON via `LOG_FORMAT=json`.
- `metrics/` — `instrument_app()`: request count + latency middleware,
  exposes `/metrics` in Prometheus text format. Closes a gap where
  `deployment/monitoring/prometheus/prometheus.yml` (Phase 2) was
  scraping an endpoint that didn't exist yet.
- `tracing/`, `alerts/` — still empty; no tracing infra or alerting
  channel exists to wire up yet. See
  `docs/decisions/0004-deferred-enterprise-patterns.md`.
