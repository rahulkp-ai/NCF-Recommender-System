# Changelog

Referenced by `SECURITY.md` (hardening items) and
`docs/decisions/0002-hybrid-engine-refactor.md` (non-trivial ports between
`research/` and `production/`). Format loosely follows
[Keep a Changelog](https://keepachangelog.com/).

## [Unreleased]

### Security

- Removed a non-placeholder `TMDB_API_KEY` value that had been committed
  to `.env.example`; replaced with `your_tmdb_api_key_here`. If this key
  was ever pushed to a remote, treat it as compromised and rotate at
  themoviedb.org regardless of this fix (see `SECURITY.md`).

### Added

- `ROADMAP.md`, `CHANGELOG.md` — previously referenced from several docs
  (`SECURITY.md`, `research/README.md`, `production/artifacts/README.md`,
  `docs/architecture/overview.md`, ADR 0001, ADR 0004) but did not exist.
- Local zero-cost monitoring stack (Prometheus + Grafana) added to
  `docker-compose.yml`, scraping the existing `/metrics` endpoints on
  `backend` and `model_server`.

### Fixed

- `CODEOWNERS` updated to reflect the post-restructure `production/`
  layout; it previously referenced pre-restructure paths (`/backend/`,
  `/model_server/`, `/ml/`, `/gateway/`, `/infra/`) that no longer exist.

## Prior phases (reconstructed from in-repo documentation, not from git

history — this repository is being git-initialized for the first time
in this session, see the Phase notes embedded in code comments and ADRs
for the real chronology)

- Research phase: scratch NumPy NCF + PyTorch NCF implementations,
  leave-one-out evaluation protocol (He et al. 2017), scratch-vs-PyTorch
  comparison pipeline.
- Production restructure ("Phase 4" per in-code comments): consolidated
  prior `backend/`, `model_server/`, `ml/`, `gateway/` directories into a
  single `production/` package; fixed the resulting `sys.path` depth bug
  in both `production/backend/app/main.py` and
  `production/serving/app/main.py`.
- Observability pass ("Phase 5" per in-code comments): added
  `prometheus-client`-based `/metrics` instrumentation to backend and
  model_server; added the `AppError` exception-handling hierarchy.
- Hygiene pass (this session): secrets audit, `CODEOWNERS` fix,
  `ROADMAP.md`/`CHANGELOG.md` creation, git init, GitHub push, local
  Docker Compose deployment, Prometheus/Grafana wiring.
