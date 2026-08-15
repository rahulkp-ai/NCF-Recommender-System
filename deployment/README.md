# `deployment/`

Docker and nginx configuration for the deployed stack. Moved from
`infra/` in Phase 4.

- `docker/` — one Dockerfile per service (`backend`, `frontend`,
  `model_server`), consumed by `docker-compose.yml` and
  `.github/workflows/cd.yaml`.
- `nginx/` — reverse-proxy config used as the default gateway in front of
  `production/backend`/`production/frontend` (see
  `production/gateway-optional/README.md` for the alternative
  application-layer gateway that is *not* currently wired in).

For Kubernetes/Terraform reference scaffolding (not the near-term
deployment path), see `deployment/kubernetes/` and
`deployment/terraform/` at the repo root, and
`docs/deployment/README.md` for the actual (free-tier PaaS) deployment
plan.
