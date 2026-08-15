# Deployment Guide

## Target: free-tier PaaS (portfolio deployment)

Given this repo's goal is a public portfolio demo rather than
production-scale traffic, the recommended deployment path is a **free-tier
Platform-as-a-Service**, not Kubernetes:

| Component             | Suggested free-tier target                                    | Notes                                                                                                                                                                                                                                                                                             |
| --------------------- | ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `production/frontend` | **Vercel**                                                    | Native Next.js support, generous free tier, zero-config from GitHub.                                                                                                                                                                                                                              |
| `backend`             | **Render** (Web Service, free tier) or **Railway**            | Deploy from `deployment/docker/backend.Dockerfile`. Free tier sleeps on idle — acceptable for a portfolio demo.                                                                                                                                                                                   |
| `model_server`        | Same platform as `backend`, separate service                  | Keep it separate so the model can be redeployed independently (see `docs/architecture/overview.md`). Watch free-tier memory limits — the loaded model + PyTorch must fit (typically 512MB on free tiers; test `artifacts/production_model.pt` size against this before committing to a provider). |
| Postgres              | **Neon** or **Supabase** free tier, or Render's free Postgres | `backend` connects via `DATABASE_URL` env var already defined in `.env.example`.                                                                                                                                                                                                                  |

## Why not Kubernetes / Terraform for now

`deployment/kubernetes/` and `deployment/terraform/` are kept in this repo
as **reference scaffolding** for a future scale-up, not the near-term
deployment path — see `ROADMAP.md`. Running k8s has real cost even at
"free tier" (cluster management overhead, no truly free managed k8s for a
side project), and adds operational complexity a single-maintainer
portfolio project doesn't need yet.

## Steps

1. Provision free Postgres (Neon/Supabase), copy connection string.
2. Deploy `model_server` first (it has no external dependencies besides
   `artifacts/`) — confirm `/health` responds.
3. Deploy `backend`, pointing `MODEL_SERVER_URL` at step 2's URL and
   `DATABASE_URL` at step 1.
4. Deploy `production/frontend` on Vercel, pointing `NEXT_PUBLIC_API_URL` at step 3's
   URL.
5. Confirm CORS origins in `backend/app/core/config.py` include the
   Vercel domain.
6. Rotate `SECRET_KEY`, `JWT_SECRET`, and `TMDB_API_KEY` to real values in
   each platform's environment variable settings — never the checked-in
   dev defaults (see `SECURITY.md`).

## Local development

Use `docker-compose.yml` + `docker-compose.override.yml` — see
`docs/developer-guide/getting-started.md`.
