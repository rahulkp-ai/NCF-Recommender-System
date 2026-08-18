# Deployment & Monitoring Architecture

This document describes the deployment and observability architecture added
after the research/production split covered in `docs/architecture/overview.md`
and the ADRs in `docs/decisions/`. It is written to be citable directly in the
thesis's systems chapter.

Every claim below is labeled with its actual status:

- **Implemented** — code/config exists and is committed.
- **Tested** — implemented _and_ verified working end-to-end on this machine.
- **Configured** — set up but not exercised under real load.
- **Optional** — present in the repo as a documented alternative, not the
  default path.
- **Future work** — not present; listed in `ROADMAP.md`.

---

## 1. Deployment Architecture

**Status: Implemented, Tested.**

The system runs as a five-service Docker Compose stack on a single host
(developed and verified on a MacBook Pro M1 Pro, 32GB RAM, Docker Desktop).
This is the default and only deployment path exercised in this project — the
free-tier cloud PaaS path described in `docs/deployment/README.md` is
**Optional**, documented but not deployed.

```
                        Browser
                           |
                           v
                    nginx  (:80)
                    /            \
                   v              v
        backend (:8000)      web (:3001->:3000)
        FastAPI                Next.js (standalone)
             |
             +--> db (:5433->5432)      Postgres 16
             |
             +--> model_server (:8001)  FastAPI, loads HybridEngine
                        |
                        v
              production/artifacts/
              (production_model.pt, content_engine.pkl,
               pop_engine.pkl — promoted from research/evaluation/)
```

All five services are defined in `docker-compose.yml` with `restart:
unless-stopped` and explicit healthchecks (`db`, `backend`, `model_server`).
`web` and `nginx` do not have healthchecks defined; `web`'s readiness is
implied by `backend`'s `service_healthy` dependency condition, and `nginx`
starts after both `backend` and `web` are up. This gap is noted as **Future
work** in `ROADMAP.md` rather than silently treated as covered.

### 1.1 Container Architecture

**Status: Implemented, Tested.**

| Service        | Base image                     | Build                                       | Purpose                                                                                    |
| -------------- | ------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `db`           | `postgres:16-alpine`           | — (stock image)                             | Interaction/user/movie data, persisted via the `pgdata` named volume                       |
| `model_server` | `python:3.11-slim`             | `deployment/docker/model_server.Dockerfile` | Standalone inference service; loads the trained `HybridEngine` at startup                  |
| `backend`      | `python:3.11-slim`             | `deployment/docker/backend.Dockerfile`      | FastAPI app — auth, search, recommend, interaction APIs; seeds and serves the demo dataset |
| `web`          | `node:20-alpine` (multi-stage) | `deployment/docker/frontend.Dockerfile`     | Next.js frontend, built with `output: "standalone"`                                        |
| `nginx`        | `nginx:alpine`                 | — (stock image, config mounted read-only)   | Single entry point (`:80`), reverse-proxies to `backend`, `model_server`, and `web`        |

`backend` and `model_server` both mount `production/artifacts/` and
`production/recommenders/` read-only, so the trained model can be swapped by
replacing files on the host without rebuilding the image — this was a
deliberate volume-mount choice, not an oversight.

### 1.2 Kubernetes

**Status: Not deployed. Reference scaffolding only.**

`deployment/kubernetes/` contains a single example manifest
(`backend-deployment.yaml`) and its own README stating explicitly that this is
not the current deployment path. No Kubernetes cluster (local `kind`/Minikube
or otherwise) was used or verified in this project. Listed here only so the
gap is not accidentally implied by the directory's existence — this is the
same caution the original project audit flagged as necessary
("do not assume that a directory existing means its functionality is
implemented").

---

## 2. Request Flow

**Status: Tested.**

```
Browser
  --GET /-->                nginx --> web (Next.js SSR/static)
  --GET /api/v1/...-->      nginx --> backend
  --GET /docs, /health-->   nginx --> backend
  --GET /model/...-->       nginx --> model_server   (debugging only)
```

Verified via direct `curl` against each path and via full browser sessions
(homepage load, search, poster rendering) through `http://localhost` — not
just the individual container ports. This distinction matters: an earlier
verification pass surfaced a real bug where the app worked when the frontend
container was accessed directly on `:3001` but failed through nginx's `:80`
entry point, because the backend's CORS allow-list did not include the
`http://localhost` origin nginx presents to the browser. Root cause and fix
are recorded in git history (commit `82227ad`) and are a useful worked
example for the thesis of why testing must exercise the _documented_ entry
point, not just any working port.

## 3. Model Inference Flow

**Status: Implemented, Tested.**

```
POST /api/v1/recommend/... (backend)
  --> RecommendationService (production/backend/app/services/)
        --> HybridEngine.recommend()  (production/recommenders/hybrid/)
              alpha(n_interactions) = min(1.0, n / N_WARMUP)
              score = alpha * NCF_score + (1 - alpha) * cold_start_score
        --> strategy label: "ncf" | "blend" | "cold_start"
GET /api/v1/search (backend)
  --> SearchService --> MovieRepository (DB text search)
        --> strategy label: "search"
```

`HybridEngine` is loaded once at `backend` startup (`RecommendationService.load()`)
and again independently inside `model_server` for the standalone inference
endpoint — the two services do not share a running process, only the same
artifact files on disk. `model_server`'s `/predict` and `/batch` endpoints
exist and are health-checked, but were not exercised under real recommendation
traffic in this session (no authenticated user session was driven through it
end-to-end) — noted as a verification gap, not a code gap.

## 4. Observability Flow

**Status: Implemented, Tested.**

```
backend, model_server  --(instrument_app, prometheus-client)-->  /metrics
                                          |
                                          v
                                    prometheus (:9090)
                                    scrapes backend:8000/metrics,
                                    model_server:8001/metrics every 15s
                                          |
                                          v
                                     grafana (:3002)
                                     datasource + dashboards
                                     auto-provisioned on startup
```

### 4.1 Metrics

**Status: Implemented, Tested.**

| Metric                                                                      | Type          | Labels                          | Source                                                                                                 |
| --------------------------------------------------------------------------- | ------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `http_requests_total`                                                       | Counter       | `service, method, path, status` | generic HTTP middleware, both services                                                                 |
| `http_request_duration_seconds`                                             | Histogram     | `service, method, path`         | generic HTTP middleware, both services                                                                 |
| `recommendation_strategy_total`                                             | Counter       | `service, strategy`             | added specifically to distinguish `ncf` / `blend` / `cold_start` / `search` traffic (commit `99d32bc`) |
| `up`                                                                        | Gauge         | `job`                           | generated automatically by Prometheus per scrape target, not app code                                  |
| `process_cpu_seconds_total`, `process_resident_memory_bytes`, `python_gc_*` | Counter/Gauge | —                               | `prometheus_client`'s built-in process/runtime collectors, no app code                                 |

No metric in either dashboard was invented without a corresponding
implementation — where a panel from the original design brief (e.g. per-model
inference-only latency, isolated from `/health`/`/metrics` scrape noise) had
no real supporting data, the panel queries the closest real metric
(`model_server`'s full request latency, which currently includes its own
healthcheck and scrape traffic since no authenticated inference traffic was
generated during testing) rather than fabricating a number.

### 4.2 Dashboards

**Status: Implemented, Tested, Provisioned.**

Both dashboards are checked into
`deployment/monitoring/grafana/dashboards/*.json` and auto-load via
`deployment/monitoring/grafana/provisioning/` — verified by a fresh Grafana
login showing both dashboards with no manual import step.

**System Overview** (`system-overview.json`): request rate, error rate (by
status code), P50/P95/P99 latency for `backend` and `model_server`
separately, CPU, memory, and a per-service `up`/`down` stat panel.

**Recommender Performance** (`recommender-performance.json`): recommendation
request rate by endpoint, strategy distribution and share (pie chart) from
`recommendation_strategy_total`, recommend-endpoint latency and error rate,
model server inference latency, and running totals by strategy.

### 4.3 Verified failure scenarios

**Status: Tested.**

Two distinct failure classes were deliberately triggered and confirmed
visible end-to-end (request → metric → Prometheus → Grafana panel):

1. **Application-level error** — `GET /api/v1/recommend/movie/999999`
   (nonexistent ID) correctly returned a structured `404` via the
   `AppError`/`NotFoundError` handler, and appeared within one Prometheus
   scrape interval as `http_requests_total{status="404"}` and on the
   Recommender Performance dashboard's error-rate panel with the specific
   path labeled.
2. **Infrastructure-level failure** — `docker compose stop backend` produced
   an immediate connection failure (not a hang), Prometheus reported
   `up{job="backend"} = 0` within one scrape interval, and Grafana's Service
   Availability panel flipped `backend` to red `DOWN` while correctly leaving
   `model_server` green `UP` — confirming per-service, not global, failure
   attribution. `docker compose start backend` restored `Up (healthy)` status
   and the dashboard returned to green within ~15 seconds.

### 4.4 Logging

**Status: Implemented (structured, stdout only). No log aggregation.**

Both services use `production/observability/logging/setup.py` for structured
log output to stdout, captured via `docker compose logs`. No centralized log
store (Loki, ELK, or similar) is deployed — logs are only as durable as the
container's log driver retains them. Listed as **Future work**.

### 4.5 Alerting

**Status: Not implemented.**

`deployment/observability/alerts/` referenced in the original scaffolding
audit does not contain active alert rules. Grafana's alerting subsystem is
available (visible in its left nav) but no alert rules were configured.
Listed as **Future work** in `ROADMAP.md`.

---

## 5. Summary Table

| Component                                                     | Implemented | Tested |             Configured              |      Optional       | Future work |
| ------------------------------------------------------------- | :---------: | :----: | :---------------------------------: | :-----------------: | :---------: |
| Docker Compose (5 services)                                   |     ✅      |   ✅   |                                     |                     |             |
| Kubernetes                                                    |             |        |                                     | ✅ (reference only) |             |
| CORS / nginx routing                                          |     ✅      |   ✅   |                                     |                     |             |
| `/health` endpoints                                           |     ✅      |   ✅   |                                     |                     |             |
| HTTP metrics (`http_requests_total`, latency histogram)       |     ✅      |   ✅   |                                     |                     |             |
| Recommendation strategy metric                                |     ✅      |   ✅   |                                     |                     |             |
| Prometheus                                                    |     ✅      |   ✅   |                                     |                     |             |
| Grafana + auto-provisioned dashboards                         |     ✅      |   ✅   |                                     |                     |             |
| Application-level failure visibility (404 test)               |     ✅      |   ✅   |                                     |                     |             |
| Infrastructure-level failure visibility (container stop test) |     ✅      |   ✅   |                                     |                     |             |
| `web`/`nginx` healthchecks                                    |             |        |                                     |                     |     ✅      |
| `model_server` inference under real (non-synthetic) traffic   |             |        | ✅ (endpoint exists, not exercised) |                     |             |
| Centralized log aggregation                                   |             |        |                                     |                     |     ✅      |
| Alerting rules                                                |             |        |                                     |                     |     ✅      |
| Free-tier cloud deployment                                    |             |        |           ✅ (documented)           |         ✅          |             |

# Deployment & Monitoring Architecture

This document describes the deployment and observability architecture added
after the research/production split covered in `docs/architecture/overview.md`
and the ADRs in `docs/decisions/`. It is written to be citable directly in the
thesis's systems chapter.

Every claim below is labeled with its actual status:

- **Implemented** — code/config exists and is committed.
- **Tested** — implemented _and_ verified working end-to-end on this machine.
- **Configured** — set up but not exercised under real load.
- **Optional** — present in the repo as a documented alternative, not the
  default path.
- **Future work** — not present; listed in `ROADMAP.md`.

---

## 1. Deployment Architecture

**Status: Implemented, Tested.**

The system runs as a five-service Docker Compose stack on a single host
(developed and verified on a MacBook Pro M1 Pro, 32GB RAM, Docker Desktop).
This is the default and only deployment path exercised in this project — the
free-tier cloud PaaS path described in `docs/deployment/README.md` is
**Optional**, documented but not deployed.

```
                        Browser
                           |
                           v
                    nginx  (:80)
                    /            \
                   v              v
        backend (:8000)      web (:3001->:3000)
        FastAPI                Next.js (standalone)
             |
             +--> db (:5433->5432)      Postgres 16
             |
             +--> model_server (:8001)  FastAPI, loads HybridEngine
                        |
                        v
              production/artifacts/
              (production_model.pt, content_engine.pkl,
               pop_engine.pkl — promoted from research/evaluation/)
```

All five services are defined in `docker-compose.yml` with `restart:
unless-stopped` and explicit healthchecks (`db`, `backend`, `model_server`).
`web` and `nginx` do not have healthchecks defined; `web`'s readiness is
implied by `backend`'s `service_healthy` dependency condition, and `nginx`
starts after both `backend` and `web` are up. This gap is noted as **Future
work** in `ROADMAP.md` rather than silently treated as covered.

### 1.1 Container Architecture

**Status: Implemented, Tested.**

| Service        | Base image                     | Build                                       | Purpose                                                                                    |
| -------------- | ------------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `db`           | `postgres:16-alpine`           | — (stock image)                             | Interaction/user/movie data, persisted via the `pgdata` named volume                       |
| `model_server` | `python:3.11-slim`             | `deployment/docker/model_server.Dockerfile` | Standalone inference service; loads the trained `HybridEngine` at startup                  |
| `backend`      | `python:3.11-slim`             | `deployment/docker/backend.Dockerfile`      | FastAPI app — auth, search, recommend, interaction APIs; seeds and serves the demo dataset |
| `web`          | `node:20-alpine` (multi-stage) | `deployment/docker/frontend.Dockerfile`     | Next.js frontend, built with `output: "standalone"`                                        |
| `nginx`        | `nginx:alpine`                 | — (stock image, config mounted read-only)   | Single entry point (`:80`), reverse-proxies to `backend`, `model_server`, and `web`        |

`backend` and `model_server` both mount `production/artifacts/` and
`production/recommenders/` read-only, so the trained model can be swapped by
replacing files on the host without rebuilding the image — this was a
deliberate volume-mount choice, not an oversight.

### 1.2 Kubernetes

**Status: Not deployed. Reference scaffolding only.**

`deployment/kubernetes/` contains a single example manifest
(`backend-deployment.yaml`) and its own README stating explicitly that this is
not the current deployment path. No Kubernetes cluster (local `kind`/Minikube
or otherwise) was used or verified in this project. Listed here only so the
gap is not accidentally implied by the directory's existence — this is the
same caution the original project audit flagged as necessary
("do not assume that a directory existing means its functionality is
implemented").

---

## 2. Request Flow

**Status: Tested.**

```
Browser
  --GET /-->                nginx --> web (Next.js SSR/static)
  --GET /api/v1/...-->      nginx --> backend
  --GET /docs, /health-->   nginx --> backend
  --GET /model/...-->       nginx --> model_server   (debugging only)
```

Verified via direct `curl` against each path and via full browser sessions
(homepage load, search, poster rendering) through `http://localhost` — not
just the individual container ports. This distinction matters: an earlier
verification pass surfaced a real bug where the app worked when the frontend
container was accessed directly on `:3001` but failed through nginx's `:80`
entry point, because the backend's CORS allow-list did not include the
`http://localhost` origin nginx presents to the browser. Root cause and fix
are recorded in git history (commit `82227ad`) and are a useful worked
example for the thesis of why testing must exercise the _documented_ entry
point, not just any working port.

## 3. Model Inference Flow

**Status: Implemented, Tested.**

```
POST /api/v1/recommend/... (backend)
  --> RecommendationService (production/backend/app/services/)
        --> HybridEngine.recommend()  (production/recommenders/hybrid/)
              alpha(n_interactions) = min(1.0, n / N_WARMUP)
              score = alpha * NCF_score + (1 - alpha) * cold_start_score
        --> strategy label: "ncf" | "blend" | "cold_start"
GET /api/v1/search (backend)
  --> SearchService --> MovieRepository (DB text search)
        --> strategy label: "search"
```

`HybridEngine` is loaded once at `backend` startup (`RecommendationService.load()`)
and again independently inside `model_server` for the standalone inference
endpoint — the two services do not share a running process, only the same
artifact files on disk. `model_server`'s `/predict` and `/batch` endpoints
exist and are health-checked, but were not exercised under real recommendation
traffic in this session (no authenticated user session was driven through it
end-to-end) — noted as a verification gap, not a code gap.

## 4. Observability Flow

**Status: Implemented, Tested.**

```
backend, model_server  --(instrument_app, prometheus-client)-->  /metrics
                                          |
                                          v
                                    prometheus (:9090)
                                    scrapes backend:8000/metrics,
                                    model_server:8001/metrics every 15s
                                          |
                                          v
                                     grafana (:3002)
                                     datasource + dashboards
                                     auto-provisioned on startup
```

### 4.1 Metrics

**Status: Implemented, Tested.**

| Metric                                                                      | Type          | Labels                          | Source                                                                                                 |
| --------------------------------------------------------------------------- | ------------- | ------------------------------- | ------------------------------------------------------------------------------------------------------ |
| `http_requests_total`                                                       | Counter       | `service, method, path, status` | generic HTTP middleware, both services                                                                 |
| `http_request_duration_seconds`                                             | Histogram     | `service, method, path`         | generic HTTP middleware, both services                                                                 |
| `recommendation_strategy_total`                                             | Counter       | `service, strategy`             | added specifically to distinguish `ncf` / `blend` / `cold_start` / `search` traffic (commit `99d32bc`) |
| `up`                                                                        | Gauge         | `job`                           | generated automatically by Prometheus per scrape target, not app code                                  |
| `process_cpu_seconds_total`, `process_resident_memory_bytes`, `python_gc_*` | Counter/Gauge | —                               | `prometheus_client`'s built-in process/runtime collectors, no app code                                 |

No metric in either dashboard was invented without a corresponding
implementation — where a panel from the original design brief (e.g. per-model
inference-only latency, isolated from `/health`/`/metrics` scrape noise) had
no real supporting data, the panel queries the closest real metric
(`model_server`'s full request latency, which currently includes its own
healthcheck and scrape traffic since no authenticated inference traffic was
generated during testing) rather than fabricating a number.

### 4.2 Dashboards

**Status: Implemented, Tested, Provisioned.**

Both dashboards are checked into
`deployment/monitoring/grafana/dashboards/*.json` and auto-load via
`deployment/monitoring/grafana/provisioning/` — verified by a fresh Grafana
login showing both dashboards with no manual import step.

**System Overview** (`system-overview.json`): request rate, error rate (by
status code), P50/P95/P99 latency for `backend` and `model_server`
separately, CPU, memory, and a per-service `up`/`down` stat panel.

**Recommender Performance** (`recommender-performance.json`): recommendation
request rate by endpoint, strategy distribution and share (pie chart) from
`recommendation_strategy_total`, recommend-endpoint latency and error rate,
model server inference latency, and running totals by strategy.

### 4.3 Verified failure scenarios

**Status: Tested.**

Two distinct failure classes were deliberately triggered and confirmed
visible end-to-end (request → metric → Prometheus → Grafana panel):

1. **Application-level error** — `GET /api/v1/recommend/movie/999999`
   (nonexistent ID) correctly returned a structured `404` via the
   `AppError`/`NotFoundError` handler, and appeared within one Prometheus
   scrape interval as `http_requests_total{status="404"}` and on the
   Recommender Performance dashboard's error-rate panel with the specific
   path labeled.
2. **Infrastructure-level failure** — `docker compose stop backend` produced
   an immediate connection failure (not a hang), Prometheus reported
   `up{job="backend"} = 0` within one scrape interval, and Grafana's Service
   Availability panel flipped `backend` to red `DOWN` while correctly leaving
   `model_server` green `UP` — confirming per-service, not global, failure
   attribution. `docker compose start backend` restored `Up (healthy)` status
   and the dashboard returned to green within ~15 seconds.

### 4.4 Logging

**Status: Implemented (structured, stdout only). No log aggregation.**

Both services use `production/observability/logging/setup.py` for structured
log output to stdout, captured via `docker compose logs`. No centralized log
store (Loki, ELK, or similar) is deployed — logs are only as durable as the
container's log driver retains them. Listed as **Future work**.

### 4.5 Alerting

**Status: Not implemented.**

`deployment/observability/alerts/` referenced in the original scaffolding
audit does not contain active alert rules. Grafana's alerting subsystem is
available (visible in its left nav) but no alert rules were configured.
Listed as **Future work** in `ROADMAP.md`.

---

## 5. Summary Table

| Component                                                     | Implemented | Tested |             Configured              |      Optional       | Future work |
| ------------------------------------------------------------- | :---------: | :----: | :---------------------------------: | :-----------------: | :---------: |
| Docker Compose (5 services)                                   |     ✅      |   ✅   |                                     |                     |             |
| Kubernetes                                                    |             |        |                                     | ✅ (reference only) |             |
| CORS / nginx routing                                          |     ✅      |   ✅   |                                     |                     |             |
| `/health` endpoints                                           |     ✅      |   ✅   |                                     |                     |             |
| HTTP metrics (`http_requests_total`, latency histogram)       |     ✅      |   ✅   |                                     |                     |             |
| Recommendation strategy metric                                |     ✅      |   ✅   |                                     |                     |             |
| Prometheus                                                    |     ✅      |   ✅   |                                     |                     |             |
| Grafana + auto-provisioned dashboards                         |     ✅      |   ✅   |                                     |                     |             |
| Application-level failure visibility (404 test)               |     ✅      |   ✅   |                                     |                     |             |
| Infrastructure-level failure visibility (container stop test) |     ✅      |   ✅   |                                     |                     |             |
| `web`/`nginx` healthchecks                                    |             |        |                                     |                     |     ✅      |
| `model_server` inference under real (non-synthetic) traffic   |             |        | ✅ (endpoint exists, not exercised) |                     |             |
| Centralized log aggregation                                   |             |        |                                     |                     |     ✅      |
| Alerting rules                                                |             |        |                                     |                     |     ✅      |
| Free-tier cloud deployment                                    |             |        |           ✅ (documented)           |         ✅          |             |
