# System Architecture Diagram

```
                          ┌─────────────────┐
                          │   production/frontend│
                          │  (Next.js UI)    │
                          └────────┬─────────┘
                                   │ REST (JSON)
                                   ▼
                    ┌──────────────────────────┐
                    │   backend (FastAPI)       │
                    │  auth · search · users    │
                    └──────┬─────────────┬──────┘
                            │             │
                 in-process │             │ REST
                  (planned) │             ▼
                            │   ┌──────────────────────┐
                            │   │  model_server          │
                            │   │  (FastAPI, inference)  │
                            │   └──────────┬─────────────┘
                            │              │ loads at startup
                            │              ▼
                            │   ┌──────────────────────┐
                            │   │  ml/models/ncf         │
                            │   │  HybridEngine,         │
                            │   │  ColdStartHandler       │
                            │   └──────────┬─────────────┘
                            │              │ reads
                            │              ▼
                            │   ┌──────────────────────┐
                            ▼   │  artifacts/            │
                    ┌───────────┤  production_model.pt   │
                    │ Postgres  │  (trained weights only) │
                    └───────────┴────────────────────────┘

     (optional, not wired in by default — see ADR "gateway" note)
     ┌──────────────────────┐
     │  gateway/fastapi_gateway │ ── JWT proxy in front of backend
     └──────────────────────┘

     ═══════════ research/ (isolated — never imported above) ═══════════
     notebooks → experiments/expNNN_* → evaluation → (manual copy today,
     scripts/export_model.py planned) → artifacts/ + data/
```

Rendered as text/ASCII intentionally so it stays version-control-friendly
and diffable; a Mermaid or draw.io version can be added under this same
path if preferred later.
