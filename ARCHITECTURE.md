# Architecture Overview

This document outlines the system architecture using Mermaid diagrams for clarity and conciseness. The design intentionally separates two independent worlds that never import from each other: `research/` and `production/`.

## Two-World Boundary

```text
research/    ──export_model.py / promote_dataset.py (planned)──►   production
   (never imported by production)                     (never imports research)
```

### Research → Production Boundary

```mermaid
flowchart LR
  subgraph R["Research World"]
    A["research/"]
    B["Training"]
    C["Evaluation"]
    D["Thesis & Paper"]
    E["Export Artifacts"]
  end

  subgraph P["Production World"]
    F["backend/"]
    G["model_server/"]
    H["frontend/"]
    I["ml/models/ncf"]
  end

  A --> B
  B --> C
  C --> D
  B --> E

  E -->|Weights / Datasets| F
  E -->|Model Artifacts| G
  E -->|Read-only Assets| I

  style R fill:#eef7ff
  style P fill:#f5fff2
```

**Rule:** Production never imports research code. Research only exports artifacts.

`research/` exists to produce artifacts: trained weights, processed datasets, evaluation results, the thesis, and the IEEE paper. It optimizes for reproducibility and experimentation speed, not runtime performance or API stability.

`production/` (`production/backend/`, `production/recommenders/`, `production/serving/`, `production/frontend/`, `production/gateway-optional/`) exists to serve those artifacts to real users. It optimizes for correctness, latency, and operability.

This split was verified in the Phase 1 audit by checking every `import` statement in the codebase — see [docs/decisions/0001-research-production-separation.md](docs/decisions/0001-research-production-separation.md).

### Component Map

| Component             | Responsibility                                        | Talks to                                     |
| --------------------- | ----------------------------------------------------- | -------------------------------------------- |
| `production/frontend` | Next.js UI                                            | `backend` (REST)                             |
| `backend`             | Auth, search, user data, orchestrates recommendations | `model_server`, Postgres                     |
| `model_server`        | Loads trained NCF/hybrid weights, serves predictions  | `ml` (in-process), `artifacts/ml/models/ncf` |
| Production ML Engine  | Hybrid engine code (no training)                      | `artifacts/` (read-only)                     |
| `research/*`          | Training, experimentation, evaluation, thesis, paper  | Nothing in production                        |

### Production Component Architecture

```mermaid
flowchart LR
  User((User))
  Gateway["Gateway (Optional)"]
  Frontend["Next.js Frontend"]
  Backend["Backend API"]
  Model["Model Server"]
  ML["Production ML\nNCF + Hybrid Engine"]
  Artifacts[("Model Artifacts")]
  Postgres[("PostgreSQL")]

  User --> Frontend
  Frontend --> Backend
  Gateway -.Optional.-> Backend
  Backend --> Postgres
  Backend --> Model
  Model --> ML
  ML --> Artifacts
```

### Request Flow

```mermaid
sequenceDiagram
  participant User
  participant Frontend
  participant Backend
  participant ModelServer
  participant ML
  participant Artifacts

  User->>Frontend: Search / Recommendation Request
  Frontend->>Backend: REST API
  Backend->>ModelServer: Predict()
  ModelServer->>Artifacts: Load trained weights
  ModelServer->>ML: Run Hybrid NCF
  ML-->>ModelServer: Ranked items
  ModelServer-->>Backend: Recommendations
  Backend-->>Frontend: JSON Response
  Frontend-->>User: Display Results
```

## Why the Hybrid Engine Has Two Copies (`research/` vs. `production/`)

Both `research/hybrid/` and `ml/models/ncf/` define `HybridEngine` and `ColdStartHandler`. This is **not accidental duplication**. The production version is a deliberate, hardened refactor of the research version (see [ADR 0002](docs/decisions/0002-hybrid-engine-hardening.md)).

### Evolution of the Hybrid Engine

```mermaid
flowchart LR
  Research["research/hybrid"]
  Refactor["Production Hardening"]
  Production["ml/models/ncf"]

  Research --> Refactor --> Production
  Research -.Independent Evolution.-> Research
  Production -.Independent Evolution.-> Production
```

They are allowed to evolve independently. Future contributors **should not merge them into a shared module**, as doing so would reintroduce coupling between research and production.

## Deployment Target

The primary deployment target is a **free-tier PaaS**:

- **Backend** → Render / Railway
- **Model Server** → Render / Railway
- **Frontend** → Vercel

Kubernetes and Terraform remain future deployment options.

### Deployment Architecture

```mermaid
flowchart TB
  User((Users))
  DNS["Internet"]
  Frontend["Vercel\nNext.js"]
  Backend["Render/Railway\nBackend API"]
  Model["Render/Railway\nModel Server"]
  DB[("PostgreSQL")]
  Artifacts[("Model Artifacts")]

  User --> DNS
  DNS --> Frontend
  Frontend --> Backend
  Backend --> DB
  Backend --> Model
  Model --> Artifacts
```

## Repository Architecture Overview

This diagram summarizes the complete architecture.

```mermaid
flowchart LR
  subgraph Research
    Train["Model Training"]
    Eval["Evaluation"]
    Export["Artifact Export"]
  end

  subgraph Production
    Frontend["Frontend"]
    Backend["Backend"]
    Model["Model Server"]
    ML["Production ML"]
    DB[("PostgreSQL")]
    Artifacts[("Artifacts")]
  end

  Train --> Eval --> Export
  Export --> Artifacts
  Frontend --> Backend
  Backend --> DB
  Backend --> Model
  Model --> ML
  ML --> Artifacts
```

## Further Reading

- [docs/architecture/overview.md](docs/architecture/overview.md) — deeper narrative walkthrough
- [docs/decisions/](docs/decisions/) — ADRs for specific architectural choices
- [docs/diagrams/system-architecture.md](docs/diagrams/system-architecture.md) — request-flow diagram
