# Developer Getting Started

## Prerequisites
- Docker + Docker Compose
- Python 3.11+ (for running things outside Docker / IDE type-checking)
- Node 20+ (for `production/frontend`)

## Quick start (Docker — recommended)

```bash
git clone https://github.com/rahulkp-ai/NCF-Recommender-System.git
cd NCF-Recommender-System
cp .env.example .env        # fill in real values, especially TMDB_API_KEY
docker compose up --build
```

- Frontend: http://localhost:3000
- Backend API docs: http://localhost:8000/docs
- Model server: http://localhost:8001/health

## Local (non-Docker) setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pre-commit install
cp .env.example .env

# backend
uvicorn production.backend.app.main:app --reload --port 8000

# model server (separate terminal)
uvicorn production.serving.app.main:app --reload --port 8001

# frontend (separate terminal)
cd production/frontend && npm install && npm run dev
```

## Running tests

```bash
make test        # or: pytest
make lint         # ruff check . && ruff format --check . && mypy backend ml model_server
```

## Working on research

Research has its own, separate environment (kept independent per
`ARCHITECTURE.md`):

```bash
cd research
pip install -r requirements.txt
jupyter lab   # notebooks/
```

## Common gotchas

- If `model_server` fails to import `ml.models.ncf`, check that you're
  running it via `docker compose` or from the repo root — the current
  `sys.path` handling (see Phase 1 audit §3.9) is path-sensitive until the
  `pyproject.toml` installable-package migration lands.
- `.env.example`'s `TMDB_API_KEY` placeholder must be replaced with a real
  key from themoviedb.org for poster images to load.
