.PHONY: help up down build logs shell-backend shell-db seed test lint frontend-install frontend-dev clean

# ── Config ────────────────────────────────────────────────────────────────────
COMPOSE = docker compose
BACKEND = backend
WEB     = production/frontend

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ── Docker ────────────────────────────────────────────────────────────────────
up: ## Start all services (build if needed)
	@cp -n .env.example .env 2>/dev/null || true
	$(COMPOSE) up --build -d
	@echo "\n  Stack is running:"
	@echo "   Frontend  → http://localhost"
	@echo "   or                           "
	@echo "   Frontend  → http://localhost:3001"
	@echo "   Backend   → http://localhost:8000/docs"
	@echo "   Gateway   → http://localhost:80"

down: ## Stop all services
	$(COMPOSE) down

down-v: ## Stop all services and remove volumes
	$(COMPOSE) down -v

build: ## Rebuild all images
	$(COMPOSE) build

logs: ## Tail all logs
	$(COMPOSE) logs -f

logs-backend: ## Tail backend logs only
	$(COMPOSE) logs -f backend

logs-model: ## Tail model server logs
	$(COMPOSE) logs -f model_server

# ── Shells ────────────────────────────────────────────────────────────────────
shell-backend: ## Open bash in backend container
	$(COMPOSE) exec backend bash

shell-db: ## Open psql in database container
	$(COMPOSE) exec db psql -U ncf_user -d ncf_db

# ── Dev (no Docker) ───────────────────────────────────────────────────────────
install: ## Install Python deps
	pip install -r requirements/dev.txt

frontend-install: ## Install frontend npm deps
	cd $(WEB) && npm install

backend-dev: ## Run backend locally (requires local Postgres + .env)
	cd . && uvicorn production.backend.app.main:app --reload --port 8000

model-dev: ## Run model server locally
	cd . && uvicorn production.serving.app.main:app --reload --port 8001

frontend-dev: ## Run Next.js dev server
	cd $(WEB) && npm run dev

# ── Database ──────────────────────────────────────────────────────────────────
seed: ## Seed database (runs inside backend container)
	$(COMPOSE) exec backend python -m production.backend.app.db.seed

posters: ## Fetch TMDB posters (requires TMDB_API_KEY in .env)
	$(COMPOSE) exec backend python -m production.backend.app.services.tmdb_service

# ── Tests ─────────────────────────────────────────────────────────────────────
test: ## Run all tests
	$(COMPOSE) exec backend pytest production/tests/ -v --tb=short

test-local: ## Run tests locally
	pytest production/tests/ -v --tb=short

# ── Lint ──────────────────────────────────────────────────────────────────────
lint: ## Lint Python code
	python -m flake8 production/backend/ production/recommenders/ production/serving/ --max-line-length=120 --ignore=E501,W503

# ── Cleanup ───────────────────────────────────────────────────────────────────
clean: ## Remove __pycache__ and .pyc files
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
