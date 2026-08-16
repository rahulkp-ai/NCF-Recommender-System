"""
production/backend/app/main.py
"""

import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# NOTE: this file moved from backend/app/main.py to
# production/backend/app/main.py in Phase 4 — that's one directory
# deeper than before, so parents[2] (which pointed at repo root
# pre-move) now points at production/. Updated to parents[3].
# This was the #1 flagged risk in the Phase 1 audit (§3.9) — verified,
# not assumed: production/backend/app/main.py -> parents[0]=app,
# [1]=backend, [2]=production, [3]=repo root.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from production.observability.metrics import instrument_app
from production.shared.exceptions.errors import AppError

from .api.v1 import auth, interaction, recommend, search, users
from .core.config import CORS_ORIGINS
from .core.logging import logger, setup_logging
from .db.seed import run_seed
from .services.recommendation_service import RecommendationService

setup_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — seeding database...")
    run_seed()
    logger.info("Loading recommendation engine...")
    app.state.rec_service = RecommendationService.load()
    logger.info("Server ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="NCF Recommendation API",
    description="Neural Collaborative Filtering — Hybrid Recommender",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS + ["http://localhost:3000", "http://web:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

instrument_app(app, service_name="backend")


# Phase 5: global handler for the AppError hierarchy (production/shared/
# exceptions/) so services/repositories can raise domain-meaningful
# errors (NotFoundError, ModelServerUnavailableError, etc.) without
# knowing they're being called over HTTP — this handler is the one place
# that translates them into a consistent JSON error shape + status code.
@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("AppError on %s %s: %s", request.method, request.url.path, exc.message)
    return JSONResponse(status_code=exc.status_code, content=exc.to_dict())


app.include_router(auth.router)
app.include_router(users.router)
app.include_router(recommend.router)
app.include_router(search.router)
app.include_router(interaction.router)


@app.get("/health", tags=["health"])
def health():
    return {
        "status": "ok",
        "models_loaded": hasattr(app.state, "rec_service"),
    }


@app.get("/", tags=["health"])
def root():
    return {"message": "NCF Recommendation API", "docs": "/docs"}
