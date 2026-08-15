"""
production/serving/app/main.py
Standalone FastAPI service for NCF inference.
Runs on port 8001; backend communicates via HTTP.
"""
import sys
from pathlib import Path
from contextlib import asynccontextmanager

# NOTE: this file moved from model_server/app/main.py to
# production/serving/app/main.py in Phase 4 — one directory deeper than
# before. parents[2] used to point at repo root; updated to parents[3].
# Second of the two highest-risk sys.path lines flagged in the Phase 1
# audit (§3.9). Verified: production/serving/app/main.py ->
# parents[0]=app, [1]=serving, [2]=production, [3]=repo root.
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .loader import load_hybrid_engine
from .routes import router
from production.observability.logging import configure_logging
from production.observability.metrics import instrument_app

logger = configure_logging("ncf_model_server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Model server: loading hybrid engine...")
    app.state.engine = load_hybrid_engine()
    logger.info("Model server ready — n_items: %d", app.state.engine.n_items)
    yield
    logger.info("Model server shutting down.")


app = FastAPI(
    title="NCF Model Server",
    description="Inference endpoint for the NCF Hybrid Recommender",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

instrument_app(app, service_name="model_server")

app.include_router(router)
