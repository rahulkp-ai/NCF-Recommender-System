"""
production/serving/app/routes.py
/predict   — single-user inference
/batch     — multi-user batch inference
/health    — liveness check
"""

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from .inference import batch_inference, run_inference

router = APIRouter()


class PredictRequest(BaseModel):
    user_id: int
    seen_items: list[int] | None = []
    k: int | None = 10


class BatchPredictRequest(BaseModel):
    requests: list[PredictRequest]


@router.post("/predict")
def predict(payload: PredictRequest, request: Request):
    engine = request.app.state.engine
    try:
        results = run_inference(engine, payload.user_id, payload.seen_items, payload.k)
        return {"user_id": payload.user_id, "recommendations": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@router.post("/batch")
def batch(payload: BatchPredictRequest, request: Request):
    engine = request.app.state.engine
    raw = [{"user_id": r.user_id, "seen_items": r.seen_items, "k": r.k} for r in payload.requests]
    results = batch_inference(engine, raw)
    return {"results": results}


@router.get("/health")
def health(request: Request):
    loaded = hasattr(request.app.state, "engine")
    return {"status": "ok", "engine_loaded": loaded}
