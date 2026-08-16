"""
research/backend/app/api/compare.py

Serves comparison.json to the dashboard frontend.
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/compare", tags=["comparison"])
LOG_PATH = Path("research/evaluation/comparison.json")


@router.get("/data")
def get_comparison_data():
    """Return the latest training comparison data."""
    if not LOG_PATH.exists():
        raise HTTPException(
            status_code=404, detail="comparison.json not found. Run: python -m evaluation.compare"
        )
    return json.loads(LOG_PATH.read_text())


@router.get("/summary")
def get_summary():
    """Return just the headline numbers for the metric cards."""
    data = get_comparison_data()
    return {
        "avg_speedup": data["comparison"]["avg_speedup"],
        "loss_delta": data["comparison"]["final_loss_delta"],
        "hit_delta": data["comparison"]["best_hit_delta"],
        "scratch_hit": data["scratch"]["best_hit"],
        "pytorch_hit": data["pytorch"]["best_hit"],
        "total_speedup": data["comparison"]["total_speedup"],
    }
