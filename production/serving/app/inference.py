"""
production/serving/app/inference.py
Pure inference helpers — no HTTP, no DB.
"""
import numpy as np


def run_inference(engine, user_id: int, seen_items: list[int], k: int = 10) -> list[dict]:
    """Run hybrid recommendation inference and return ranked list."""
    return engine.recommend(user_id=user_id, seen_items=seen_items, k=k)


def batch_inference(engine, requests: list[dict]) -> list[list[dict]]:
    """
    Process multiple recommendation requests.
    requests: [{"user_id": int, "seen_items": [...], "k": int}, ...]
    """
    return [
        run_inference(engine, r["user_id"], r.get("seen_items", []), r.get("k", 10))
        for r in requests
    ]
