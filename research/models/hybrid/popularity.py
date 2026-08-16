"""
research/hybrid/popularity.py

Thin wrapper that loads and caches the popularity engine from disk.
Used by the FastAPI recommendation service in Phase 7.
"""

import pickle
from pathlib import Path

import pandas as pd

from .cold_start import ContentEngine, PopularityEngine

CACHE_DIR = Path("research/datasets/processed")


def build_and_save_engines(
    interactions_path: str = "research/datasets/processed/train.csv",
    movies_path: str = "research/datasets/raw/movielens/ml-1m/movies.dat",
) -> tuple[PopularityEngine, ContentEngine]:
    """
    Build popularity and content engines from processed data and save to disk.
    Run once after Phase 2 data pipeline completes.
    """
    # Load data
    interactions = pd.read_csv(interactions_path)
    movies = pd.read_csv(
        movies_path,
        sep="::",
        engine="python",
        names=["item_id", "title", "genres"],
        encoding="latin-1",
    )

    # Fit engines
    pop_engine = PopularityEngine().fit(interactions)
    content_engine = ContentEngine().fit(movies)

    # Cache to disk
    with open(CACHE_DIR / "pop_engine.pkl", "wb") as f:
        pickle.dump(pop_engine, f)
    with open(CACHE_DIR / "content_engine.pkl", "wb") as f:
        pickle.dump(content_engine, f)

    print("Engines saved to data/processed/")
    return pop_engine, content_engine


def load_engines() -> tuple[PopularityEngine, ContentEngine]:
    """Load pre-built engines from disk (fast, used by API at startup)."""
    with open(CACHE_DIR / "pop_engine.pkl", "rb") as f:
        pop = pickle.load(f)
    with open(CACHE_DIR / "content_engine.pkl", "rb") as f:
        content = pickle.load(f)
    return pop, content
