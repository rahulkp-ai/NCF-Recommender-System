"""
research/models/hybrid/popularity.py

Thin wrapper that loads and caches the popularity engine from disk.
Used by the FastAPI recommendation service.
"""

import pickle
import pandas as pd
from pathlib import Path
from .cold_start import PopularityEngine, ContentEngine

RESEARCH_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = RESEARCH_DIR / "datasets" / "processed"


def _register_pickle_compat() -> None:
    """Resolve cached engines saved under the former research.hybrid module path."""
    import research

    from . import cold_start

    legacy_package = sys.modules.get("research.hybrid")
    if legacy_package is None:
        legacy_package = types.ModuleType("research.hybrid")
        legacy_package.__path__ = []
        sys.modules["research.hybrid"] = legacy_package

    legacy_package.cold_start = cold_start
    research.hybrid = legacy_package
    sys.modules["research.hybrid.cold_start"] = cold_start


def build_and_save_engines(
    interactions_path: str | Path = CACHE_DIR / "train.csv",
    movies_path:       str | Path = RESEARCH_DIR / "datasets" / "raw" / "movielens" / "ml-1m" / "movies.dat",
) -> tuple[PopularityEngine, ContentEngine]:
    """
    Build popularity and content engines from processed data and save to disk.
    Run once after Phase 2 data pipeline completes.
    """
    # Load data
    interactions = pd.read_csv(interactions_path)
    movies = pd.read_csv(
        movies_path, sep="::", engine="python",
        names=["item_id", "title", "genres"], encoding="latin-1"
    )

    # Fit engines
    pop_engine     = PopularityEngine().fit(interactions)
    content_engine = ContentEngine().fit(movies)

    # Cache to disk
    with open(CACHE_DIR / "pop_engine.pkl", "wb") as f:
        pickle.dump(pop_engine, f)
    with open(CACHE_DIR / "content_engine.pkl", "wb") as f:
        pickle.dump(content_engine, f)

    print(f"Engines saved to {CACHE_DIR}/")
    return pop_engine, content_engine


def load_engines() -> tuple[PopularityEngine, ContentEngine]:
    """Load pre-built engines from disk (fast, used by API at startup)."""
    with open(CACHE_DIR / "pop_engine.pkl",     "rb") as f:
        pop = pickle.load(f)
    with open(CACHE_DIR / "content_engine.pkl", "rb") as f:
        content = pickle.load(f)
    return pop, content