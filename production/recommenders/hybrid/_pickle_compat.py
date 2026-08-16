"""
production/recommenders/hybrid/_pickle_compat.py

PopularityEngine and ContentEngine were pickled during research runs
against the module path `hybrid.cold_start` (see
docs/decisions/0002-hybrid-engine-refactor.md — this predates even the
pre-Phase-4 `ml/models/ncf/cold_start.py` location). Phase 4 further
split those two classes into separate sibling packages:
  - production/recommenders/popularity/engine.py
  - production/recommenders/content/engine.py

This module re-exports both under one namespace so that
HybridEngine._register_pickle_shim() (in engine.py) can alias
`sys.modules["hybrid.cold_start"]` to this module, and `pickle.load()`
can still resolve `hybrid.cold_start.PopularityEngine` /
`hybrid.cold_start.ContentEngine` for existing .pkl artifacts without
requiring a re-train.
"""

from ..content.engine import ContentEngine
from ..popularity.engine import PopularityEngine

__all__ = ["PopularityEngine", "ContentEngine"]
