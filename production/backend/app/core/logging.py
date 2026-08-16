"""
production/backend/app/core/logging.py

Phase 5: delegates to production/observability/logging/ (single source
of truth for logging config across backend + serving) instead of
duplicating logging.basicConfig(...) here. `setup_logging()` and
`logger` keep their original names/signatures so every existing
`from .core.logging import ...` / `from ..core.logging import ...`
import elsewhere in the backend keeps working unchanged.
"""

from production.observability.logging import configure_logging

logger = configure_logging("ncf_api")


def setup_logging(level: str = "INFO") -> None:
    """Kept for backward compatibility with existing call sites
    (production/backend/app/main.py calls this at import time)."""
    global logger
    logger = configure_logging("ncf_api", level=level)
