"""
production/observability/logging/setup.py

Single source of truth for logging configuration across production/
services. Both production/backend/app/core/logging.py and
production/serving's logging setup delegate here, rather than each
service defining its own `logging.basicConfig(...)` call (the backend
already had one pre-Phase-5; this consolidates it instead of copying it
a second time for `serving`, which had none).

Default format is human-readable text (good for `docker compose logs` /
local dev). Set LOG_FORMAT=json in the environment to switch to
single-line JSON (good for shipping to a log aggregator — Render/Railway
both capture stdout, and JSON lines are easy to query there or in
Grafana Loki if that's ever added, see deployment/monitoring/).
"""

import json
import logging
import os
import sys
from datetime import UTC, datetime


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_logging(service_name: str, level: str | None = None) -> logging.Logger:
    """Configure root logging once and return a named logger for the caller.

    Idempotent-ish: calling this more than once (e.g. once from
    core/logging.py's module import and once explicitly) just re-applies
    the same handler config rather than stacking duplicate handlers,
    because we clear existing handlers on the root logger first.
    """
    level = level or os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "text").lower()

    root = logging.getLogger()
    root.handlers.clear()

    handler = logging.StreamHandler(stream=sys.stdout)
    if log_format == "json":
        handler.setFormatter(_JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )

    root.addHandler(handler)
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    return logging.getLogger(service_name)
