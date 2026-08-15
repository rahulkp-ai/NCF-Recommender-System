"""
production/observability/metrics/middleware.py

Phase 5: minimal request-metrics instrumentation. This exists specifically
to close a gap discovered during this phase — Phase 2 generated
deployment/monitoring/prometheus/prometheus.yml configured to scrape
/metrics on both backend:8000 and model_server:8001, but no such
endpoint existed anywhere. That config was explicitly marked as
"reference, not deployed by default" (see its own README), so this stays
equally minimal: a request counter and latency histogram, no new
external dependencies beyond `prometheus-client` (a single small,
well-known library — not a Prometheus *server*, just the text-format
exposition helper).

Usage (in a FastAPI app):
    from production.observability.metrics.middleware import instrument_app
    instrument_app(app, service_name="backend")
"""
import time

from fastapi import FastAPI, Request
from starlette.responses import Response
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["service", "method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["service", "method", "path"]
)


def instrument_app(app: FastAPI, service_name: str) -> None:
    """Adds a timing middleware + a /metrics endpoint to `app`."""

    @app.middleware("http")
    async def _metrics_middleware(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start
        path = request.url.path
        REQUEST_COUNT.labels(service_name, request.method, path, response.status_code).inc()
        REQUEST_LATENCY.labels(service_name, request.method, path).observe(duration)
        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics():
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
