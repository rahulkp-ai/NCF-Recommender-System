# Integration tests

Cross-component tests, e.g. exercising the FastAPI app's exception
handlers end-to-end (`test_error_handling.py`, added in Phase 5), or
`backend` calling the real `model_server` over HTTP — as opposed to
`production/tests/unit/test_ml.py`'s in-process unit tests.

Requires a live Postgres (`docker compose up db`) — same requirement as
`production/tests/unit/test_api.py`.
