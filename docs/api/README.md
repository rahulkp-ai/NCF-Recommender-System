# API Documentation

The backend exposes an OpenAPI schema automatically via FastAPI at
`/docs` (Swagger UI) and `/redoc` when running locally
(`http://localhost:8000/docs` with the default `docker-compose.yml` port
mapping).

This directory is the place for hand-written API documentation that
doesn't belong in the auto-generated schema — versioning notes, auth flow
explanation, rate-limit policy, and example request/response payloads for
the main endpoints (`/api/auth/*`, `/api/recommendations/*`,
`/api/search/*`, `/api/users/*`).

*Content to be filled in during Phase 5 (Production Cleanup), once the
service layer and schemas are finalized — documenting an endpoint before
its request/response shape is stable would go stale immediately.*
