# `production/gateway-optional/`

An **optional** JWT-proxy FastAPI gateway, currently
**not wired into `docker-compose.yml`** — the default stack uses plain
`nginx` as the gateway/reverse proxy instead (see `deployment/nginx/`).
Moved from `gateway/fastapi_gateway/` in Phase 4; renamed to make its
non-default status visible from the path itself.

## Status
Reference implementation, not active in the default stack. Kept in the
repo as an example of an application-layer gateway (auth proxying, rate
limiting) for anyone who wants that instead of / in addition to nginx.

If you want to actually run it: it is not currently included in
`docker-compose.yml`'s service list — you'd need to add it manually
(open an issue/PR if you'd like this promoted to a first-class service).

See Phase 1 audit finding §3.2 and `docs/decisions/` for the reasoning.
