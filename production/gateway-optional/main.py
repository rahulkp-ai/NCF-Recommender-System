"""
gateway/fastapi_gateway/main.py
Lightweight API Gateway — validates JWT and proxies to backend.
For production, nginx handles this; this gateway is optional middleware.
"""

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response

from .middleware import jwt_middleware

BACKEND_URL = "http://backend:8000"

app = FastAPI(title="NCF Gateway", docs_url=None)
app.middleware("http")(jwt_middleware)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
async def proxy(path: str, request: Request):
    url = f"{BACKEND_URL}/{path}"
    params = dict(request.query_params)
    headers = dict(request.headers)
    headers.pop("host", None)
    body = await request.body()

    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.request(
                method=request.method,
                url=url,
                params=params,
                headers=headers,
                content=body,
            )
            return Response(
                content=resp.content,
                status_code=resp.status_code,
                headers=dict(resp.headers),
                media_type=resp.headers.get("content-type"),
            )
        except httpx.ConnectError:
            raise HTTPException(status_code=503, detail="Backend unavailable") from None
