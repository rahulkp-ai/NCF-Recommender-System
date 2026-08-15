"""gateway/fastapi_gateway/middleware.py"""
from fastapi import Request
from fastapi.responses import JSONResponse

# Public paths that don't need JWT
PUBLIC_PATHS = {
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/recommend/homepage",
    "/api/v1/recommend/trending",
    "/api/v1/recommend/popular",
    "/api/v1/search",
    "/health",
    "/",
    "/docs",
    "/redoc",
    "/openapi.json",
}


async def jwt_middleware(request: Request, call_next):
    path = request.url.path
    if any(path.startswith(p) for p in PUBLIC_PATHS):
        return await call_next(request)
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        return JSONResponse({"detail": "Not authenticated"}, status_code=401)
    return await call_next(request)
