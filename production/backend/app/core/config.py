"""
production/backend/app/core/config.py
Application configuration via environment variables.

Phase 5 correction: ROOT_DIR's parents[] depth was broken by the Phase 4
move (same bug class as seed.py, see that file's docstring) — fixed from
parents[3] to parents[4]. DATA_DIR/ARTIFACTS were unused elsewhere in
the codebase at the time of this fix (verified via grep), but repointed
to production/artifacts to avoid leaving a broken public constant as a
footgun for future code.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT_DIR  = Path(__file__).resolve().parents[4]
ARTIFACTS = ROOT_DIR / "production" / "artifacts"
DATA_DIR  = ARTIFACTS  # kept as an alias — no separate data/ directory exists in production post-Phase-4

DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://ncf_user:ncf_pass@db:5432/ncf_db",
)

SECRET_KEY: str = os.getenv("SECRET_KEY", "change-me-super-secret-key-min-32-chars!!")
ALGORITHM: str  = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))

MODEL_SERVER_URL: str = os.getenv("MODEL_SERVER_URL", "http://model_server:8001")

TMDB_API_KEY: str = os.getenv("TMDB_API_KEY", "")

APP_ENV: str = os.getenv("APP_ENV", "development")
DEBUG: bool  = APP_ENV == "development"
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS", "http://localhost:3000,http://localhost:5173"
).split(",")
