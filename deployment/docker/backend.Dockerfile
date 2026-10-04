FROM python:3.14-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements/ ./requirements/
# Install CPU-only PyTorch first — the default PyPI wheel bundles ~1.3GB
# of unused CUDA/cuDNN libraries irrelevant on Render's free tier (no GPU).
# Once torch is satisfied here, the base.txt install below skips re-resolving it.
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements/prod.txt

# Source
COPY production/backend/        ./production/backend/
COPY production/recommenders/   ./production/recommenders/
COPY production/observability/  ./production/observability/
COPY production/shared/         ./production/shared/
COPY production/__init__.py     ./production/__init__.py
COPY production/artifacts/       ./production/artifacts/

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "production.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]