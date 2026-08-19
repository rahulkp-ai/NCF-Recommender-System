FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/
# Install CPU-only PyTorch first — the default PyPI wheel bundles ~1.3GB
# of unused CUDA/cuDNN libraries irrelevant on Render's free tier (no GPU).
# Once torch is satisfied here, the base.txt install below skips re-resolving it.
RUN pip install --no-cache-dir torch==2.3.0 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements/prod.txt

COPY production/serving/        ./production/serving/
COPY production/recommenders/   ./production/recommenders/
COPY production/observability/  ./production/observability/
COPY production/shared/         ./production/shared/
COPY production/__init__.py     ./production/__init__.py

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["uvicorn", "production.serving.app.main:app", "--host", "0.0.0.0", "--port", "8001"]