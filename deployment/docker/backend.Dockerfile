FROM python:3.11-slim

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl libpq-dev gcc && \
    rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/prod.txt

# Source
COPY production/backend/        ./production/backend/
COPY production/recommenders/   ./production/recommenders/
COPY production/observability/  ./production/observability/
COPY production/shared/         ./production/shared/
COPY production/__init__.py     ./production/__init__.py

ENV PYTHONPATH=/app
EXPOSE 8000

CMD ["uvicorn", "production.backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]