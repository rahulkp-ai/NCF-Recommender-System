FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl gcc && \
    rm -rf /var/lib/apt/lists/*

COPY requirements/ ./requirements/
RUN pip install --no-cache-dir -r requirements/prod.txt

COPY production/serving/        ./production/serving/
COPY production/recommenders/   ./production/recommenders/
COPY production/observability/  ./production/observability/
COPY production/shared/         ./production/shared/
COPY production/__init__.py     ./production/__init__.py

ENV PYTHONPATH=/app
EXPOSE 8001

CMD ["uvicorn", "production.serving.app.main:app", "--host", "0.0.0.0", "--port", "8001"]