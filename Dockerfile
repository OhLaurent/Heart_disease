FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy project files (use .dockerignore to omit large/dev-only files)
COPY . /app

RUN python -m pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir ".[dev]"

RUN chmod +x /app/scripts/docker-entrypoint.sh || true

EXPOSE 8000

ENTRYPOINT ["/app/scripts/docker-entrypoint.sh"]
