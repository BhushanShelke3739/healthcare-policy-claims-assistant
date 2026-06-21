# =============================================================================
# Backend Dockerfile — Phase 1
#
# Single-stage image for development convenience. A multi-stage production
# image will be added in Phase 10 (smaller surface, no build toolchain).
# =============================================================================

FROM python:3.11-slim

# Don't write .pyc files, flush stdout immediately so logs appear in `docker logs`.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# Build deps for psycopg2 + a few diagnostic utilities. Kept minimal; we'll
# switch to psycopg[binary] or a multi-stage build later to drop these.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libpq-dev \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer.
COPY requirements.txt ./
RUN pip install -r requirements.txt

# Copy the rest of the backend source.
COPY . .

EXPOSE 8000

# Default command. docker-compose.yml overrides this with --reload for dev.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
