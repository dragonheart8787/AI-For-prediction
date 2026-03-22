# syntax=docker/dockerfile:1

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy requirements (choose the most complete one if present)
COPY requirements_enhanced.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy project
COPY . /app

# Default environment
ENV APP_HOST=0.0.0.0 \
    APP_PORT=8080 \
    PYTHONIOENCODING=utf-8

# Expose web port (aiohttp web_server.py)
EXPOSE 8080

# Healthcheck (simple): web server endpoint
HEALTHCHECK --interval=30s --timeout=5s --retries=3 CMD curl -fsS http://localhost:8080/health || exit 1

# Default command: start unified launcher (can be overridden by Compose)
CMD ["python", "launch_system.py"]


