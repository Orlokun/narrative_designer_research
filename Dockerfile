FROM python:3.12-slim

# Copy uv binary from the official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# System deps needed by trafilatura, pdfplumber, spacy, etc.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2 \
    libxslt1.1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# Install dependencies first (cached layer — only re-runs if pyproject.toml or uv.lock changes)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copy source and install the project itself
COPY . .
RUN uv sync --frozen --no-dev

# data/ is expected to be a mounted volume; create placeholder dirs
RUN mkdir -p data/archivo data/chroma data/datasets data/logs \
             data/checkpoints data/gatekeeper_cache

# Default: admin dashboard. Override with docker compose command.
CMD ["uv", "run", "admin"]
