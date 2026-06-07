# ═══════════════════════════════════════════════════════════════
# mentorX AI — Production Dockerfile
# CPU-only | uv | multi-stage | python 3.11
# ═══════════════════════════════════════════════════════════════

# ── Stage 1: Builder ───────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv (fast package manager used in dev)
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Install build deps needed to compile native extensions
# (chromadb, numpy, sentence-transformers need gcc)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first → Docker cache layer
COPY requirements.txt pyproject.toml ./

# Install all deps into /app/.venv using uv
# --no-cache keeps the layer small
RUN uv venv .venv && \
    uv pip install --no-cache -r requirements.txt

# ── Stage 2: Runtime ───────────────────────────────────────────
FROM python:3.11-slim AS runtime

# Env flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    # Keep HuggingFace models in a mounted cache volume
    TRANSFORMERS_CACHE=/app/.cache/huggingface \
    SENTENCE_TRANSFORMERS_HOME=/app/.cache/sentence_transformers \
    # Tell Python to use the venv
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Only curl needed at runtime (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the compiled venv from builder — no gcc/build tools in runtime
COPY --from=builder /app/.venv /app/.venv

# Copy application source (structure unchanged from your repo)
COPY api/           ./api/
COPY config/        ./config/
COPY dataIngestion/ ./dataIngestion/
COPY retrieval/     ./retrieval/
COPY vectorStore/   ./vectorStore/
COPY utils/         ./utils/
COPY main.py        .

# Create mount points for volumes (data never baked into image)
RUN mkdir -p \
    /app/dataIngestion/pdf_data \
    /app/vectorStore/chroma_db \
    /app/.cache

EXPOSE 8000

# Docker/K8s healthcheck — hits /health which is a cheap endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://127.0.0.1:8000/health || exit 1

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
