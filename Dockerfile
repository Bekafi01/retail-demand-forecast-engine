# Multi-stage Dockerfile optimized with uv package manager
FROM python:3.11-slim AS base

# Install curl and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set working directory and environment variables
WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=0 \
    UV_LINK_MODE=copy \
    MLFLOW_ALLOW_FILE_STORE=true \
    MLFLOW_TRACKING_URI=sqlite:///mlflow.db

# Copy dependency manifests
COPY pyproject.toml uv.lock .python-version ./

# Synchronize dependencies with uv (frozen locked dependencies)
RUN uv sync --frozen --no-dev

# Copy application source code
COPY src/ /app/src/
COPY mlops/ /app/mlops/
COPY api/ /app/api/
COPY ui/ /app/ui/
COPY configs/ /app/configs/
COPY README.md /app/

# Expose ports: 8000 for FastAPI, 8501 for Streamlit, 5000 for MLflow
EXPOSE 8000 8501 5000

# Default target runs FastAPI
CMD ["uv", "run", "uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
