# Production Dockerfile for Hugging Face Spaces (Docker SDK)
FROM python:3.11-slim

# Install system dependencies (curl for healthchecks, git, build-essential if needed)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user matching Hugging Face Spaces convention (uid 1000)
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    IS_DOCKER=1

# Set working directory
WORKDIR /app

# Install Python dependencies first for caching layers
COPY --chown=user:user requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt && \
    echo "Using pre-quantized ONNX model from repository" 

# Copy application source code and pre-built index data (data/qdrant_db and data/bm25.pkl)
COPY --chown=user:user . /app

# Ensure /tmp cache and data directories exist and are writable
RUN chown -R user:user /app

# Switch to non-root user
USER user

# Hugging Face Spaces default port
EXPOSE 10000

# Start FastAPI server on port 7860
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000}
