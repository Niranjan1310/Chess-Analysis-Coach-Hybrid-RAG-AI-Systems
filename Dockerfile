# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Install system dependencies including Stockfish chess engine
RUN apt-get update && apt-get install -y --no-install-recommends \
    stockfish \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy pyproject.toml and install Python dependencies
COPY backend/pyproject.toml /app/backend/
RUN pip install --no-cache-dir -e /app/backend

# Copy backend, frontend, data, ingestion, and evals
COPY backend/ /app/backend/
COPY frontend/ /app/frontend/
COPY data/ /app/data/
COPY ingestion/ /app/ingestion/
COPY evals/ /app/evals/

WORKDIR /app/backend

ENV PYTHONPATH="/app/backend:/app"
ENV STOCKFISH_PATH="/usr/games/stockfish"
ENV DATABASE_URL="sqlite:////app/data/chessmind.db"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
