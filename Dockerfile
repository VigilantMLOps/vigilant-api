FROM python:3.12-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Tell Poetry to install into the system Python (no venv needed inside a container)
ENV POETRY_VIRTUALENVS_CREATE=false
ENV POETRY_NO_INTERACTION=1

WORKDIR /app

# Install system dependencies (needed for DuckDB/Health checks)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir "poetry>=2.0.0,<3.0.0"

# Install dependencies before copying source for better layer caching
COPY pyproject.toml poetry.lock ./
RUN poetry install --no-root --without dev

# Copy the application
COPY . .

# Create directories for persistent data and make entrypoint executable
RUN mkdir -p core/database core/logs && chmod +x entrypoint.sh

# Expose FastAPI port
EXPOSE 8000

ENTRYPOINT ["./entrypoint.sh"]