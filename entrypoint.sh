#!/bin/sh
set -e

echo "🚀 Running database initialization and seeding..."
# We keep 'init' if your db_manager handles that argument
python -m core.db_manager init

echo "🌐 Starting FastAPI server on port ${PORT:-8000}..."
exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}