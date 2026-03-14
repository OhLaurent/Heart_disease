#!/usr/bin/env bash
set -euo pipefail

export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////app/mlflow.db}"
export GIT_PYTHON_REFRESH=quiet
PORT=${PORT:-8000}

# Ensure the MLflow SQLite file and mlruns directory exist and are writable
DB_PATH="${MLFLOW_TRACKING_URI#sqlite:///}"
mkdir -p "$(dirname "$DB_PATH")" /app/mlruns
touch "$DB_PATH"

exec uvicorn heart_disease.api.app:app --host 0.0.0.0 --port "$PORT"
