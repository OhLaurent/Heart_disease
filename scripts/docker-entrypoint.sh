#!/usr/bin/env bash
set -euo pipefail

PORT=${PORT:-8000}
exec uvicorn heart_disease.api.app:app --host 0.0.0.0 --port "$PORT"
