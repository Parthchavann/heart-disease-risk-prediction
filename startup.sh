#!/bin/bash
set -e

echo "=== Heart Disease Risk Prediction API ==="

MODEL_PATH="data/models/best_model.pkl"

if [ ! -f "$MODEL_PATH" ]; then
    echo "[startup] No trained model found. Running full pipeline..."
    python scripts/run_full_pipeline.py
    echo "[startup] Pipeline complete."
else
    echo "[startup] Model found at $MODEL_PATH. Skipping training."
fi

echo "[startup] Starting FastAPI server..."
exec uvicorn api.main:app --host 0.0.0.0 --port ${API_PORT:-8000}
