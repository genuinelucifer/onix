#!/usr/bin/env bash
# Aggressively stop a background training run.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${YALLM_MODELS_DIR:-$SCRIPT_DIR/models}"

MODEL_NAME="${1:?Usage: $0 <model_name>}"
PID_FILE="$MODELS_DIR/$MODEL_NAME/.pid"

if [ ! -f "$PID_FILE" ]; then
    echo "Error: No .pid file found for model '$MODEL_NAME' (Expected $PID_FILE)"
    exit 1
fi

PID=$(cat "$PID_FILE")

if ps -p "$PID" > /dev/null; then
    echo "Stopping model '$MODEL_NAME' (PID $PID)..."
    kill "$PID"
    echo "Sent termination signal. Any recent progress since the last checkpoint is discarded."
else
    echo "Model '$MODEL_NAME' (PID $PID) is not currently running."
fi
