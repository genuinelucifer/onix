#!/usr/bin/env bash
# Monitor the status of a background training run.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="$SCRIPT_DIR/models"

MODEL_NAME="${1:?Usage: $0 <model_name>}"
MODEL_PATH="$MODELS_DIR/$MODEL_NAME"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory '$MODEL_PATH' does not exist."
    exit 1
fi

echo "=========================================================="
echo " MODEL: $MODEL_NAME"
echo "=========================================================="

# 1. Print status.txt (latest progress)
STATUS_FILE="$MODEL_PATH/status.txt"
if [ -f "$STATUS_FILE" ]; then
    echo "--- [LATEST STATUS] ---"
    tail -n 10 "$STATUS_FILE"
else
    echo "Status file not found."
fi

# 2. Print stderr if not empty (errors/warnings)
STDERR_FILE="$MODEL_PATH/stderr.log"
if [ -s "$STDERR_FILE" ]; then
    echo ""
    echo "--- [ERRORS / LOGS (stderr)] ---"
    cat "$STDERR_FILE"
fi

# 3. Check if process is alive
PID_FILE="$MODEL_PATH/.pid"
echo ""
echo "--- [PROCESS STATUS] ---"
if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p "$PID" > /dev/null; then
        echo "Status: ALIVE (PID: $PID)"
    else
        echo "Status: DEAD (Process not found)"
    fi
else
    echo "Status: UNKNOWN (No .pid file found)"
fi
echo "=========================================================="
