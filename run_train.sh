#!/usr/bin/env bash
# Run YALLM pretraining in the background.
# Usage: ./run_train.sh [extra args for train.py]
# Monitor: tail -f train_status.txt
# Stop:    kill $(cat .train.pid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Default args (override via env vars)
DATA="${DATA:-the-verdict.txt}"
EPOCHS="${EPOCHS:-10}"
MODEL="${MODEL:-124M}"

echo "Starting pretraining in background..."
echo "  data=$DATA  epochs=$EPOCHS  model=$MODEL"
echo "  Monitor progress: tail -f train_status.txt"

nohup python train.py \
    --data "$DATA" \
    --model-size "$MODEL" \
    --epochs "$EPOCHS" \
    "$@" \
    > train_stdout.log 2>&1 &

echo $! > .train.pid
echo "PID: $(cat .train.pid)"
echo "To stop: kill \$(cat .train.pid)"
