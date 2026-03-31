#!/usr/bin/env bash
# Run YALLM instruction fine-tuning in the background.
# Usage: ./run_finetune.sh [extra args for finetune.py]
# Monitor: tail -f finetune_status.txt
# Stop:    kill $(cat .finetune.pid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source .venv/bin/activate

# Default args (override via env vars)
DATA="${DATA:-instruction-data.json}"
EPOCHS="${EPOCHS:-2}"
MODEL="${MODEL:-355M}"

echo "Starting instruction fine-tuning in background..."
echo "  data=$DATA  epochs=$EPOCHS  model=$MODEL"
echo "  Monitor progress: tail -f finetune_status.txt"

nohup python finetune.py \
    --data "$DATA" \
    --model-size "$MODEL" \
    --epochs "$EPOCHS" \
    "$@" \
    > finetune_stdout.log 2>&1 &

echo $! > .finetune.pid
echo "PID: $(cat .finetune.pid)"
echo "To stop: kill \$(cat .finetune.pid)"
