#!/usr/bin/env bash
# Run YALLM pretraining in the background.
#
# New run:    ./run_train.sh my-gpt
# Resume:    ./run_train.sh my-gpt --resume
# Override:  EPOCHS=50 DATA=myfile.txt ./run_train.sh my-gpt
#
# Monitor:   tail -f models/<model_name>/status.txt
# Stop:      kill $(cat models/<model_name>/.pid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv (lives in parent yallm/ dir)
source ~/repos/pytorch_env/bin/activate

MODEL_NAME="${1:?Usage: $0 <model_name> [extra args]}"
shift

# Default args (override via env vars)
DATA="${DATA:-../the-verdict.txt}"
EPOCHS="${EPOCHS:-10}"
MODEL_SIZE="${MODEL_SIZE:-124M}"
SAVE_EVERY="${SAVE_EVERY:-5}"

# Build args list
ARGS=(--model-name "$MODEL_NAME")

# Only add defaults if not resuming
if [[ " $* " != *" --resume "* ]]; then
    ARGS+=(--data "$DATA" --model-size "$MODEL_SIZE")
fi
ARGS+=(--epochs "$EPOCHS" --save-every "$SAVE_EVERY")

echo "Starting pretraining in background..."
echo "  model_name=$MODEL_NAME"
echo "  args: ${ARGS[*]} $*"
echo "  Monitor: tail -f models/$MODEL_NAME/status.txt"

mkdir -p "models/$MODEL_NAME"

nohup python -u train.py "${ARGS[@]}" "$@" \
    > "models/$MODEL_NAME/stdout.log" 2> "models/$MODEL_NAME/stderr.log" &

echo $! > "models/$MODEL_NAME/.pid"
echo "PID: $(cat "models/$MODEL_NAME/.pid")"
echo "To stop: kill \$(cat models/$MODEL_NAME/.pid)"
