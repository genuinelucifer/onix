#!/usr/bin/env bash
# Run YALLM instruction fine-tuning in the background.
#
# New run:    ./run_finetune.sh my-sft
# Resume:    ./run_finetune.sh my-sft --resume
# Override:  EPOCHS=5 DATA=custom.json ./run_finetune.sh my-sft
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
DATA="${DATA:-../instruction-data.json}"
EPOCHS="${EPOCHS:-2}"
MODEL_SIZE="${MODEL_SIZE:-355M}"
SAVE_EVERY="${SAVE_EVERY:-1}"

# Build args list
ARGS=(--model-name "$MODEL_NAME")

# Only add defaults if not resuming
if [[ " $* " != *" --resume "* ]]; then
    ARGS+=(--data "$DATA" --model-size "$MODEL_SIZE")
fi
ARGS+=(--epochs "$EPOCHS" --save-every "$SAVE_EVERY")

MODELS_DIR="${YALLM_MODELS_DIR:-$SCRIPT_DIR/models}"

echo "Starting instruction fine-tuning in background..."
echo "  model_name=$MODEL_NAME"
echo "  args: ${ARGS[*]} $*"
echo "  Monitor: tail -f $MODELS_DIR/$MODEL_NAME/status.txt"

mkdir -p "$MODELS_DIR/$MODEL_NAME"

nohup python -u finetune.py "${ARGS[@]}" "$@" \
    > "$MODELS_DIR/$MODEL_NAME/stdout.log" 2> "$MODELS_DIR/$MODEL_NAME/stderr.log" &

echo $! > "$MODELS_DIR/$MODEL_NAME/.pid"
echo "PID: $(cat "$MODELS_DIR/$MODEL_NAME/.pid")"
echo "To stop: kill \$(cat $MODELS_DIR/$MODEL_NAME/.pid)"
