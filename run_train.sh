#!/usr/bin/env bash
# Run YALLM pretraining in the background.
#
# Preset:     ./run_train.sh my-llama --preset llama-1b
# Config:     ./run_train.sh my-model --config configs/custom.json
# Legacy:     ./run_train.sh my-gpt --model-size 124M
# Resume:     ./run_train.sh my-gpt --resume
# Shards:     ./run_train.sh my-llama --preset llama-1b --data-dir pretrain_data/fineweb_edu_10bt/
# Override:   EPOCHS=50 DATA=myfile.txt ./run_train.sh my-gpt --preset llama-1b
#
# Monitor:    tail -f $MODELS_DIR/<model_name>/status.txt
# Stop:       kill $(cat $MODELS_DIR/<model_name>/.pid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source ~/repos/pytorch_env/bin/activate

MODEL_NAME="${1:?Usage: $0 <model_name> [--preset <preset> | --config <file> | --model-size <size>] [extra args]}"
shift

# Default args (override via env vars)
DATA="${DATA:-../the-verdict.txt}"
EPOCHS="${EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:-5}"

MODELS_DIR="${YALLM_MODELS_DIR:-$SCRIPT_DIR/models}"

# Build args list
ARGS=(--model-name "$MODEL_NAME")

# Only add defaults if not resuming
if [[ " $* " != *" --resume "* ]]; then
    # Add data source (unless --data or --data-dir already specified)
    if [[ " $* " != *" --data "* ]] && [[ " $* " != *" --data-dir "* ]]; then
        ARGS+=(--data "$DATA")
    fi
fi
ARGS+=(--epochs "$EPOCHS" --save-every "$SAVE_EVERY")

echo "Starting pretraining in background..."
echo "  model_name=$MODEL_NAME"
echo "  args: ${ARGS[*]} $*"
echo "  Monitor: tail -f $MODELS_DIR/$MODEL_NAME/status.txt"

mkdir -p "$MODELS_DIR/$MODEL_NAME"

nohup python -u train.py "${ARGS[@]}" "$@" \
    > "$MODELS_DIR/$MODEL_NAME/stdout.log" 2> "$MODELS_DIR/$MODEL_NAME/stderr.log" &

echo $! > "$MODELS_DIR/$MODEL_NAME/.pid"
echo "PID: $(cat "$MODELS_DIR/$MODEL_NAME/.pid")"
echo "To stop: kill \$(cat $MODELS_DIR/$MODEL_NAME/.pid)"
