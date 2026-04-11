#!/usr/bin/env bash
# Run YALLM training in the background.
#
# LLM:        ./run_train.sh my-llama --preset llama-1b
# LLM Config: ./run_train.sh my-model --config configs/custom.json
# VQ-VAE:     ./run_train.sh my-vqvae --mode vqvae --config configs/vqvae_default.json --data-dir /path/to/images/
# MultiModal: ./run_train.sh my-imggen --mode multimodal --config configs/multimodal_pixelart.json --data-dir /path/to/pairs/
# Resume:     ./run_train.sh my-model --resume
# Override:   EPOCHS=50 DATA=myfile.txt ./run_train.sh my-gpt --preset llama-1b
#
# Monitor:    tail -f $MODELS_DIR/<model_name>/status.txt
# Stop:       kill $(cat $MODELS_DIR/<model_name>/.pid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source ~/repos/pytorch_env/bin/activate

# Force experimental Flash/Mem-Eff Attention on AMD Consumer GPUs
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

MODEL_NAME="${1:?Usage: $0 <model_name> [--mode <llm|vqvae|multimodal>] [--preset <preset> | --config <file>] [extra args]}"
shift

# Default args (override via env vars)
DATA="${DATA:-../the-verdict.txt}"
EPOCHS="${EPOCHS:-10}"
SAVE_EVERY="${SAVE_EVERY:-5}"
EVAL_FREQ="${EVAL_FREQ:-50}"
LOG_FREQ="${LOG_FREQ:-5}"

MODELS_DIR="${YALLM_MODELS_DIR:-$SCRIPT_DIR/models}"

# Build args list
ARGS=(--model-name "$MODEL_NAME")

# Only add defaults if not resuming and not in vqvae/multimodal mode
if [[ " $* " != *" --resume "* ]]; then
    # Check if mode is vqvae or multimodal (they use --data-dir, not --data)
    if [[ " $* " != *" --mode vqvae "* ]] && [[ " $* " != *" --mode multimodal "* ]]; then
        # LLM mode: add data source unless already specified
        if [[ " $* " != *" --data "* ]] && [[ " $* " != *" --data-dir "* ]]; then
            ARGS+=(--data "$DATA")
        fi
    fi
fi
ARGS+=(--epochs "$EPOCHS" --save-every "$SAVE_EVERY" --eval-freq "$EVAL_FREQ" --log-freq "$LOG_FREQ")

echo "Starting training in background..."
echo "  model_name=$MODEL_NAME"
echo "  args: ${ARGS[*]} $*"
echo "  Monitor: tail -f $MODELS_DIR/$MODEL_NAME/status.txt"

mkdir -p "$MODELS_DIR/$MODEL_NAME"

nohup python -u train.py "${ARGS[@]}" "$@" \
    > "$MODELS_DIR/$MODEL_NAME/stdout.log" 2> "$MODELS_DIR/$MODEL_NAME/stderr.log" &

echo $! > "$MODELS_DIR/$MODEL_NAME/.pid"
echo "PID: $(cat "$MODELS_DIR/$MODEL_NAME/.pid")"
echo "To stop: kill \$(cat $MODELS_DIR/$MODEL_NAME/.pid)"

