#!/usr/bin/env bash
# Run YALLM training in the background.
#
# LLM:        ./run_train.sh my-llama --mode llm --preset llama-1b
# LLM Config: ./run_train.sh my-model --mode llm --config configs/custom.json
# VQ-VAE:     ./run_train.sh my-vqvae --mode vqvae --config configs/vqvae_default.json --data-dir /path/to/images/
# MultiModal: ./run_train.sh my-imggen --mode multimodal --config configs/multimodal_pixelart.json --data-dir /path/to/pairs/
# Resume:     ./run_train.sh my-model --resume
#
# Monitor:    tail -f $MODELS_DIR/<model_name>/status.txt
# Stop:       kill $(cat $MODELS_DIR/<model_name>/.pid)

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Activate venv
source ~/repos/pytorch_env/bin/activate


MODEL_NAME="${1:?Usage: $0 <model_name> [--mode <llm|vqvae|multimodal>] [--preset <preset> | --config <file>] [extra args]}"
shift

MODELS_DIR="$SCRIPT_DIR/models"

# Force experimental Flash/Mem-Eff Attention on AMD Consumer GPUs
# (Breaks VQ-VAE Convs, so we skip it for VQ-VAE mode)
IS_VQVAE=0
if [[ " $* " == *" --mode vqvae "* ]] || [[ "$MODEL_NAME" == *"vqvae"* ]]; then
    IS_VQVAE=1
fi
if [[ " $* " == *" --resume "* ]] && [ -f "$MODELS_DIR/$MODEL_NAME/config.json" ]; then
    if grep -q '"model_type": "vqvae"' "$MODELS_DIR/$MODEL_NAME/config.json" 2>/dev/null; then
        IS_VQVAE=1
    fi
fi

if [ "$IS_VQVAE" -eq 0 ]; then
    export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1
fi

# Safety check: is model already training?
if [ -f "$MODELS_DIR/$MODEL_NAME/.pid" ]; then
    PID=$(cat "$MODELS_DIR/$MODEL_NAME/.pid")
    if kill -0 "$PID" 2>/dev/null; then
        echo "ERROR: Training is already running for '$MODEL_NAME' (PID $PID)."
        echo "Check status: tail -f $MODELS_DIR/$MODEL_NAME/status.txt"
        echo "To stop it first: kill $PID"
        exit 1
    fi
fi

# Build args list
ARGS=(--model-name "$MODEL_NAME")

echo "Starting training in background..."
echo "  model_name=$MODEL_NAME"
echo "  args: ${ARGS[*]} $*"
echo "  Monitor: tail -f $MODELS_DIR/$MODEL_NAME/status.txt"

mkdir -p "$MODELS_DIR/$MODEL_NAME"

# Persist torch.compile cache so it survives reboots (default /tmp/ is cleared)
export TORCHINDUCTOR_CACHE_DIR="$SCRIPT_DIR/.torch_cache"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"

nohup python -u train.py "${ARGS[@]}" "$@" \
    > "$MODELS_DIR/$MODEL_NAME/stdout.log" 2> "$MODELS_DIR/$MODEL_NAME/stderr.log" &

echo $! > "$MODELS_DIR/$MODEL_NAME/.pid"
echo "PID: $(cat "$MODELS_DIR/$MODEL_NAME/.pid")"
echo "To stop: kill \$(cat $MODELS_DIR/$MODEL_NAME/.pid)"

