#!/usr/bin/env bash
# Run YALLM instruction fine-tuning in the background.
#
# Usage:
#   ./run_finetune.sh my-sft --data instruction-data.json
#   ./run_finetune.sh my-sft --base-model llama1b-8192-v6 --data instruction-data.json
#   ./run_finetune.sh my-sft --resume
#   ./run_finetune.sh my-sft --resume --epochs 5
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

MODELS_DIR="$SCRIPT_DIR/models"

# Force experimental Flash/Mem-Eff Attention on AMD Consumer GPUs
export TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1

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

echo "Starting instruction fine-tuning in background..."
echo "  model_name=$MODEL_NAME"
echo "  args: $*"
echo "  Monitor: tail -f $MODELS_DIR/$MODEL_NAME/status.txt"

mkdir -p "$MODELS_DIR/$MODEL_NAME"

# Persist torch.compile and Triton caches so they survive reboots (default /tmp/ is cleared)
export TORCHINDUCTOR_CACHE_DIR="$SCRIPT_DIR/.torch_cache"
export TRITON_CACHE_DIR="$SCRIPT_DIR/.torch_cache/triton"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR"
mkdir -p "$TRITON_CACHE_DIR"

nohup python -u finetune.py --model-name "$MODEL_NAME" "$@" \
    > "$MODELS_DIR/$MODEL_NAME/stdout.log" 2> "$MODELS_DIR/$MODEL_NAME/stderr.log" &

echo $! > "$MODELS_DIR/$MODEL_NAME/.pid"
echo "PID: $(cat "$MODELS_DIR/$MODEL_NAME/.pid")"
echo "To stop: kill \$(cat $MODELS_DIR/$MODEL_NAME/.pid)"
