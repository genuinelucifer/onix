# Onix Model Evaluation

Run [EleutherAI's lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) benchmarks on Onix models.

## Prerequisites

Install `lm-evaluation-harness` into your environment:

```bash
source onix_env/bin/activate
pip install lm-eval
# — or for the latest dev version —
pip install -e /path/to/lm-evaluation-harness
```

## Usage

```bash
# From the project root:
python evals/run_eval.py \
    --model onix \
    --model_args model_name=<model_name>,dtype=bfloat16 \
    --device cuda:0 \
    --tasks <task_or_group> \
    --batch_size 32
```

### Arguments

| Argument | Description |
|---|---|
| `model_name` | Name of the model directory under `models/` (e.g. `llama1b-8192-v6`) |
| `dtype` | Weight dtype: `bfloat16`, `float16`, `float32` |
| `checkpoint_tag` | Checkpoint to load (default: `latest`) |
| `--device` | PyTorch device (default: `cuda:0`) |
| `--tasks` | Benchmark task or group (e.g. `blimp`, `hellaswag`, `arc_easy`) |
| `--batch_size` | Batch size for evaluation. Use an integer (e.g. `32`) or `auto` |

### Examples

```bash
# BLiMP (linguistic acceptability, 67 subtasks)
python evals/run_eval.py \
    --model onix \
    --model_args model_name=llama1b-8192-v6,dtype=bfloat16 \
    --device cuda:0 \
    --tasks blimp \
    --batch_size 32

# HellaSwag
python evals/run_eval.py \
    --model onix \
    --model_args model_name=llama1b-8192-v6,dtype=bfloat16 \
    --tasks hellaswag \
    --batch_size 16

# Multiple tasks
python evals/run_eval.py \
    --model onix \
    --model_args model_name=llama1b-8192-v6,dtype=bfloat16 \
    --tasks blimp,hellaswag,arc_easy \
    --batch_size 16
```

## How It Works

- **`run_eval.py`** — Entry point. Sets up `sys.path`, enables ROCm experimental flash attention, imports the model wrapper to register it, then delegates to `lm_eval`'s CLI.
- **`lm_eval_onix.py`** — Registers an `"onix"` model type with `lm_eval`. Loads the model from `models/<model_name>/checkpoint_latest.pt` using the Onix architecture, tokenizes with tiktoken GPT-2 BPE, and implements `loglikelihood`, `loglikelihood_rolling`, and `generate_until`.
