# YALLM Pretraining & Architecture Suite

A flexible, high-performance training suite for building modern decoder-only transformers (GPT-2, LLaMA, Mistral, etc.) from scratch on large-scale datasets.

## 🚀 Quick Start
### 1. Environment Setup
```bash
# Ensure you are using the correct virtual environment
source ~/repos/pytorch_env/bin/activate
cd ~/repos/yallm/myllms
```

### 2. Smoke Test (Small Dataset)
```bash
# Train a 124M parameter model on a small text file
./run_train.sh my-gpt --model-size 124M --data ../the-verdict.txt
```

---

## 📚 Pretraining Datasets
We support a sharded tokenization pipeline to handle datasets with billions of tokens efficiently.

### List Available Datasets
```bash
python -m pretrain_data.download --list
```

### Download & Pre-tokenize
This will stream the dataset from HuggingFace, tokenize it, and save it as memory-mapped `.npy` shards.
```bash
# Download TinyStories (approx. 470M tokens) - Recommended for testing
python -m pretrain_data.download --dataset tiny-stories

# Download FineWeb-Edu 10B token sample (default)
python -m pretrain_data.download

# Download specific amount
python -m pretrain_data.download --dataset fineweb-edu-10bt --max-tokens 1_000_000_000
```

---

## 🎒 Instruction & Fine-tuning Datasets
For supervised fine-tuning (SFT) or other tasks (Vision, etc.), use the general downloader:

### List Common Datasets
```bash
python download_hf.py --list
```

### Download Popular SFT Datasets
SFT datasets are automatically converted to clean JSON for easy use with `finetune.py`.
```bash
# Alpaca (clean 52k instructions)
python download_hf.py --dataset alpaca

# Dolly-15k (human-generated data)
python download_hf.py --dataset dolly-15k

# OpenOrca (large-scale augmented data)
python download_hf.py --dataset open-orca
```

### Vision & Other Datasets
```bash
# Tiny-ImageNet for small-scale vision training
python download_hf.py --dataset tiny-imagenet

# COCO-2017 for detection/captioning
python download_hf.py --dataset coco-2017
```

---

## 🏗️ Model Architectures & Presets
YALLM supports configurable architectures including:
- **Attention**: Multi-Head (MHA), Multi-Query (MQA), and Grouped-Query (GQA).
- **Position Encodings**: RoPE (Rotary), ALiBi, and Learned absolute.
- **Activations**: GELU, SwiGLU, GeGLU, ReGLU.
- **Normalization**: RMSNorm, LayerNorm.

### Using Architectural Presets
Instead of manual configuration, use one of our optimized presets:
```bash
# Available presets: gpt2-124m, llama-1b, llama-3b, mistral-1b, gptj-1b
./run_train.sh my-llama --preset llama-1b

# Full Example: Train a Llama-1B model on TinyStories dataset
./run_train.sh llama-stories --preset llama-1b --data-dir pretrain_data/tiny_stories/
```

### Custom Model Configuration
Create a JSON config file (e.g., `configs/my_arch.json`) to define every layer detail, then run:
```bash
./run_train.sh my-model --config configs/my_arch.json
```

---

## 🛠️ Training Models

### Running in the Background (Recommended)
Use the included bash script to run training in the background with `nohup`. It automatically handles logging and PID tracking.
```bash
./run_train.sh <model_name> --preset <preset_name> --data-dir <path_to_shards>
```
- **Logs**: Located in `models/<model_name>/stdout.log` and `stderr.log`.
- **Status**: Monitor `tail -f models/<model_name>/status.txt`.
- **Stop**: Use `./stop_train.sh <model_name>`.

### 🧠 Low-Memory (VRAM/RAM) Optimization
To train large models (1B+) on consumer GPUs or with limited system RAM, use our optimized 8-bit training configuration:

```bash
# Example: 1B Llama model optimized for low-memory & fast status updates
./run_train.sh llama-1b-fast \
    --preset llama-1b \
    --data-dir pretrain_data/tiny_stories/ \
    --optimizer adamw8bit \
    --batch-size 8 \
    --eval-freq 50 \
    --log-freq 1 \
    --save-iters 20
```

#### Key Parameters Explained:
*   **`--optimizer adamw8bit`**: Uses `bitsandbytes` to quantize optimizer states. Reduces memory usage for optimizer moments by ~75% (e.g., from 8GB down to 2GB for a 1B model).
*   **`--batch-size 8`**: Increases GPU utilization. Combined with 8-bit optimizers, this allows larger batches in the same VRAM.
*   **`--log-freq 1`**: Logs a "PROGRESS" line to `status.txt` every single step. Useful for monitoring slow iterations in real-time.
*   **`--eval-freq 50`**: Runs expensive validation (train/val loss averaging) only every 50 steps to avoid performance bottlenecks.
*   **`--save-iters 20`**: Saves a checkpoint every 20 steps. Crucial for avoiding lost work if training is interrupted.

---

## 🛑 Managing Background Runs
If you used `./run_train.sh`, the process runs in the background even if you close your terminal.

### Monitoring Progress
Use the `train_status.sh` script for a quick summary of progress, potential errors, and the current process state:
```bash
# General summary (Status + Errors + PID check)
./train_status.sh <model_name>

# Real-time view of status logs alone
tail -f models/<model_name>/status.txt
```

### Stopping Training Safely
Use the `stop_train.sh` script to cleanly terminate a model run using its recorded PID:
```bash
./stop_train.sh <model_name>
```

### Resuming Training
To pick up exactly where you left off after a crash or planned interruption:
```bash
./run_train.sh <model_name> --resume
```

**How it works:**
The script loads the latest checkpoint and extracts the `global_step` and `tokens_seen`. It then **fast-forwards** the data loader to skip any batches already processed. This ensures you continue with the next unseen sample in your dataset and that your learning rate schedule remains accurate.

---

## 💾 Storage & Memory Management
### Changing Model Storage Location
If your root drive is full, you can store checkpoints on an external drive by setting an environment variable:
```bash
export YALLM_MODELS_DIR="/mnt/my_big_drive/models"
./run_train.sh my-model ...
```

### Checkpoint Size & Retention
*   **Size**: A **llama-1b** checkpoint is approximately **6.7 GB** (4.4GB weights + 2.2GB 8-bit optimizer state).
*   **Retention**: To save disk space, the trainer automatically keeps only the **top 3 most recent checkpoints** + the final model.
*   **Budget**: Plan for ~30-40 GB of free space per active training run.

---

## 🧠 Architecture Implementation Details
The core logic resides in the `architecture/` package:
- `config.py`: The `ModelConfig` schema and presets.
- `layers.py`: Implementation of GQA, RoPE, ALiBi, and GLU activations.
- `model.py`: The `CausalLM` assembly and `TransformerBlock` logic.
- `generate.py`: Text generation with temperature, top-k, top-p, and repetition penalty.
