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
- **Stop**: `kill $(cat models/<model_name>/.pid)`.

### Running Directly via Python
For debugging or interactive visibility:
```bash
python train.py --model-name my-model --preset llama-1b --data ../the-verdict.txt
```

### Resuming Training
To pick up from the latest checkpoint:
```bash
./run_train.sh my-model --resume
```
You can also increase the total epochs when resuming:
```bash
./run_train.sh my-model --resume --epochs 50
```

---

## 💾 Storage & Memory Management
### Changing Model Storage Location
If your root drive is full, you can store checkpoints on an external drive by setting an environment variable:
```bash
export YALLM_MODELS_DIR="/mnt/my_big_drive/models"
./run_train.sh my-model ...
```

### Checkpoint Retention
To save disk space, the trainer automatically keeps only the **top 2 most recent checkpoints** + the final model. A 1B parameter model requires ~4GB-10GB per checkpoint.

---

## 🧠 Architecture Implementation Details
The core logic resides in the `architecture/` package:
- `config.py`: The `ModelConfig` schema and presets.
- `layers.py`: Implementation of GQA, RoPE, ALiBi, and GLU activations.
- `model.py`: The `CausalLM` assembly and `TransformerBlock` logic.
- `generate.py`: Text generation with temperature, top-k, top-p, and repetition penalty.
