# YALLM Pretraining & Architecture Suite

YALLM is a comprehensive, high-performance training suite designed for building modern text-only transformers (GPT-2, LLaMA, Mistral, etc.) and autoregressive generative image models (VQ-VAE tokenizers + Multimodal generation).

## 🛠️ Developer Setup

Follow these steps to set up your local development environment:

### 1. Create a Python Virtual Environment
It is recommended to use Python 3.10+ for full compatibility with ROCm and PyTorch. Run the following commands to create your virtual environment inside the repository:

```bash
# Navigate to the workspace
cd onix

# Create the virtual environment
python3 -m venv onix_env
```

### 2. Enable/Activate the Virtual Environment
Activate the environment to ensure pip installs packages to this isolated sandbox:

```bash
# Activate the virtual environment
source onix_env/bin/activate
```

### 3. Install Project Dependencies from `requirements.txt`
Install all required libraries. The `requirements.txt` is pre-configured with the PyTorch ROCm index URL, so this single command will automatically fetch the ROCm-optimized PyTorch packages along with all data processing, tokenization, and web GUI tools:

```bash
# Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Quick Start

The project provides helper shell scripts (`./run_train.sh` and `./run_finetune.sh`) to easily launch and manage background training jobs. To train a model, you configure its architecture using either a pre-defined preset name or a custom JSON configuration file.

**A. Standard Text LLMs**
Run decoder-only transformer training optimized for high performance and low memory footprint.

*Example: Optimized LLaMA 1B pre-training (uses 512 context, gradient checkpointing, `bf16` mixed precision, and `torch.compile` kernel fusion)*
```bash
./run_train.sh my-llama \
    --mode llm \
    --config configs/llama1b_512_opt.json \
    --data ../the-verdict.txt \
    --bf16 \
    --compile
```

**B. Vision Models (VQ-VAE)**
Train the high-fidelity spatial tokenizer (Phase 1) on raw images:
```bash
./run_train.sh my-vqvae \
    --mode vqvae \
    --config configs/vqvae_250m.json \
    --data-dir datasets/pixelart_processed/ 
```

**C. Multimodal Autoregression**
Utilize your trained VQ-VAE alongside a custom context generation model to align and infer on text+image datasets (Phase 2):
```bash
./run_train.sh my-multimodal \
    --mode multimodal \
    --config configs/multimodal_pixelart.json \
    --data-dir datasets/pixelart_processed/ 
```

---

## 📚 Documentation Reference

For more detailed guides regarding dataset manipulation, large scale dataset sharding, low-memory footprints, and tracking background processes efficiently, please refer to:

- [**Data Preparation & Datasets (`docs/DATASETS.md`)**](docs/DATASETS.md)
   - Sharded pre-tokenizable pipelines (`fineweb`, `tinystories`)
   - Supervised Fine-Tuning parsing (`dolly`, `alpaca`)
   - Vision & Hugging Face raw dataset fallback downloading
   - Universal multimodality pre-processing (`preprocess_image_dataset.py`)

- [**Training & Ops Engine (`docs/TRAINING.md`)**](docs/TRAINING.md)
   - Using the Modality Dispatcher (`train.py`)
   - Stopping/Starting detached jobs safely via PID identifiers
   - Handling Auto-Resumption and continuous Fast-forwarding
   - Checkpoint retention logic and 8-bit memory optimizations

## 📖 Example: Training a Model on TinyStories

For a complete, step-by-step walkthrough on how to prepare the TinyStories dataset, configure the LLaMA 1B architecture, and launch optimized pre-training on your system, please refer to our dedicated guide:

- [**Training LLaMA 1B on TinyStories (`docs/training_llama1b_on_tinystories.md`)**](docs/training_llama1b_on_tinystories.md)
