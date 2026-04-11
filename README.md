# YALLM Pretraining & Architecture Suite

YALLM is a comprehensive, high-performance training suite designed for building modern text-only transformers (GPT-2, LLaMA, Mistral, etc.) and autoregressive generative image models (VQ-VAE tokenizers + Multimodal generation).

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Ensure you are using the correct virtual environment
source ~/repos/pytorch_env/bin/activate
cd ~/repos/yallm/myllms
```

### 2. Available Modalities & Examples

The framework uses a unified `.sh` tracking structure. Depending on the modality, simply refer to a pre-defined architecture preset or a custom JSON config.

**A. Standard Text LLMs**
Run standard transformer training optimized for memory via 8-bit `adamw` bitsandbytes implementations.
```bash
./run_train.sh my-gpt --model-size 124M --data ../the-verdict.txt
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

## 🧠 Core Package Features
The core intelligence for model routing lies in the `architecture/` code:
- `config.py`: The validation schema handling the diverse parameters.
- `layers.py`: Grouped Query Attentions, KV heads, and Positional Encodings (ALiBi/RoPE).
- `model.py`: Fast `CausalLM` assembly and recursive modular abstractions.
