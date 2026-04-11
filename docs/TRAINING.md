# YALLM Training Guide

YALLM supports configurable architectures and automatically routes the unified logic under `train.py` to target the correct modality.

## Modality Dispatcher
The `train.py` dispatcher routes to:
- `--mode llm`: Trains a standard text-only transformer (GPT/LLama).
- `--mode vqvae`: Phase 1 Image Training. Trains a VQ-VAE model for continuous-to-discrete spatial image compression (tokenizer).
- `--mode multimodal`: Phase 2 Image Generation. Trains an autoregressive transformer to generate spatial image tokens from text descriptions.

## 1. Standard LLM Training

Text-based models can be initialized using pre-optimized presets or custom `configs/*.json` configurations.

```bash
# Using a 1B LLaMA preset
./run_train.sh my-llama --preset llama-1b --data-dir pretrain_data/tiny_stories/

# Using a low-memory 8-bit config structure for heavy iterations
./run_train.sh llama-1b-fast \
    --preset llama-1b \
    --data-dir pretrain_data/tiny_stories/ \
    --optimizer adamw8bit \
    --batch-size 8 \
    --eval-freq 50 \
    --save-iters 20
```

## 2. VQ-VAE Training (Vision)

The first step to building generative architectures is training the spatial Tokenizer. The VQ-VAE requires the flat output directory from the `preprocess_image_dataset.py` step.

```bash
./run_train.sh my-vqvae \
    --mode vqvae \
    --config configs/vqvae_250m.json \
    --data-dir datasets/pixelart_processed/ \
    --epochs 100 \
    --save-every 5 \
    --save-iters 500
```
This freezes the configuration inside `models/my-vqvae/config.json`.

## 3. Multimodal Autoregression

Once your `vqvae_checkpoint` is generated, ensure it's referenced accurately within your combined Multimodal JSON config (`configs/multimodal_pixelart.json`). Then execute training on your paired captions:

```bash
./run_train.sh my-multimodal \
    --mode multimodal \
    --config configs/multimodal_pixelart.json \
    --data-dir datasets/pixelart_processed/ \
    --epochs 50 \
    --save-every 5
```

---

## Managing Background Training Runs

Running scripts locally via `./run_train.sh` utilizes `nohup` to detach and keep running locally regardless of your SSH/Terminal connection.

### Monitoring Output
```bash
# Rich terminal output including process status
./train_status.sh <model_name>

# Real-time tailing of log
tail -f models/<model_name>/status.txt
```

### Stopping Training Safely
Instead of struggling to find background PIDs, cleanly terminate a model's run using:
```bash
./stop_train.sh <model_name>
```

### Resuming Work
YALLM remembers your state and fast-forwards the dataloaders back to exactly the iteration where you stopped. The `train.py` dispatcher will automatically realize which `--mode` you were in.
```bash
./run_train.sh <model_name> --resume
```

## Checkpoint Retention & Settings
- **Retention**: To save extreme disk space, `train.py` automatically maintains only the **last 3 checkpoints** and deletes older ones automatically.
- Ensure your `YALLM_MODELS_DIR` environment variable is linked to a heavy-capacity drive if training massively scaled variants.
