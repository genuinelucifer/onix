# YALLM Instruction Fine-Tuning Pipeline

This document details the pipeline used to instruction fine-tune a base LLM using the `TinyStoriesInstruct` dataset.

## 1. Download Dataset from HuggingFace
The dataset is downloaded using the custom HuggingFace downloader script. It pulls the dataset as raw text blocks containing JSON-like entries.

```bash
python download_hf.py --dataset tiny-stories-instruct
```

## 2. Preprocess into JSONL Format
The raw downloaded format (`train.json`) contains fragmented text blocks. We run a preprocessing script to reconstruct the data into a standard Alpaca format (Instruction, Input, Output) and output it as a `.jsonl` file.

```bash
python utils/preprocess_tinystories_instruct_data.py \
    --input datasets/tiny-stories-instruct/train.json \
    --output datasets/tiny-stories-instruct/instruction-data.jsonl
```

## 3. Pre-tokenize into Binary Shards (Recommended)
For large datasets (e.g., 2.7GB JSONL), tokenizing in-memory during training startup can cause massive RAM spikes due to Python's object overhead. To achieve near-zero RAM startup and instant loading, convert the JSONL into binary `.npy` shards.

```bash
python utils/preprocess_sft.py \
    --input datasets/tiny-stories-instruct/instruction-data.jsonl \
    --output datasets/tiny-stories-instruct/processed
```

**Why Binary?**
Binary shards are memory-mapped (`mmap`). This allows the training script to stream tokens directly from disk to VRAM without using CPU RAM. It also bypasses the slow tokenization loop, making the script start training almost instantly.

## 4. Fine-Tune the Base Model
Launch the fine-tuning run using the background worker script. If you pre-tokenized the data in Step 3, point `--data` to the prefix of your binary files.

```bash
./run_finetune.sh llama1b-sft \
    --base-model llama1b-8192-v6 \
    --data datasets/tiny-stories-instruct/processed \
    --batch-size 32 \
    --bf16 \
    --compile \
    --checkpointing \
    --num-workers 4
```

*Note: If you haven't pre-tokenized, you can still pass the `.jsonl` file directly, but be prepared for a longer startup time and higher RAM usage.*

## 5. Monitor Training
Track progress, loss, and hardware status:

```bash
./train_status.sh llama1b-sft
```
