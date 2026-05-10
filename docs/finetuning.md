# YALLM Instruction Fine-Tuning Pipeline

This document details the pipeline used to instruction fine-tune a base LLM using the `TinyStoriesInstruct` dataset.

## 1. Download Dataset from HuggingFace
The dataset is downloaded using the custom HuggingFace downloader script. It pulls the dataset as raw text blocks containing JSON-like entries.

```bash
python download_hf.py --dataset tiny-stories-instruct
```

## 2. Preprocess into JSONL Format
The raw downloaded format (`train.json`) cannot be streamed efficiently because it contains fragmented text blocks. We run a preprocessing script to reconstruct the data into a standard Alpaca format (Instruction, Input, Output) and output it as a `.jsonl` (JSON Lines) file.

```bash
python utils/preprocess_tinystories_instruct_data.py \
    --input datasets/tiny-stories-instruct/train.json \
    --output datasets/tiny-stories-instruct/instruction-data.jsonl
```

**Why JSONL?** 
A standard 2.7GB `.json` array loads entirely into memory when parsed with `json.load()`. Due to Python's dictionary overhead, this causes CPU RAM to spike to **15-20 GB**, which can crash 32GB systems. By outputting `.jsonl`, the training script can stream the file line-by-line, tokenize the text instantly into a zero-overhead `array.array('i')`, and store the entire 375-million token dataset in just **1.5 GB of CPU RAM**.

## 3. Fine-Tune the Base Model
Once the `.jsonl` data is prepared, launch the fine-tuning run using the background worker script. This command utilizes the same performance optimizations (BF16, Torch Compile, DataLoader Prefetching) as the pre-training script.

```bash
./run_finetune.sh llama1b-sft \
    --base-model llama1b-8192-v6 \
    --data datasets/tiny-stories-instruct/instruction-data.jsonl \
    --batch-size 32 \
    --bf16 \
    --compile \
    --num-workers 4 \
    --prefetch-factor 2 \
    --eval-freq 500 \
    --save-iters 1000
```

*Note: For fine-tuning, `batch-size` can be significantly increased (e.g., to 32) compared to pre-training because instruction/response lengths are generally much shorter than the full 8192 context window.*

## 4. Monitor Training
Track the epoch progress, loss, and hardware status using the status monitor:

```bash
./train_status.sh llama1b-sft
```
