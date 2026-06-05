#!/usr/bin/env python3
"""
General Purpose HuggingFace Dataset Downloader.
Supports Instruction Fine-tuning (SFT), Vision, and other common dataset types.

Usage:
  python download_hf.py --list
  python download_hf.py --dataset alpaca
  python download_hf.py --dataset dolly-15k --out-dir ./data/sft
  python download_hf.py --dataset tiny-imagenet --out-dir ./data/vision
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tiktoken
from datasets import load_dataset
from huggingface_hub import snapshot_download

DATASETS = {
    # --- Instruction Fine-tuning (SFT) ---
    "alpaca": {
        "path": "tatsu-lab/alpaca",
        "description": "52k instruction-following data used for the original Alpaca model.",
        "type": "sft"
    },
    "dolly-15k": {
        "path": "databricks/databricks-dolly-15k",
        "description": "15k human-generated instruction records by Databricks employees.",
        "type": "sft"
    },
    "open-orca": {
        "path": "Open-Orca/OpenOrca",
        "description": "Large-scale collection of augmented instruction data (GPT-4/3.5 outputs).",
        "type": "sft"
    },
    "ultrachat": {
        "path": "Stingning/UltraChat_sharegpt",
        "description": "High-quality multi-turn dialogue data.",
        "type": "sft"
    },
    "wizard-lm": {
        "path": "WizardLM/WizardLM_evol_instruct_70k",
        "description": "70k complex instructions generated via Evol-Instruct.",
        "type": "sft"
    },
    "tiny-stories-instruct": {
        "path": "roneneldan/TinyStoriesInstruct",
        "description": "Instruction fine-tuning dataset for TinyStories.",
        "type": "sft"
    },

    # --- Vision & Multimodal ---
    "tiny-imagenet": {
        "path": "MaySee/tiny-imagenet",
        "description": "Subset of ImageNet (200 classes, 100k images, 64x64 resolution).",
        "type": "vision"
    },
    "mnist": {
        "path": "mnist",
        "description": "Classic handwritten digits dataset.",
        "type": "vision"
    },
    "coco-2017": {
        "path": "detection-datasets/coco2017",
        "description": "Common Objects in Context - Object detection and captioning.",
        "type": "vision"
    },

    "diffusiondb-pixelart": {
        "path": "jainr3/diffusiondb-pixelart",
        "name": "2k_all",
        "description": "2k pixel-art images from DiffusionDB with text prompts.",
        "type": "vision"
    },
    "gz-evo": {
        "path": "mwalmsley/gz_evo_internal",
        "description": "Galaxy Zoo Evolution dataset (total ~18.3 GB).",
        "type": "vision"
    },
    "diffusiondb-large": {
        "path": "poloclub/diffusiondb",
        "description": "Large DiffusionDB dataset. Each shard is ~750MB.",
        "type": "vision",
        "shards": 70,  # Default to ~52 GB
        "shard_start": 1,
        "shard_pattern": "images/part-{i:06d}.zip",
        "extra_files": ["metadata.parquet", "dataset_info.json"]
    },
    "gz-decals": {
        "path": "BigBang/galaxyzoo-decals",
        "description": "Galaxy Zoo DECaLS dataset. Total ~105 GB. Images in tar.gz shards.",
        "type": "vision",
        "shards": 100, # Default to ~50 GB
        "shard_start": 0,
        "shard_pattern": "images/J{i:03d}.tar.gz",
        "extra_files": ["annotations/*.parquet"]
    },

    # --- Other ---
    "wikitext-103": {
        "path": "wikitext",
        "name": "wikitext-103-v1",
        "description": "Large-scale collection of high-quality Wikipedia articles.",
        "type": "text"
    },

    # --- Pretraining (Tokenized Shards) ---
    "fineweb-edu-10bt": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-10BT",
        "split": "train",
        "text_field": "text",
        "description": "FineWeb-Edu 10B token sample - high-quality educational web text.",
        "type": "text",
        "tokenized": True,
        "tokenizer": "gpt2",
        "approx_tokens": 10_000_000_000
    },
    "fineweb-edu-100bt": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "sample-100BT",
        "split": "train",
        "text_field": "text",
        "description": "FineWeb-Edu 100B token sample.",
        "type": "text",
        "tokenized": True,
        "tokenizer": "gpt2",
        "approx_tokens": 100_000_000_000
    },
    "slimpajama": {
        "path": "cerebras/SlimPajama-627B",
        "name": None,
        "split": "train",
        "text_field": "text",
        "description": "SlimPajama 627B - diverse cleaned web+book+code data.",
        "type": "text",
        "tokenized": True,
        "tokenizer": "gpt2",
        "approx_tokens": 627_000_000_000
    },
    "tiny-stories": {
        "path": "roneneldan/TinyStories",
        "name": None,
        "split": "train",
        "text_field": "text",
        "description": "TinyStories - small dataset for testing (~470M tokens).",
        "type": "text",
        "tokenized": True,
        "tokenizer": "gpt2",
        "approx_tokens": 470_000_000
    }
}

def download(name, out_dir=None, shard_limit=None, max_tokens=None, tokens_per_shard=None):
    if name not in DATASETS:
        print(f"Error: Dataset '{name}' not found in registry.")
        return

    info = DATASETS[name]
    path = info["path"]
    config_name = info.get("name")
    
    if out_dir is None:
        out_dir = Path("./datasets") / name
    else:
        out_dir = Path(out_dir) / name
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # Handle tokenized datasets (e.g. pretraining datasets tokenized into shards)
    if info.get("tokenized", False):

        if tokens_per_shard is None:
            tokens_per_shard = 100_000_000  # 100M tokens per shard (~200MB as uint16)

        split = info.get("split", "train")
        text_field = info.get("text_field", "text")
        approx_tokens = info.get("approx_tokens", 0)

        print(f"\nDataset:    {info['description']}")
        print(f"HF path:    {path} ({config_name or 'default'})")
        print(f"Output:     {out_dir}")
        print(f"Max tokens: {max_tokens or 'all (~' + str(approx_tokens // 1_000_000_000) + 'B)'}")
        print(f"Shard size: {tokens_per_shard:,} tokens\n")

        # Check for existing progress
        metadata_path = out_dir / "metadata.json"
        if metadata_path.exists():
            try:
                with open(metadata_path) as f:
                    existing = json.load(f)
                print(f"Resuming: found {existing['total_tokens']:,} tokens in {existing['num_shards']} shards")
                shard_idx = existing["num_shards"]
                total_tokens = existing["total_tokens"]
                docs_processed = existing.get("docs_processed", 0)
            except Exception:
                shard_idx = 0
                total_tokens = 0
                docs_processed = 0
        else:
            shard_idx = 0
            total_tokens = 0
            docs_processed = 0

        if max_tokens and total_tokens >= max_tokens:
            print(f"Already have {total_tokens:,} tokens (target: {max_tokens:,}). Done.")
            return

        tokenizer_name = info.get("tokenizer", "gpt2")
        tokenizer = tiktoken.get_encoding(tokenizer_name)
        EOT_TOKEN_ID = 50256

        # Stream dataset
        print("Loading dataset (streaming mode)...")
        ds = load_dataset(
            path,
            name=config_name,
            split=split,
            streaming=True,
        )

        # Skip already-processed documents
        if docs_processed > 0:
            print(f"Skipping {docs_processed:,} already-processed documents...")
            ds = ds.skip(docs_processed)

        current_shard = []
        t0 = time.time()

        try:
            for doc in ds:
                text = doc[text_field]
                if not text or not text.strip():
                    docs_processed += 1
                    continue

                tokens = tokenizer.encode(text, allowed_special="all")
                tokens.append(EOT_TOKEN_ID)  # document separator
                current_shard.extend(tokens)
                docs_processed += 1

                # Shard is full — write it
                while len(current_shard) >= tokens_per_shard:
                    shard_data = np.array(current_shard[:tokens_per_shard], dtype=np.uint16)
                    shard_path = out_dir / f"shard_{shard_idx:05d}.npy"
                    np.save(shard_path, shard_data)

                    total_tokens += tokens_per_shard
                    current_shard = current_shard[tokens_per_shard:]
                    shard_idx += 1

                    elapsed = time.time() - t0
                    rate = total_tokens / elapsed if elapsed > 0 else 0
                    print(f"  Shard {shard_idx:5d} saved | "
                          f"{total_tokens / 1e9:.2f}B tokens | "
                          f"{rate / 1e6:.1f}M tok/s | "
                          f"{docs_processed:,} docs")

                    # Save progress
                    with open(metadata_path, "w") as f:
                        json.dump({
                            "dataset": name,
                            "num_shards": shard_idx,
                            "total_tokens": total_tokens,
                            "docs_processed": docs_processed,
                            "tokens_per_shard": tokens_per_shard,
                            "vocab_size": 50257,
                            "tokenizer": "tiktoken-gpt2",
                        }, f, indent=2)

                    if max_tokens and total_tokens >= max_tokens:
                        print(f"\nReached target of {max_tokens:,} tokens.")
                        return

        except KeyboardInterrupt:
            print("\nInterrupted! Saving progress...")

        # Save any remaining tokens as a final partial shard
        if current_shard:
            shard_data = np.array(current_shard, dtype=np.uint16)
            shard_path = out_dir / f"shard_{shard_idx:05d}.npy"
            np.save(shard_path, shard_data)
            total_tokens += len(current_shard)
            shard_idx += 1
            print(f"  Shard {shard_idx} saved (partial: {len(current_shard):,} tokens)")

        with open(metadata_path, "w") as f:
            json.dump({
                "dataset": name,
                "num_shards": shard_idx,
                "total_tokens": total_tokens,
                "docs_processed": docs_processed,
                "tokens_per_shard": tokens_per_shard,
                "vocab_size": 50257,
                "tokenizer": "tiktoken-gpt2",
            }, f, indent=2)

        elapsed = time.time() - t0
        print(f"\nDone! {total_tokens:,} tokens in {shard_idx} shards ({elapsed / 60:.1f} min)")
        return

    print(f"Downloading {name} ({path})...")
    print(f"Description: {info['description']}")

    # Handle shard-based download if requested
    shards = shard_limit if shard_limit is not None else info.get("shards")
    if shards:
        print(f"Downloading first {shards} shards...")
        shard_start = info.get("shard_start", 1)
        shard_pattern = info.get("shard_pattern", "images/part-{i:06d}.zip")
        
        allow_patterns = []
        for i in range(shard_start, shard_start + shards):
            allow_patterns.append(shard_pattern.format(i=i))
        
        # Include extra files (e.g. metadata, annotations) if defined
        extra_files = info.get("extra_files", [])
        if isinstance(extra_files, list):
            allow_patterns.extend(extra_files)
        else:
            allow_patterns.append(extra_files)

        snapshot_download(
            repo_id=path, 
            repo_type="dataset", 
            local_dir=str(out_dir),
            allow_patterns=allow_patterns
        )
        print(f"  Downloaded {shards} shards to {out_dir}")
        return

    # Standard download
    try:
        ds = load_dataset(path, name=config_name)
        print(f"Saving to {out_dir}...")
        # Save as JSONL for SFT or arrow/parquet for others
        if info["type"] == "sft":
            for split, data in ds.items():
                split_file = out_dir / f"{split}.json"
                data.to_json(split_file, indent=2)
                print(f"  Saved {split} split to {split_file.name}")
        else:
            ds.save_to_disk(str(out_dir))
            print(f"  Saved to disk format in {out_dir}")
    except RuntimeError as e:
        if "Dataset scripts are no longer supported" in str(e):
            print("Dataset script not supported. Falling back to snapshot_download...")
            snapshot_download(repo_id=path, repo_type="dataset", local_dir=str(out_dir))
            print(f"  Downloaded raw repository to {out_dir}")
        else:
            raise e

    print("\nDownload complete.")

def list_datasets():
    print(f"{'Name':<20} | {'Type':<10} | {'Description'}")
    print("-" * 80)
    for name, info in DATASETS.items():
        print(f"{name:<20} | {info['type']:<10} | {info['description']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General HuggingFace Dataset Downloader")
    parser.add_argument("--dataset", type=str, help="Name of the dataset to download")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--shards", type=int, default=None, help="Number of shards to download (if supported)")
    parser.add_argument("--max-tokens", type=int, default=None, help="Stop after this many tokens (tokenized only)")
    parser.add_argument("--tokens-per-shard", type=int, default=None, help="Tokens per shard file (tokenized only)")
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset:
        download(
            args.dataset, 
            args.out_dir, 
            shard_limit=args.shards, 
            max_tokens=args.max_tokens, 
            tokens_per_shard=args.tokens_per_shard
        )
    else:
        parser.print_help()
