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
import os
from pathlib import Path

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

    # --- Other ---
    "wikitext-103": {
        "path": "wikitext",
        "name": "wikitext-103-v1",
        "description": "Large-scale collection of high-quality Wikipedia articles.",
        "type": "text"
    }
}

def download(name, out_dir=None):
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not found. Install it with: pip install datasets")
        return

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

    print(f"Downloading {name} ({path})...")
    print(f"Description: {info['description']}")
    
    # Load and save to disk
    # We load as a DatasetDict (all splits)
    ds = load_dataset(path, name=config_name)
    
    print(f"Saving to {out_dir}...")
    # Save as JSONL for SFT or arrow/parquet for others
    # For SFT we often prefer JSON for easy inspection
    if info["type"] == "sft":
        for split, data in ds.items():
            split_file = out_dir / f"{split}.json"
            data.to_json(split_file, indent=2)
            print(f"  Saved {split} split to {split_file.name}")
    else:
        # Standard save_to_disk for vision/others
        ds.save_to_disk(str(out_dir))
        print(f"  Saved to disk format in {out_dir}")

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
    parser.add_argument("--list", action="store_true", help="List all available datasets")
    
    args = parser.parse_args()

    if args.list:
        list_datasets()
    elif args.dataset:
        download(args.dataset, args.out_dir)
    else:
        parser.print_help()
