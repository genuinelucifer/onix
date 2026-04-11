#!/usr/bin/env python3
"""
YALLM Unified Training Dispatcher

Routes training to the appropriate script based on mode:
  --mode llm          → train_llm.py    (text-only decoder transformer)
  --mode vqvae        → train_vqvae.py  (VQ-VAE image tokenizer, Phase 1)
  --mode multimodal   → train_multimodal.py (text-to-image LLM, Phase 2)

For backward compatibility, if no --mode is specified, defaults to 'llm'.

Usage:
    # LLM (default, same as before)
    python train.py --model-name my-llama --preset llama-1b --data ../the-verdict.txt

    # VQ-VAE
    python train.py --mode vqvae --model-name my-vqvae --config configs/vqvae_default.json \\
        --data-dir /path/to/images/ --epochs 100

    # Multi-modal
    python train.py --mode multimodal --model-name my-imggen \\
        --config configs/multimodal_pixelart.json \\
        --data-dir /path/to/image_text_pairs/ --epochs 50

    # Resume (auto-detects mode from saved config)
    python train.py --model-name my-vqvae --resume
"""

import argparse
import json
import sys
from pathlib import Path

# For auto-detection of model type on resume
MODELS_DIR = Path(__file__).parent / "models"


def detect_model_type(model_name: str) -> str:
    """Auto-detect model type from saved config.json."""
    config_path = MODELS_DIR / model_name / "config.json"
    if not config_path.exists():
        return "llm"  # default

    with open(config_path) as f:
        cfg = json.load(f)

    model_type = cfg.get("model_type", "")
    if model_type == "vqvae":
        return "vqvae"
    elif model_type == "multimodal":
        return "multimodal"
    else:
        return "llm"


def main():
    # We need to parse --mode and --model-name early, then forward
    # all remaining args to the appropriate training script.

    # Quick parse just for --mode and --resume + --model-name
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--mode", default=None,
                            choices=["llm", "vqvae", "multimodal"],
                            help="Training mode: llm, vqvae, or multimodal")
    pre_parser.add_argument("--model-name", default=None)
    pre_parser.add_argument("--resume", action="store_true")

    pre_args, remaining = pre_parser.parse_known_args()

    # Determine mode
    mode = pre_args.mode

    if mode is None:
        if pre_args.resume and pre_args.model_name:
            # Auto-detect from saved config
            mode = detect_model_type(pre_args.model_name)
            print(f"Auto-detected model type: {mode}")
        else:
            mode = "llm"  # default for backward compatibility

    # Build the full arg list for the sub-script
    # Re-add --model-name and --resume if they were parsed
    forward_args = list(remaining)
    if pre_args.model_name:
        forward_args = ["--model-name", pre_args.model_name] + forward_args
    if pre_args.resume:
        forward_args = ["--resume"] + forward_args

    # Replace sys.argv and call the appropriate main()
    sys.argv = [f"train_{mode}.py"] + forward_args

    if mode == "llm":
        from train_llm import main as llm_main
        llm_main()
    elif mode == "vqvae":
        from train_vqvae import main as vqvae_main
        vqvae_main()
    elif mode == "multimodal":
        from train_multimodal import main as mm_main
        mm_main()
    else:
        print(f"Unknown mode: {mode}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
