#!/usr/bin/env python3
"""
YALLM Shared Utilities
Contains directory helpers, status logging, checkpointing, and tokenizer helpers.
"""

import json
import os
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn

EOT_TOKEN = "<" + "|endoftext|" + ">"
EOT_TOKEN_ID = 50256

MODELS_DIR = Path(os.environ.get("YALLM_MODELS_DIR", Path(__file__).parent / "models"))


# ===========================================================================
#  Model directory helpers  (models/<model_name>/)
# ===========================================================================

def get_model_dir(model_name):
    """Return the Path to models/<model_name>, creating it if needed."""
    d = MODELS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model_config(model_name, config):
    """Write the full model config dict to models/<model_name>/config.json."""
    d = get_model_dir(model_name)
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_model_config(model_name):
    """Load the config dict from models/<model_name>/config.json."""
    d = get_model_dir(model_name)
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {d}")
    with open(cfg_path) as f:
        return json.load(f)


def save_checkpoint(model_name, model, optimizer, epoch, global_step,
                    tokens_seen, train_losses, val_losses, tag=None):
    """Save a training checkpoint to models/<model_name>/."""
    d = get_model_dir(model_name)
    fname = f"checkpoint_step{global_step}.pt" if tag is None else f"checkpoint_{tag}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, d / fname)
    # Also save a "latest" symlink/copy for easy resume
    latest = d / "checkpoint_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    os.symlink(fname, latest)
    
    # Keep only the latest 3 checkpoints across all epochs/steps (excluding final)
    if tag != "final":
        checkpoints = sorted(
            [p for p in d.glob("checkpoint_*.pt") if "latest" not in p.name and "final" not in p.name],
            key=lambda p: os.path.getmtime(p)
        )
        while len(checkpoints) > 3:
            old_ckpt = checkpoints.pop(0)
            if old_ckpt.name != fname:
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass

    return d / fname


def load_checkpoint(model_name, model, optimizer=None, tag="latest"):
    """Load a checkpoint from models/<model_name>/. Returns the metadata dict."""
    d = get_model_dir(model_name)
    ckpt_path = d / f"checkpoint_{tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint_{tag}.pt found in {d}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    
    # Strip torch.compile prefix (_orig_mod.) if present in the checkpoint
    state_dict = ckpt["model_state_dict"]
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        
    model.load_state_dict(state_dict)
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "epoch": ckpt["epoch"],
        "global_step": ckpt["global_step"],
        "tokens_seen": ckpt["tokens_seen"],
        "train_losses": ckpt.get("train_losses", []),
        "val_losses": ckpt.get("val_losses", []),
    }


def get_status_file(model_name):
    """Return the path to the status log for this model."""
    d = get_model_dir(model_name)
    return d / "status.txt"


# ===========================================================================
#  Status logging (writes to models/<model_name>/status.txt)
# ===========================================================================

_status_path = None   # set by scripts at startup


def set_status_file(path):
    global _status_path
    _status_path = Path(path)


def write_status(msg):
    """Append a timestamped message to the status file for external monitoring."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    if _status_path is not None:
        with open(_status_path, "a") as f:
            f.write(line)
    print(msg)


# ===========================================================================
#  Tokenizer helpers
# ===========================================================================

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={EOT_TOKEN})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(tokens, tokenizer):
    return tokenizer.decode(tokens.squeeze(0).tolist())
