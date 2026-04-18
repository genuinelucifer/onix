#!/usr/bin/env python3
"""
YALLM Shared Training Utilities

Consolidates common boilerplate used across train_llm.py, train_multimodal.py,
and train_vqvae.py:
  - Optimizer creation (adamw, sgd, 8-bit variants)
  - Status file setup
  - Device selection
  - Checkpoint resume with graceful fallback (no checkpoint → fresh start)
  - Training hyperparameter extraction
  - Early stopping (for LLM and multimodal training)
"""

import sys
from pathlib import Path

import torch

from model import (
    write_status, set_status_file, get_status_file, get_model_dir,
)


# ---------------------------------------------------------------------------
#  Device
# ---------------------------------------------------------------------------

def setup_device():
    """Select CUDA if available, else CPU."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return device


# ---------------------------------------------------------------------------
#  Status file
# ---------------------------------------------------------------------------

def setup_status_file(model_name, resume=False):
    """Initialize the status file for a training run.

    On a fresh run the file is truncated; on resume it is appended to.
    Returns the status file Path.
    """
    status_file = get_status_file(model_name)
    set_status_file(status_file)
    if not resume:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            f.write("")
    return status_file


# ---------------------------------------------------------------------------
#  Optimizer factory
# ---------------------------------------------------------------------------

def create_optimizer(model, opt_name, lr, weight_decay=0.1):
    """Unified optimizer factory.

    Supports: adamw, sgd, adamw8bit, sgd8bit.
    """
    opt_name = opt_name.lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name in ("adamw8bit", "sgd8bit"):
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                f"Optimizer '{opt_name}' requires the 'bitsandbytes' library. "
                "Install with: pip install bitsandbytes"
            )
        if opt_name == "adamw8bit":
            return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            return bnb.optim.SGD8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


# ---------------------------------------------------------------------------
#  Resume helpers
# ---------------------------------------------------------------------------

def migrate_optimizer_to_device(optimizer, device):
    """Move all optimizer state tensors to *device*."""
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(device)


def handle_resume_no_checkpoint(model_name):
    """When --resume is passed but no checkpoint exists:

    1. Verify config.json exists (error if not).
    2. Clear stderr.log (previous run likely crashed).
    3. Log a warning.

    Returns the loaded config dict.
    """
    from model import load_model_config  # avoid circular at module level

    model_dir = get_model_dir(model_name)

    # Config must exist
    config_path = model_dir / "config.json"
    if not config_path.exists():
        print(f"Error: --resume specified but no config.json found in {model_dir}. "
              f"Cannot resume without a saved configuration.", file=sys.stderr)
        sys.exit(1)

    full_cfg = load_model_config(model_name)

    # Clear stderr.log (previous attempt probably crashed)
    stderr_log = model_dir / "stderr.log"
    if stderr_log.exists():
        with open(stderr_log, "w") as f:
            f.write("")
        write_status("WARNING: stderr.log cleared (previous run may have failed)")

    write_status(
        "WARNING: --resume but no checkpoint found. "
        "Starting fresh with existing config."
    )
    return full_cfg


def has_checkpoint(model_name, tag="latest"):
    """Check whether a checkpoint file exists for this model."""
    d = get_model_dir(model_name)
    return (d / f"checkpoint_{tag}.pt").exists()


# ---------------------------------------------------------------------------
#  Training parameter extraction
# ---------------------------------------------------------------------------

_TRAIN_PARAM_KEYS = [
    "save_every", "save_iters", "log_freq", "eval_freq", "eval_iter",
]


def get_train_params(train_cfg, args):
    """Extract training hyperparams, preferring saved config over CLI defaults.

    Returns a dict with keys: save_every, save_iters, log_freq, eval_freq, eval_iter.
    """
    result = {}
    for key in _TRAIN_PARAM_KEYS:
        result[key] = train_cfg.get(key, getattr(args, key.replace("-", "_"), None))
    return result


# ---------------------------------------------------------------------------
#  Early stopping
# ---------------------------------------------------------------------------

class EarlyStopper:
    """Tracks eval-loss trend and triggers early stopping.

    Parameters
    ----------
    patience_evals : int
        Number of consecutive evaluation points with no improvement before
        stopping.  (Not raw training steps — evaluation points.)
    min_delta : float
        Minimum improvement in loss to count as progress.
    min_epochs : int
        Don't stop before this many *full* epochs have completed (i.e. all
        data has been seen at least this many times).
    """

    def __init__(self, patience_evals=6, min_delta=1e-4, min_epochs=1):
        self.patience_evals = patience_evals
        self.min_delta = min_delta
        self.min_epochs = min_epochs

        self.best_loss = float("inf")
        self.best_step = -1
        self.evals_since_best = 0

    def check(self, loss, global_step, completed_epochs):
        """Return True if training should stop.

        Parameters
        ----------
        loss : float
            Current evaluation loss.
        global_step : int
            Current training step (for logging only).
        completed_epochs : int
            Number of *full* epochs completed so far (0-indexed epoch that just
            finished, +1).  E.g., after epoch 0 ends, pass 1.
        """
        if loss < self.best_loss - self.min_delta:
            self.best_loss = loss
            self.best_step = global_step
            self.evals_since_best = 0
        else:
            self.evals_since_best += 1

        # Don't stop before min_epochs full epochs have completed
        if completed_epochs < self.min_epochs:
            return False

        return self.evals_since_best >= self.patience_evals

    def status_message(self):
        """Human-readable status string for logging."""
        return (
            f"best_loss={self.best_loss:.4f} at step {self.best_step}, "
            f"evals_since_best={self.evals_since_best}/{self.patience_evals}"
        )
