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

# Cosmetic: safe to change even if resuming from a checkpoint
_COSMETIC_PARAMS = [
    "save_every", "save_iters", "log_freq", "eval_freq", "eval_iter",
    "patience", "min_delta", "min_epochs", "window_size",
]

# Functional: potentially disruptive to change mid-training if a checkpoint exists
_FUNCTIONAL_PARAMS = [
    "batch_size", "lr", "optimizer", "epochs"
]

# Mode-specific defaults
DEFAULT_CONFIGS = {
    "llm": {
        "epochs": 10, "batch_size": 8, "lr": 4e-4,
        "save_every": 5, "save_iters": 0,
        "eval_freq": 50, "log_freq": 5, "eval_iter": 5,
        "optimizer": "adamw",
        "patience": 6, "min_delta": 1e-4, "min_epochs": 2, "window_size": 3,
    },
    "vqvae": {
        "epochs": 100, "batch_size": 16, "lr": 3e-4,
        "save_every": 10, "save_iters": 0,
        "eval_freq": 100, "log_freq": 10, "eval_iter": 5,
        "optimizer": "adamw",
        "patience": 6, "min_delta": 1e-4, "min_epochs": 2, "window_size": 3,
    },
    "multimodal": {
        "epochs": 50, "batch_size": 32, "lr": 4e-4,
        "save_every": 5, "save_iters": 0,
        "eval_freq": 100, "log_freq": 1, "eval_iter": 5,
        "optimizer": "adamw",
        "patience": 6, "min_delta": 1e-4, "min_epochs": 2, "window_size": 3,
    }
}


def add_common_training_args(parser):
    """Add standardized training arguments and early stopping params to a parser."""
    # Identification
    parser.add_argument("--model-name", required=True,
                        help="Name for this model (creates models/<name>/)")
    parser.add_argument("--config", default=None,
                        help="Path to model architecture config JSON")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")

    # Hyperparameters (all default to None to honor config.json on resume)
    train_group = parser.add_argument_group("Training Hyperparameters")
    train_group.add_argument("--epochs", type=int, default=None)
    train_group.add_argument("--batch-size", type=int, default=None)
    train_group.add_argument("--lr", type=float, default=None)

    # Checkpointing & Logging
    log_group = parser.add_argument_group("Logging & Checkpointing")
    log_group.add_argument("--save-every", type=int, default=None,
                           help="Save checkpoint every N epochs")
    log_group.add_argument("--save-iters", type=int, default=None,
                           help="Save checkpoint every N iterations (0 to disable)")
    log_group.add_argument("--eval-freq", type=int, default=None,
                           help="Evaluate every N steps")
    log_group.add_argument("--log-freq", type=int, default=None,
                           help="Log status message every N steps")
    log_group.add_argument("--eval-iter", type=int, default=None,
                           help="Number of batches per evaluation")

    # Optimization
    opt_group = parser.add_argument_group("Optimization & Memory")
    opt_group.add_argument("--optimizer", default=None,
                           choices=["adamw", "sgd", "adamw8bit", "sgd8bit"],
                           help="Optimizer to use")
    opt_group.add_argument("--checkpointing", action="store_true",
                           help="Enable gradient checkpointing (saves VRAM)")

    # Early stopping
    stop_group = parser.add_argument_group("Early Stopping")
    stop_group.add_argument("--patience", type=int, default=None,
                            help="Stop after N consecutive evals with no improvement (0 to disable)")
    stop_group.add_argument("--min-delta", type=float, default=None,
                            help="Minimum relative improvement in loss")
    stop_group.add_argument("--min-epochs", type=int, default=None,
                            help="Minimum full epochs to complete before allow stop")
    stop_group.add_argument("--window-size", type=int, default=None,
                            help="Smoothing window size for validation loss")

    return parser


def get_default_training_config(mode, args=None):
    """
    Return a training config dict for a new run, using mode defaults
    and optional command-line overrides.
    """
    if mode not in DEFAULT_CONFIGS:
        raise ValueError(f"Unknown mode: {mode}")

    cfg = DEFAULT_CONFIGS[mode].copy()
    if args:
        # Override with any non-None arguments
        for key in cfg.keys():
            val = getattr(args, key, None)
            if val is not None:
                cfg[key] = val
    return cfg


def get_train_params(mode, train_cfg, args, has_checkpoint: bool = False):
    """Extract training hyperparams from CLI and saved config.

    Rules for overrides:
    1. Cosmetic params (logs, save frequency) can always be overridden by CLI.
    2. Functional params (batch_size, LR) can only be overridden if no checkpoint exists
       (i.e., a fresh start or resume-without-checkpoint).

    Returns a dict with all parameters populated.
    """
    # 1. Start with mode defaults
    result = DEFAULT_CONFIGS[mode].copy()

    # 2. Update with everything from the saved config
    result.update(train_cfg)

    # 2. Check for CLI overrides
    # All cosmetic params are allowed
    for key in _COSMETIC_PARAMS:
        val = getattr(args, key, None)
        if val is not None:
            # Only override if the user didn't leave it at total default (argparse check)
            # Actually, we can just check if it's explicitly provided if we change the defaults 
            # to None in the training scripts. For now, we'll check if they differ.
            result[key] = val

    # 3. Functional params are only allowed if we haven't reached a checkpoint yet
    if not has_checkpoint:
        for key in _FUNCTIONAL_PARAMS:
            val = getattr(args, key, None)
            if val is not None:
                result[key] = val

    return result


# ---------------------------------------------------------------------------
#  Early stopping
# ---------------------------------------------------------------------------

from collections import deque

class EarlyStopper:
    """Tracks eval-loss trend and triggers early stopping with noise smoothing.

    Parameters
    ----------
    patience_evals : int
        Number of consecutive evaluation points with no improvement before stopping.
    min_delta : float
        Minimum relative improvement in loss (e.g., 0.001 = 0.1% improvement).
    min_epochs : int
        Don't stop before this many *full* epochs have completed.
    window_size : int
        Smoothing window size for validation loss.
    """

    def __init__(self, patience_evals=6, min_delta=1e-4, min_epochs=2, window_size=3):
        self.patience_evals = patience_evals
        self.min_delta = min_delta
        self.min_epochs = min_epochs
        self.window_size = window_size

        self.best_loss = float("inf")
        self.best_step = -1
        self.evals_since_best = 0
        self.history = deque(maxlen=window_size)

    def check(self, loss, global_step, completed_epochs):
        """Return True if training should stop. Uses moving average of recent losses."""
        self.history.append(loss)
        
        # Don't check until window is full
        if len(self.history) < self.window_size:
            return False

        # Calculate smoothed loss
        smoothed_loss = sum(self.history) / len(self.history)

        # Relative improvement check: smoothed_loss must be < best_loss * (1 - delta)
        # Handle first valid window
        if self.best_loss == float("inf"):
            self.best_loss = smoothed_loss
            self.best_step = global_step
            return False

        if smoothed_loss < self.best_loss * (1.0 - self.min_delta):
            self.best_loss = smoothed_loss
            self.best_step = global_step
            self.evals_since_best = 0
        else:
            self.evals_since_best += 1

        # Logic for stopping
        if completed_epochs < self.min_epochs:
            return False

        return self.evals_since_best >= self.patience_evals

    def status_message(self):
        """Human-readable status string for logging."""
        avg_loss = sum(self.history) / len(self.history) if self.history else 0.0
        return (
            f"smoothed_loss={avg_loss:.4f}, best_loss={self.best_loss:.4f} at step {self.best_step}, "
            f"evals_since_best={self.evals_since_best}/{self.patience_evals}"
        )
