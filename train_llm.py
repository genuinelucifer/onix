#!/usr/bin/env python3
"""
YALLM Pretraining Script
Train any decoder-only transformer from scratch.

Supports:
  - Preset architectures: gpt2-124m, llama-1b, llama-3b, mistral-1b, gptj-1b
  - Custom architecture via JSON config file
  - Small text files or large pre-tokenized shard datasets
  - Automatic resuming from latest checkpoints
  - Early stopping when loss stagnates (after at least 1 epoch)

Examples:
  # Train with a preset
  ./run_train.sh my-llama --mode llm --preset llama-1b --data ../the-verdict.txt

  # Train with custom config
  ./run_train.sh my-model --mode llm --config configs/custom.json --data ../the-verdict.txt

  # Train on pre-tokenized shards
  ./run_train.sh my-llama --mode llm --preset llama-1b --data-dir pretrain_data/fineweb_edu_10bt/

  # Resume training
  ./run_train.sh my-llama --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Modern architecture imports
from architecture import ModelConfig, CausalLM, PRESETS, get_preset
from architecture.generate import generate

# Utilities (shared)
from model import (
    get_tokenizer, text_to_token_ids, token_ids_to_text,
    write_status, save_model_config, load_model_config,
    save_checkpoint, load_checkpoint,
    MODELS_DIR,
)

# Shared training utilities
from training_utils import (
    create_optimizer, setup_status_file, setup_device,
    migrate_optimizer_to_device, handle_resume_no_checkpoint, has_checkpoint,
    get_train_params, EarlyStopper,
)

# Data pipeline
from pretrain_data.dataset import create_pretrain_dataloaders


# ---------------------------------------------------------------------------
#  Loss helpers
# ---------------------------------------------------------------------------

def calc_loss_batch(inp, tgt, model, device):
    inp, tgt = inp.to(device), tgt.to(device)
    logits = model(inp)
    return nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten())


def calc_loss_loader(loader, model, device, num_batches=None):
    if len(loader) == 0:
        return float("nan")
    n = min(num_batches, len(loader)) if num_batches else len(loader)
    total = 0.0
    for i, (inp, tgt) in enumerate(loader):
        if i >= n:
            break
        total += calc_loss_batch(inp, tgt, model, device).item()
    return total / n


def evaluate(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, eval_iter)
        vl = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return tl, vl


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_loop(model, train_loader, val_loader, optimizer, device,
               num_epochs, log_freq, eval_freq, eval_iter, start_context, tokenizer,
               model_name, save_every_n_epochs, save_every_n_iters=None,
               start_epoch=0, start_global_step=-1, start_tokens_seen=0,
               prev_train_losses=None, prev_val_losses=None,
               early_stopper=None):
    train_losses = list(prev_train_losses or [])
    val_losses = list(prev_val_losses or [])
    tokens_seen = start_tokens_seen
    global_step = start_global_step

    # Get context size from model
    ctx_size = model.config.context_length

    # Track completed epochs for early stopping
    completed_epochs = start_epoch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, (inp, tgt) in enumerate(train_loader):
            # Fast-forward resume logic: skip already processed batches in start_epoch
            current_abs_step = epoch * len(train_loader) + i
            if current_abs_step <= start_global_step:
                continue

            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inp.numel()
            global_step += 1

            if global_step % log_freq == 0:
                write_status(
                    f"PROGRESS epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"tokens={tokens_seen} loss={loss.item():.4f}"
                )

            if global_step % eval_freq == 0:
                tl, vl = evaluate(model, train_loader, val_loader,
                                  device, eval_iter)
                train_losses.append(tl)
                val_losses.append(vl)
                write_status(
                    f"EVAL epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"tokens={tokens_seen} train_loss={tl:.4f} val_loss={vl:.4f}"
                )

                # Early stopping check
                if early_stopper is not None:
                    if early_stopper.check(vl, global_step, completed_epochs):
                        write_status(
                            f"EARLY_STOP triggered at step {global_step} "
                            f"({early_stopper.status_message()})"
                        )
                        # Save checkpoint before stopping
                        ckpt_path = save_checkpoint(
                            model_name, model, optimizer, epoch + 1, global_step,
                            tokens_seen, train_losses, val_losses, tag="early_stop")
                        save_checkpoint(
                            model_name, model, optimizer, epoch + 1, global_step,
                            tokens_seen, train_losses, val_losses)
                        write_status(f"CHECKPOINT saved (early stop) -> {ckpt_path}")
                        return train_losses, val_losses

            # Iteration checkpoint
            if save_every_n_iters is not None and save_every_n_iters > 0 and global_step > 0:
                if global_step % save_every_n_iters == 0:
                    # Save current epoch (not epoch + 1) because the epoch is not yet complete
                    ckpt_path = save_checkpoint(
                        model_name, model, optimizer, epoch, global_step,
                        tokens_seen, train_losses, val_losses, tag=f"step{global_step}")
                    write_status(f"CHECKPOINT saved at step {global_step} -> {ckpt_path}")

        # Epoch completed
        completed_epochs = epoch + 1

        # Generate a sample after each epoch
        model.eval()
        enc = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            gen = generate(model, enc, max_new_tokens=50, context_size=ctx_size)
        sample = token_ids_to_text(gen, tokenizer).replace("\n", " ")
        write_status(f"SAMPLE epoch={epoch+1}: {sample}")
        model.train()

        # Periodic checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            ckpt_path = save_checkpoint(
                model_name, model, optimizer, epoch + 1, global_step,
                tokens_seen, train_losses, val_losses)
            write_status(f"CHECKPOINT saved at epoch {epoch+1} -> {ckpt_path}")

    # Final checkpoint (always)
    ckpt_path = save_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        tokens_seen, train_losses, val_losses, tag="final")
    save_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        tokens_seen, train_losses, val_losses)
    write_status(f"FINAL checkpoint saved -> {ckpt_path}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
#  Model creation helpers
# ---------------------------------------------------------------------------

def create_model_from_config(config: ModelConfig, device: torch.device) -> CausalLM:
    """Create a new CausalLM from a ModelConfig."""
    torch.manual_seed(123)
    model = CausalLM(config)
    model.to(device)
    write_status(f"MODEL created: {config.name}")
    write_status(model.summary())
    return model


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YALLM Pretrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New architecture presets
  python train.py --mode llm --model-name my-llama --preset llama-1b --data ../the-verdict.txt
  python train.py --mode llm --model-name my-llama --preset llama-1b --data-dir pretrain_data/fineweb_edu_10bt/

  # Custom config file
  python train.py --mode llm --model-name my-custom --config my_config.json --data ../the-verdict.txt

  # Resume any model
  python train.py --model-name my-llama --resume
""",
    )
    # Model specification
    model_group = parser.add_argument_group("Model specification")
    model_group.add_argument("--preset", default=None,
                             choices=list(PRESETS.keys()),
                             help="Use a preset architecture (e.g. llama-1b, gpt2-124m)")
    model_group.add_argument("--config", default=None,
                             help="Path to custom model config JSON")
    model_group.add_argument("--model-size", default=None,
                             help="Legacy compatibility mapping (e.g. 124M -> gpt2-124m)")

    # Common args
    parser.add_argument("--model-name", required=True,
                        help="Name for this model (creates models/<name>/)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training text file")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to pre-tokenized shard directory")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--context-length", type=int, default=None,
                        help="Override context length (default: from preset/config)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total epochs (not additional)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save-iters", type=int, default=0,
                        help="Save checkpoint every N iterations (0 to disable)")
    parser.add_argument("--eval-freq", type=int, default=50,
                        help="Evaluate (train/val loss over multiple batches) every N steps")
    parser.add_argument("--log-freq", type=int, default=5,
                        help="Log current batch loss to status.txt every N steps")
    parser.add_argument("--eval-iter", type=int, default=5,
                        help="Number of batches per evaluation")
    parser.add_argument("--drop-rate", type=float, default=0.1,
                        help="Dropout rate (legacy mode)")
    parser.add_argument("--max-shards", type=int, default=None,
                        help="Limit number of data shards (for testing)")

    # Early stopping
    parser.add_argument("--patience-evals", type=int, default=6,
                        help="Stop after N consecutive evals with no improvement "
                             "(0 to disable). Only active after first full epoch.")

    # Optimization args
    opt_group = parser.add_argument_group("Optimization & Memory")
    opt_group.add_argument("--no-sdpa", action="store_false", dest="use_sdpa",
                           help="Disable SDPA (Flash/Mem-Eff Attention)")
    opt_group.add_argument("--checkpointing", action="store_true",
                           help="Enable gradient checkpointing (saves VRAM, slower training)")
    opt_group.add_argument("--optimizer", default="adamw",
                           choices=["adamw", "sgd", "adamw8bit", "sgd8bit"],
                           help="Optimizer to use (default: adamw)")
    opt_group.set_defaults(use_sdpa=True)

    args = parser.parse_args()

    # ---- Setup ----
    setup_status_file(args.model_name, resume=args.resume)
    device = setup_device()
    write_status(f"START device={device} model_name={args.model_name} resume={args.resume}")

    tokenizer = get_tokenizer()

    if args.resume:
        # ---- Resume mode ----
        checkpoint_exists = has_checkpoint(args.model_name)

        if checkpoint_exists:
            write_status("RESUME loading config and checkpoint...")
            full_cfg = load_model_config(args.model_name)
        else:
            # No checkpoint — use saved config, start fresh
            full_cfg = handle_resume_no_checkpoint(args.model_name)

        train_cfg = full_cfg["training"]

        total_epochs = args.epochs if args.epochs != 10 else train_cfg["epochs"]
        data_path = train_cfg.get("data")
        data_dir = train_cfg.get("data_dir")

        # Determine model type from saved config
        if "architecture" in full_cfg:
            # Modern architecture
            model_config = ModelConfig.from_dict(full_cfg["architecture"])
            model = CausalLM(model_config).to(device)
            write_status(f"RESUMED architecture: {model_config.name}")
        else:
            raise ValueError(
                "This trainer no longer supports legacy GPT2 class models. "
                "Please use a previous version of YALLM or convert the state_dict."
            )

        opt_name = train_cfg.get("optimizer", "adamw")
        optimizer = create_optimizer(model, opt_name, lr=train_cfg["lr"])

        if checkpoint_exists:
            ckpt_meta = load_checkpoint(args.model_name, model, optimizer)
            model.to(device)
            migrate_optimizer_to_device(optimizer, device)

            start_epoch = ckpt_meta["epoch"]
            start_step = ckpt_meta["global_step"]
            start_tokens = ckpt_meta["tokens_seen"]
            prev_tl = ckpt_meta["train_losses"]
            prev_vl = ckpt_meta["val_losses"]
        else:
            # Fresh start with existing config
            start_epoch, start_step, start_tokens = 0, -1, 0
            prev_tl, prev_vl = [], []

        if start_epoch >= total_epochs:
            write_status(f"Already trained {start_epoch} epochs (target={total_epochs}). "
                         f"Increase --epochs to continue.")
            return

        write_status(f"RESUMED from epoch={start_epoch + 1} step={start_step} "
                     f"tokens={start_tokens} -> training to epoch {total_epochs}")

        # Get context length for data loading
        context_length = model.config.context_length
        batch_size = train_cfg["batch_size"]

    else:
        # ---- New run ----
        if args.data is None and args.data_dir is None:
            parser.error("--data or --data-dir is required for a new training run")

        # Handle legacy --model-size mapping
        preset_name = args.preset
        if not preset_name and not args.config and args.model_size:
            msize = args.model_size.lower()
            if msize in ("124m", "355m", "774m", "1558m"):
                preset_name = f"gpt2-{msize}"
                write_status(f"MAPPED legacy --model-size {args.model_size} to preset {preset_name}")
            else:
                parser.error(f"Unknown legacy model-size: {args.model_size}")

        if not preset_name and not args.config:
            parser.error("Specify --preset or --config")

        # Load architecture configuration
        if args.config:
            model_config = ModelConfig.load(args.config)
            write_status(f"CONFIG loaded from {args.config}")
        else:
            model_config = get_preset(preset_name)

        # Override context length and memory optimizations if specified
        if args.context_length:
            model_config.context_length = args.context_length
        model_config.use_sdpa = args.use_sdpa
        model_config.grad_checkpointing = args.checkpointing

        model = create_model_from_config(model_config, device)
        context_length = model_config.context_length

        # Save config
        train_cfg = {
            "data": args.data,
            "data_dir": args.data_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "save_every": args.save_every,
            "save_iters": args.save_iters,
            "log_freq": args.log_freq,
            "eval_freq": args.eval_freq,
            "eval_iter": args.eval_iter,
            "optimizer": args.optimizer,
        }
        full_cfg = {
            "architecture": model_config.to_dict(),
            "training": train_cfg,
        }

        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")

        total_epochs = args.epochs
        data_path = args.data
        data_dir = args.data_dir
        batch_size = args.batch_size

        optimizer = create_optimizer(model, args.optimizer, lr=args.lr)
        start_epoch, start_step, start_tokens = 0, -1, 0
        prev_tl, prev_vl = [], []

    # ---- Load data ----
    if data_dir and Path(data_dir).is_dir():
        # Sharded pre-tokenized data
        train_loader, val_loader, data_info = create_pretrain_dataloaders(
            data_dir, seq_len=context_length, batch_size=batch_size,
            max_shards=args.max_shards,
        )
        write_status(f"DATA sharded: {data_info}")
    elif data_path:
        # Text file (small dataset)
        train_loader, val_loader, data_info = create_pretrain_dataloaders(
            data_path, seq_len=context_length, batch_size=batch_size,
            tokenizer=tokenizer,
        )
        write_status(f"DATA text file: {data_info}")
    else:
        parser.error("No data source found. Provide --data or --data-dir.")

    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} batches")

    n_params = sum(p.numel() for p in model.parameters())
    write_status(f"MODEL params={n_params:,}")

    # ---- Early stopper ----
    early_stopper = None
    if args.patience_evals > 0:
        early_stopper = EarlyStopper(
            patience_evals=args.patience_evals,
            min_epochs=1,
        )
        write_status(f"EARLY_STOP enabled: patience={args.patience_evals} evals, min_epochs=1")

    # ---- Train ----
    t0 = time.time()
    tp = get_train_params(train_cfg, args)

    train_loop(model, train_loader, val_loader, optimizer, device,
               num_epochs=total_epochs,
               log_freq=tp["log_freq"],
               eval_freq=tp["eval_freq"],
               eval_iter=tp["eval_iter"],
               start_context="Every effort moves you",
               tokenizer=tokenizer,
               model_name=args.model_name,
               save_every_n_epochs=tp["save_every"],
               save_every_n_iters=tp["save_iters"],
               start_epoch=start_epoch,
               start_global_step=start_step,
               start_tokens_seen=start_tokens,
               prev_train_losses=prev_tl,
               prev_val_losses=prev_vl,
               early_stopper=early_stopper)

    elapsed = (time.time() - t0) / 60
    write_status(f"DONE training completed in {elapsed:.2f} min")


if __name__ == "__main__":
    main()
