#!/usr/bin/env python3
"""
YALLM VQ-VAE Pretraining Script (Phase 1)

Train a VQ-VAE to compress and reconstruct images using a discrete codebook.
Once trained, the VQ-VAE is frozen and used as an image tokenizer for
multi-modal LLM training (Phase 2).

Usage:
    python train_vqvae.py --model-name my-vqvae --config configs/vqvae_default.json \
        --data-dir /path/to/images/ --epochs 100

    # Resume
    python train_vqvae.py --model-name my-vqvae --resume
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn

from architecture.config import VQVAEConfig
from architecture.vqvae import VQVAE
from architecture.losses import vqvae_loss

from pretrain_data.image_dataset import create_image_dataloaders

from model import (
    write_status, set_status_file, get_status_file,
    save_model_config, load_model_config, get_model_dir,
    MODELS_DIR,
)


# ---------------------------------------------------------------------------
#  Checkpointing (VQ-VAE specific)
# ---------------------------------------------------------------------------

def save_vqvae_checkpoint(model_name, model, optimizer, epoch, global_step,
                          train_losses, val_losses, tag=None):
    """Save a VQ-VAE checkpoint."""
    import os
    d = get_model_dir(model_name)
    fname = f"checkpoint_epoch{epoch}.pt" if tag is None else f"checkpoint_{tag}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, d / fname)

    # Symlink latest
    latest = d / "checkpoint_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    os.symlink(fname, latest)

    # Keep only latest 3 checkpoints (excluding final)
    if tag != "final":
        checkpoints = sorted(
            [p for p in d.glob("checkpoint_*.pt")
             if "latest" not in p.name and "final" not in p.name],
            key=lambda p: os.path.getmtime(p)
        )
        while len(checkpoints) > 3:
            old = checkpoints.pop(0)
            if old.name != fname:
                try:
                    old.unlink()
                except OSError:
                    pass

    return d / fname


def load_vqvae_checkpoint(model_name, model, optimizer=None, tag="latest"):
    """Load a VQ-VAE checkpoint."""
    d = get_model_dir(model_name)
    ckpt_path = d / f"checkpoint_{tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint_{tag}.pt in {d}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "epoch": ckpt["epoch"],
        "global_step": ckpt["global_step"],
        "train_losses": ckpt.get("train_losses", []),
        "val_losses": ckpt.get("val_losses", []),
    }


# ---------------------------------------------------------------------------
#  Evaluation
# ---------------------------------------------------------------------------

def evaluate_vqvae(model, val_loader, device, vqvae_config, num_batches=None):
    """Evaluate VQ-VAE on validation set."""
    model.eval()
    total_recon = 0.0
    total_vq = 0.0
    total_usage = 0.0
    n = 0

    with torch.no_grad():
        for i, images in enumerate(val_loader):
            if num_batches and i >= num_batches:
                break
            images = images.to(device)
            recon, vq_loss, indices = model(images)

            recon_loss = nn.functional.mse_loss(recon, images).item()
            total_recon += recon_loss
            total_vq += vq_loss.item()
            # Compute codebook usage
            unique = indices.unique().numel()
            total_usage += unique / vqvae_config.codebook_size
            n += 1

    model.train()
    if n == 0:
        return float("nan"), float("nan"), 0.0
    return total_recon / n, total_vq / n, total_usage / n


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_vqvae_loop(model, train_loader, val_loader, optimizer, device,
                     vqvae_config, num_epochs, log_freq, eval_freq, eval_iter,
                     model_name, save_every_n_epochs, save_every_n_iters=None,
                     start_epoch=0, start_global_step=-1,
                     prev_train_losses=None, prev_val_losses=None):
    """Main VQ-VAE training loop."""
    train_losses = list(prev_train_losses or [])
    val_losses = list(prev_val_losses or [])
    global_step = start_global_step

    commitment_weight = vqvae_config.commitment_weight

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, images in enumerate(train_loader):
            current_abs_step = epoch * len(train_loader) + i
            if current_abs_step <= start_global_step:
                continue

            images = images.to(device)
            optimizer.zero_grad()

            recon, vq_loss, indices = model(images)
            total_loss, loss_dict = vqvae_loss(
                recon, images, vq_loss,
                recon_weight=1.0,
                commitment_weight=commitment_weight,
            )

            total_loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % log_freq == 0:
                # Compute codebook usage
                usage = indices.unique().numel() / vqvae_config.codebook_size
                write_status(
                    f"PROGRESS epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"loss={total_loss.item():.4f} recon={loss_dict['recon_loss']:.4f} "
                    f"vq={loss_dict['commitment_loss']:.4f} usage={usage:.2%}"
                )

            if global_step % eval_freq == 0:
                recon_l, vq_l, usage = evaluate_vqvae(
                    model, val_loader, device, vqvae_config, eval_iter
                )
                train_losses.append(loss_dict["total_loss"])
                val_losses.append(recon_l + vq_l)
                write_status(
                    f"EVAL epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"val_recon={recon_l:.4f} val_vq={vq_l:.4f} "
                    f"codebook_usage={usage:.2%}"
                )

            if save_every_n_iters and save_every_n_iters > 0 and global_step > 0:
                if global_step % save_every_n_iters == 0:
                    ckpt_path = save_vqvae_checkpoint(
                        model_name, model, optimizer, epoch, global_step,
                        train_losses, val_losses, tag=f"step{global_step}")
                    write_status(f"CHECKPOINT saved at step {global_step} -> {ckpt_path}")

        # Epoch checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            ckpt_path = save_vqvae_checkpoint(
                model_name, model, optimizer, epoch + 1, global_step,
                train_losses, val_losses)
            write_status(f"CHECKPOINT saved at epoch {epoch+1} -> {ckpt_path}")

    # Final
    ckpt_path = save_vqvae_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        train_losses, val_losses, tag="final")
    save_vqvae_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        train_losses, val_losses)
    write_status(f"FINAL checkpoint saved -> {ckpt_path}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YALLM VQ-VAE Pretraining (Phase 1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train VQ-VAE
  python train_vqvae.py --model-name my-vqvae --config configs/vqvae_default.json \\
      --data-dir /path/to/images/ --epochs 100 --batch-size 16

  # Resume
  python train_vqvae.py --model-name my-vqvae --resume --epochs 200
""",
    )

    parser.add_argument("--model-name", required=True,
                        help="Name for this model (creates models/<name>/)")
    parser.add_argument("--config", default=None,
                        help="Path to VQ-VAE config JSON")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Path to directory containing training images")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from latest checkpoint")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--save-every", type=int, default=10,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--save-iters", type=int, default=0,
                        help="Save checkpoint every N iterations (0=disabled)")
    parser.add_argument("--eval-freq", type=int, default=100,
                        help="Evaluate every N steps")
    parser.add_argument("--log-freq", type=int, default=10,
                        help="Log every N steps")
    parser.add_argument("--eval-iter", type=int, default=5,
                        help="Batches per evaluation")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")
    parser.add_argument("--optimizer", default="adamw",
                        choices=["adamw", "adamw8bit"],
                        help="Optimizer (default: adamw)")
    parser.add_argument("--checkpointing", action="store_true",
                        help="Enable gradient checkpointing")

    args = parser.parse_args()

    # Status file
    status_file = get_status_file(args.model_name)
    set_status_file(status_file)
    if not args.resume:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            f.write("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_status(f"START [VQVAE] device={device} model_name={args.model_name}")

    if args.resume:
        write_status("RESUME loading config and checkpoint...")
        full_cfg = load_model_config(args.model_name)
        train_cfg = full_cfg["training"]
        vqvae_config = VQVAEConfig.from_dict(full_cfg["vqvae"])

        total_epochs = args.epochs if args.epochs != 100 else train_cfg["epochs"]
        data_dir = train_cfg["data_dir"]

        model = VQVAE(vqvae_config).to(device)
        opt_name = train_cfg.get("optimizer", "adamw")
        optimizer = _create_optimizer(model, opt_name, train_cfg["lr"])
        ckpt_meta = load_vqvae_checkpoint(args.model_name, model, optimizer)
        model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt_meta["epoch"]
        start_step = ckpt_meta["global_step"]
        prev_tl = ckpt_meta["train_losses"]
        prev_vl = ckpt_meta["val_losses"]
        batch_size = train_cfg["batch_size"]

        if start_epoch >= total_epochs:
            write_status(f"Already trained {start_epoch} epochs (target={total_epochs}).")
            return

        write_status(f"RESUMED from epoch={start_epoch+1} step={start_step}")

    else:
        if not args.config:
            parser.error("--config is required for new VQ-VAE training")
        if not args.data_dir:
            parser.error("--data-dir is required for new VQ-VAE training")

        vqvae_config = VQVAEConfig.load(args.config)
        if args.checkpointing:
            vqvae_config.grad_checkpointing = True
        write_status(f"CONFIG loaded from {args.config}")

        model = VQVAE(vqvae_config).to(device)
        write_status(f"MODEL created:\n{model.summary()}")

        # Save config
        train_cfg = {
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
            "model_type": "vqvae",
            "vqvae": vqvae_config.to_dict(),
            "training": train_cfg,
        }
        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")

        total_epochs = args.epochs
        data_dir = args.data_dir
        batch_size = args.batch_size

        optimizer = _create_optimizer(model, args.optimizer, args.lr)
        start_epoch, start_step = 0, -1
        prev_tl, prev_vl = [], []

    # Load data
    train_loader, val_loader, data_info = create_image_dataloaders(
        data_dir,
        image_size=vqvae_config.image_size,
        image_channels=vqvae_config.image_channels,
        batch_size=batch_size,
        num_workers=args.num_workers,
    )
    write_status(f"DATA: {data_info}")
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} batches")
    write_status(f"MODEL params={model.param_count():,}")

    # Train
    t0 = time.time()
    save_every = train_cfg.get("save_every", args.save_every)
    save_iters = train_cfg.get("save_iters", args.save_iters)
    log_freq = train_cfg.get("log_freq", args.log_freq)
    eval_freq = train_cfg.get("eval_freq", args.eval_freq)
    eval_iter = train_cfg.get("eval_iter", args.eval_iter)

    train_vqvae_loop(
        model, train_loader, val_loader, optimizer, device,
        vqvae_config, total_epochs,
        log_freq=log_freq, eval_freq=eval_freq, eval_iter=eval_iter,
        model_name=args.model_name,
        save_every_n_epochs=save_every,
        save_every_n_iters=save_iters,
        start_epoch=start_epoch,
        start_global_step=start_step,
        prev_train_losses=prev_tl,
        prev_val_losses=prev_vl,
    )

    elapsed = (time.time() - t0) / 60
    write_status(f"DONE VQ-VAE training completed in {elapsed:.2f} min")


def _create_optimizer(model, opt_name, lr, weight_decay=0.01):
    """Create optimizer for VQ-VAE training."""
    opt_name = opt_name.lower()
    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_name == "adamw8bit":
        import bitsandbytes as bnb
        return bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")


if __name__ == "__main__":
    main()
