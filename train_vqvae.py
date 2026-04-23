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

try:
    import torchvision
    from torchvision.utils import save_image, make_grid
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

from pretrain_data.image_dataset import create_image_dataloaders

from model import (
    write_status, save_model_config, load_model_config,
    get_model_dir, MODELS_DIR,
)

# Shared training utilities
from training_utils import (
    create_optimizer, setup_status_file, setup_device,
    migrate_optimizer_to_device, handle_resume_no_checkpoint, has_checkpoint,
    get_train_params, EarlyStopper,
    add_common_training_args, get_default_training_config,
)


# ---------------------------------------------------------------------------
#  Checkpointing (VQ-VAE specific)
# ---------------------------------------------------------------------------

def save_vqvae_checkpoint(model_name, model, optimizer, epoch, global_step,
                          train_losses, val_losses, tag=None):
    """Save a VQ-VAE checkpoint."""
    import os
    d = get_model_dir(model_name)
    fname = f"checkpoint_step{global_step}.pt" if tag is None else f"checkpoint_{tag}.pt"
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

def evaluate_vqvae(model, val_loader, device, vqvae_config, num_batches=None, use_bf16=False):
    """Evaluate VQ-VAE on validation set."""
    model.eval()
    total_recon = 0.0
    total_vq = 0.0
    total_usage = 0.0
    n = 0
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            for i, images in enumerate(val_loader):
                if num_batches and i >= num_batches:
                    break
                images = images.to(device)
                recon, vq_loss, indices = model(images)

                recon_loss = nn.functional.mse_loss(recon, images).item()
                total_recon += recon_loss
                total_vq += vq_loss.item()
                # Compute codebook usage.
                unique = indices.cpu().unique().numel()
                total_usage += unique / vqvae_config.codebook_size
                n += 1
    model.train()
    if n == 0:
        return float("nan"), float("nan"), 0.0
    return total_recon / n, total_vq / n, total_usage / n


def save_reconstruction_samples(model, loader, device, model_name, step):
    """Save a grid of original vs. reconstructed images from the validation set."""
    if not HAS_TORCHVISION:
        return

    model.eval()
    with torch.no_grad():
        # Get a single batch
        images = next(iter(loader))
        images = images[:8].to(device)  # Take up to 8 images

        # Reconstruct
        recon, _, _ = model(images)

        # Prepare grid: [orig1, recon1, orig2, recon2, ...]
        combined = torch.stack([images, recon], dim=1).flatten(0, 1)
        grid = make_grid(combined, nrow=4, normalize=True, value_range=(-1, 1))

        # Save
        sample_dir = get_model_dir(model_name) / "samples"
        sample_dir.mkdir(exist_ok=True)
        save_path = sample_dir / f"step_{step:06d}.png"
        save_image(grid, save_path)
        write_status(f"SAMPLE images saved to {save_path}")

    model.train()


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_vqvae_loop(model, train_loader, val_loader, optimizer, device,
                     vqvae_config, num_epochs, log_freq, eval_freq, eval_iter,
                     model_name, save_every_n_epochs, save_every_n_iters=None,
                     start_epoch=0, start_global_step=-1,
                     prev_train_losses=None, prev_val_losses=None,
                     early_stopper=None, use_bf16=False):
    """Main VQ-VAE training loop."""
    train_losses = list(prev_train_losses or [])
    val_losses = list(prev_val_losses or [])
    global_step = start_global_step
    completed_epochs = start_epoch

    commitment_weight = vqvae_config.commitment_weight

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, images in enumerate(train_loader):
            current_abs_step = epoch * len(train_loader) + i
            if current_abs_step <= start_global_step:
                continue

            images = images.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
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
                # Compute codebook usage. (Bypass ROCm GPU unique() crash via .cpu() fallback)
                usage = indices.cpu().unique().numel() / vqvae_config.codebook_size
                write_status(
                    f"PROGRESS epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"loss={total_loss.item():.4f} recon={loss_dict['recon_loss']:.4f} "
                    f"vq={loss_dict['commitment_loss']:.4f} usage={usage:.2%}"
                )

            if global_step % eval_freq == 0:
                recon_l, vq_l, usage = evaluate_vqvae(
                    model, val_loader, device, vqvae_config, eval_iter, use_bf16=use_bf16
                )
                val_loss = recon_l + vq_l
                train_losses.append(loss_dict["total_loss"])
                val_losses.append(val_loss)
                write_status(
                    f"EVAL epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"val_loss={recon_l + vq_l:.4f} val_recon={recon_l:.4f} val_vq={vq_l:.4f} "
                    f"codebook_usage={usage:.2%}"
                )
                # Save visual samples
                save_reconstruction_samples(model, val_loader, device, model_name, global_step)

                # Early stopping check
                if early_stopper is not None:
                    if early_stopper.check(val_loss, global_step, completed_epochs):
                        # Ensure all GPU work is done before saving and exiting
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        write_status(
                            f"EARLY_STOP triggered at step {global_step} "
                            f"({early_stopper.status_message()})"
                        )
                        ckpt_path = save_vqvae_checkpoint(
                            model_name, model, optimizer, epoch + 1, global_step,
                            train_losses, val_losses, tag="early_stop")
                        save_vqvae_checkpoint(
                            model_name, model, optimizer, epoch + 1, global_step,
                            train_losses, val_losses)
                        write_status(f"CHECKPOINT saved (early stop) -> {ckpt_path}")
                        return train_losses, val_losses

            if save_every_n_iters and save_every_n_iters > 0 and global_step > 0:
                if global_step % save_every_n_iters == 0:
                    ckpt_path = save_vqvae_checkpoint(
                        model_name, model, optimizer, epoch, global_step,
                        train_losses, val_losses, tag=f"step{global_step}")
                    write_status(f"CHECKPOINT saved at step {global_step} -> {ckpt_path}")

        # Epoch completed
        completed_epochs = epoch + 1

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
    )
    # Common training args
    add_common_training_args(parser)

    # VQ-VAE specific args
    vq_group = parser.add_argument_group("VQ-VAE Specific")
    vq_group.add_argument("--data-dir", type=str, default=None,
                        help="Path to directory containing training images")
    vq_group.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers")

    args = parser.parse_args()

    # ---- Setup ----
    setup_status_file(args.model_name, resume=args.resume)
    device = setup_device()
    write_status(f"START [VQVAE] device={device} model_name={args.model_name}")

    if args.resume:
        checkpoint_exists = has_checkpoint(args.model_name)
        if checkpoint_exists:
            write_status("RESUME loading config and checkpoint...")
            full_cfg = load_model_config(args.model_name)
        else:
            full_cfg = handle_resume_no_checkpoint(args.model_name)

        train_cfg = full_cfg["training"]
        vqvae_config = VQVAEConfig.from_dict(full_cfg["vqvae"])
        if args.checkpointing:
            vqvae_config.grad_checkpointing = True

        data_dir = train_cfg["data_dir"]
        model = VQVAE(vqvae_config).to(device)

        if not checkpoint_exists:
            opt_name = train_cfg.get("optimizer", "adamw")
            optimizer = create_optimizer(model, opt_name, train_cfg["lr"],
                                         weight_decay=0.01)

        start_epoch, start_step = 0, -1
        prev_tl, prev_vl = [], []
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

        # Set initial training config with defaults
        train_cfg = get_default_training_config("vqvae", args)
        train_cfg["data_dir"] = args.data_dir

        full_cfg = {
            "model_type": "vqvae",
            "vqvae": vqvae_config.to_dict(),
            "training": train_cfg,
        }
        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")

        data_dir = args.data_dir
        optimizer = create_optimizer(model, train_cfg["optimizer"], train_cfg["lr"],
                                     weight_decay=0.01)
        start_epoch, start_step = 0, -1
        prev_tl, prev_vl = [], []
        checkpoint_exists = False

    # ---- Train ----
    t0 = time.time()

    # ---- Merge Parameters ----
    # Resolve final training parameters
    tp = get_train_params("vqvae", train_cfg, args, has_checkpoint=checkpoint_exists)

    # ---- Resume Checkpoint Load ----
    if args.resume and checkpoint_exists:
        optimizer = create_optimizer(model, tp["optimizer"], tp["lr"], weight_decay=0.01)
        ckpt_meta = load_vqvae_checkpoint(args.model_name, model, optimizer)
        model.to(device)
        migrate_optimizer_to_device(optimizer, device)

        start_epoch = ckpt_meta["epoch"]
        start_step = ckpt_meta["global_step"]
        prev_tl = ckpt_meta["train_losses"]
        prev_vl = ckpt_meta["val_losses"]

        if start_epoch >= tp["epochs"]:
            write_status(f"Already trained {start_epoch} epochs (target={tp['epochs']}).")
            return
        write_status(f"RESUMED from epoch={start_epoch+1} step={start_step} -> training to epoch {tp['epochs']}")

    # Load data
    train_loader, val_loader, data_info = create_image_dataloaders(
        data_dir,
        image_size=vqvae_config.image_size,
        image_channels=vqvae_config.image_channels,
        batch_size=tp["batch_size"],
        num_workers=tp["num_workers"],
        pin_memory=tp["pin_memory"],
        prefetch_factor=tp["prefetch_factor"],
    )
    write_status(f"DATA: {data_info}")
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} batches")
    write_status(f"MODEL params={model.param_count():,}")

    # Early stopping
    stopper = EarlyStopper(
        patience_evals=tp["patience"],
        min_delta=tp["min_delta"],
        min_epochs=tp["min_epochs"],
        window_size=tp["window_size"]
    )

    # Initial sample
    save_reconstruction_samples(model, val_loader, device, args.model_name, max(0, start_step))

    # ---- Compile ----
    if tp["compile"]:
        write_status("torch.compile: Compiling model... (This will take a few minutes)")
        model = torch.compile(model)

    train_vqvae_loop(
        model, train_loader, val_loader, optimizer, device,
        vqvae_config, tp["epochs"],
        log_freq=tp["log_freq"], eval_freq=tp["eval_freq"], eval_iter=tp["eval_iter"],
        model_name=args.model_name,
        save_every_n_epochs=tp["save_every"],
        save_every_n_iters=tp["save_iters"],
        start_epoch=start_epoch,
        start_global_step=start_step,
        prev_train_losses=prev_tl,
        prev_val_losses=prev_vl,
        early_stopper=stopper,
        use_bf16=tp["bf16"],
    )
    elapsed = (time.time() - t0) / 60
    write_status(f"DONE VQ-VAE training completed in {elapsed:.2f} min")


if __name__ == "__main__":
    main()
