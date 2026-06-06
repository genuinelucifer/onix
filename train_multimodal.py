#!/usr/bin/env python3
"""
Onix Multi-Modal LLM Training Script (Phase 2)

Train a decoder-only transformer to generate visual tokens from text prompts.
Uses a frozen VQ-VAE (trained in Phase 1) as the image tokenizer.

Features:
  - Early stopping when loss stagnates (after at least 1 epoch)
  - Graceful --resume when no checkpoint exists (uses saved config)
  - CUDA synchronization and explicit cleanup for stability on ROCm

Usage:
    python train_multimodal.py --model-name my-imggen \
        --config configs/multimodal_pixelart.json \
        --data-dir /path/to/image_text_pairs/ \
        --epochs 50
"""

import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.utils import save_image

from architecture.config import ModelConfig, VQVAEConfig, MultiModalConfig
from architecture.model import CausalLM
from architecture.vqvae import VQVAE
from architecture.losses import masked_cross_entropy, calc_loss_batch_masked
from architecture.generate import generate_image

from pretrain_data.multimodal_dataset import create_multimodal_dataloaders

from model import (
    get_tokenizer, write_status, save_model_config, load_model_config,
    save_checkpoint, load_checkpoint,
    get_model_dir, MODELS_DIR,
)

# Shared training utilities
from training_utils import (
    create_optimizer, setup_status_file, setup_device, setup_performance,
    migrate_optimizer_to_device, handle_resume_no_checkpoint, has_checkpoint,
    get_train_params, EarlyStopper,
    add_common_training_args, get_default_training_config,
)


# ---------------------------------------------------------------------------
#  Loss helpers
# ---------------------------------------------------------------------------

def calc_loss_batch_mm(inp, tgt, loss_mask, model, device):
    """Forward pass + masked cross-entropy for multi-modal training."""
    inp, tgt = inp.to(device), tgt.to(device)
    loss_mask = loss_mask.to(device)
    logits = model(inp)
    return masked_cross_entropy(logits, tgt, loss_mask)


def evaluate_mm(model, val_loader, device, num_batches=None, use_bf16=False):
    """Evaluate on validation set with masked loss."""
    model.eval()
    total = 0.0
    n = 0
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            # Use iterator to allow early break without worker leakage
            loader_iter = iter(val_loader)
            for i in range(num_batches if num_batches else len(val_loader)):
                try:
                    inp, tgt, mask = next(loader_iter)
                except StopIteration:
                    break
                total += calc_loss_batch_mm(inp, tgt, mask, model, device).item()
                n += 1
    model.train()
    return total / n if n > 0 else float("nan")


# ---------------------------------------------------------------------------
#  VQ-VAE loading
# ---------------------------------------------------------------------------

def load_frozen_vqvae(vqvae_config, checkpoint_path, device):
    """Load a trained VQ-VAE with frozen weights."""
    model = VQVAE(vqvae_config)
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    write_status(f"VQ-VAE loaded and frozen from {checkpoint_path}")
    return model


# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_multimodal_loop(model, train_loader, val_loader, optimizer, device,
                          mm_config, num_epochs, log_freq, eval_freq, eval_iter,
                          model_name, save_every_n_epochs, save_every_n_iters=None,
                          start_epoch=0, start_global_step=-1,
                          prev_train_losses=None, prev_val_losses=None,
                          frozen_vqvae=None, tokenizer=None,
                          early_stopper=None, use_bf16=False, save_limit=3):
    """Main multi-modal training loop."""
    train_losses = list(prev_train_losses or [])
    val_losses = list(prev_val_losses or [])
    global_step = start_global_step

    # Track completed epochs for early stopping
    completed_epochs = start_epoch

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, (inp, tgt, loss_mask) in enumerate(train_loader):
            current_abs_step = epoch * len(train_loader) + i
            if current_abs_step <= start_global_step:
                continue

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss = calc_loss_batch_mm(inp, tgt, loss_mask, model, device)
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % log_freq == 0:
                write_status(
                    f"PROGRESS epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"loss={loss.item():.4f}"
                )

            if global_step % eval_freq == 0:
                vl = evaluate_mm(model, val_loader, device, eval_iter, use_bf16=use_bf16)
                train_losses.append(loss.item())
                val_losses.append(vl)
                write_status(
                    f"EVAL epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"val_loss={vl:.4f}"
                )

                # Early stopping check
                if early_stopper is not None:
                    if early_stopper.check(vl, global_step, completed_epochs):
                        # Ensure all GPU work is done before saving and exiting to avoid IPC errors
                        if device.type == "cuda":
                            torch.cuda.synchronize()

                        write_status(
                            f"EARLY_STOP triggered at step {global_step} "
                            f"({early_stopper.status_message()})"
                        )
                        ckpt_path = save_checkpoint(
                            model_name, model, optimizer, epoch + 1, global_step,
                            0, train_losses, val_losses, tag="early_stop", keep_last_n=save_limit)
                        save_checkpoint(
                            model_name, model, optimizer, epoch + 1, global_step,
                            0, train_losses, val_losses, keep_last_n=save_limit)
                        write_status(f"CHECKPOINT saved (early stop) -> {ckpt_path}")

                        # Final sample before return
                        if frozen_vqvae is not None and tokenizer is not None:
                            try:
                                model.eval()
                                sample_prompt = "A colorful pixel art character"
                                image, _ = generate_image(
                                    model, frozen_vqvae, sample_prompt, tokenizer, mm_config,
                                    temperature=0.9, top_k=100,
                                )
                                write_status(f"FINAL SAMPLE: generated image from \"{sample_prompt}\"")
                                try:
                                    sample_dir = get_model_dir(model_name) / "samples"
                                    save_image((image.squeeze(0) + 1) / 2, sample_dir / "early_stop.png")
                                except Exception: pass
                            except Exception as e:
                                write_status(f"FINAL SAMPLE failed: {e}")

                        return train_losses, val_losses

            if save_every_n_iters and save_every_n_iters > 0 and global_step > 0:
                if global_step % save_every_n_iters == 0:
                    ckpt_path = save_checkpoint(
                        model_name, model, optimizer, epoch, global_step,
                        0, train_losses, val_losses, tag=f"step{global_step}", keep_last_n=save_limit)
                    write_status(f"CHECKPOINT saved at step {global_step} -> {ckpt_path}")

        # Epoch completed
        completed_epochs = epoch + 1

        # Generate a sample image after each epoch (if VQ-VAE available)
        if frozen_vqvae is not None and tokenizer is not None:
            try:
                model.eval()
                sample_prompt = "A colorful pixel art character"
                image, _ = generate_image(
                    model, frozen_vqvae, sample_prompt, tokenizer, mm_config,
                    temperature=0.9, top_k=100,
                )
                write_status(
                    f"SAMPLE epoch={epoch+1}: generated image from \"{sample_prompt}\" "
                    f"(shape={tuple(image.shape)})"
                )
                try:
                    sample_dir = get_model_dir(model_name) / "samples"
                    sample_dir.mkdir(exist_ok=True)
                    save_image(
                        (image.squeeze(0) + 1) / 2,
                        sample_dir / f"epoch_{epoch+1:03d}.png"
                    )
                except Exception:
                    pass
                model.train()
            except Exception as e:
                write_status(f"SAMPLE epoch={epoch+1}: generation failed: {e}")
                model.train()

        # Epoch checkpoint
        if (epoch + 1) % save_every_n_epochs == 0:
            ckpt_path = save_checkpoint(
                model_name, model, optimizer, epoch + 1, global_step,
                0, train_losses, val_losses, keep_last_n=save_limit)
            write_status(f"CHECKPOINT saved at epoch {epoch+1} -> {ckpt_path}")

    # Final
    ckpt_path = save_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        0, train_losses, val_losses, tag="final", keep_last_n=save_limit)
    save_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        0, train_losses, val_losses, keep_last_n=save_limit)
    write_status(f"FINAL checkpoint saved -> {ckpt_path}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Onix Multi-Modal LLM Training (Phase 2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # Common training args
    add_common_training_args(parser)

    # Multi-modal specific args
    mm_group = parser.add_argument_group("Multi-modal Specific")
    mm_group.add_argument("--data-dir", type=str, default=None,
                        help="Path to image+text pairs or pre-encoded shards")
    mm_group.add_argument("--pre-encoded", action="store_true",
                        help="Data is pre-encoded (from encode_dataset.py)")
    mm_group.add_argument("--use-sdpa", action="store_true", default=None,
                        help="Use Scaled Dot Product Attention (default: True)")
    mm_group.add_argument("--no-sdpa", action="store_false", dest="use_sdpa",
                        help="Disable SDPA")

    args = parser.parse_args()

    t0 = time.time()
    # ---- Setup ----
    setup_status_file(args.model_name, resume=args.resume)
    setup_performance()
    device = setup_device()
    write_status(f"START [MULTIMODAL] device={device} model_name={args.model_name}")

    tokenizer = get_tokenizer()

    if args.resume:
        checkpoint_exists = has_checkpoint(args.model_name)

        if checkpoint_exists:
            write_status("RESUME loading config and checkpoint...")
            full_cfg = load_model_config(args.model_name)
        else:
            full_cfg = handle_resume_no_checkpoint(args.model_name)

        train_cfg = full_cfg["training"]
        mm_config = MultiModalConfig.from_dict(full_cfg["multimodal"])

        data_dir = train_cfg.get("data_dir")
        pre_encoded = train_cfg.get("pre_encoded", False)

        transformer_config = mm_config.build_transformer_config()
        model = CausalLM(transformer_config).to(device)
        start_epoch, start_step, prev_tl, prev_vl = 0, -1, [], []
    else:
        if not args.config:
            parser.error("--config is required for new multi-modal training")
        if not args.data_dir:
            parser.error("--data-dir is required for new multi-modal training")

        mm_config = MultiModalConfig.load(args.config)
        write_status(f"CONFIG loaded from {args.config}")

        transformer_config = mm_config.build_transformer_config()
        if args.checkpointing:
            transformer_config.grad_checkpointing = True
        if args.use_sdpa is not None:
            transformer_config.use_sdpa = args.use_sdpa

        torch.manual_seed(123)
        model = CausalLM(transformer_config).to(device)
        write_status(f"MODEL created:\n{model.summary()}")

        train_cfg = get_default_training_config("multimodal", args)
        train_cfg["data_dir"] = args.data_dir
        train_cfg["pre_encoded"] = args.pre_encoded

        save_model_config(args.model_name, {
            "model_type": "multimodal",
            "multimodal": mm_config.to_dict(),
            "architecture": transformer_config.to_dict(),
            "training": train_cfg,
        })

        data_dir = args.data_dir
        pre_encoded = args.pre_encoded
        start_epoch, start_step, prev_tl, prev_vl, checkpoint_exists = 0, -1, [], [], False

    # Load frozen VQ-VAE
    frozen_vqvae = None
    if mm_config.vqvae_checkpoint and Path(mm_config.vqvae_checkpoint).exists():
        frozen_vqvae = load_frozen_vqvae(
            mm_config.vqvae, mm_config.vqvae_checkpoint, device
        )
    else:
        write_status(f"WARNING: VQ-VAE checkpoint not found at '{mm_config.vqvae_checkpoint}'.")

    tp = get_train_params("multimodal", train_cfg, args, has_checkpoint=checkpoint_exists)

    # Load data
    train_loader, val_loader, data_info = create_multimodal_dataloaders(
        data_dir, tokenizer, frozen_vqvae, mm_config,
        batch_size=tp["batch_size"], device=device,
        pre_encoded=pre_encoded,
        num_workers=tp["num_workers"],
        pin_memory=tp["pin_memory"],
        prefetch_factor=tp["prefetch_factor"],
    )
    write_status(f"DATA: {data_info}")
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} batches")

    if args.resume and checkpoint_exists:
        optimizer = create_optimizer(model, tp["optimizer"], tp["lr"])
        ckpt_meta = load_checkpoint(args.model_name, model, optimizer)
        model.to(device)
        migrate_optimizer_to_device(optimizer, device)
        start_epoch, start_step, prev_tl, prev_vl = ckpt_meta["epoch"], ckpt_meta["global_step"], ckpt_meta["train_losses"], ckpt_meta["val_losses"]
        if start_epoch >= tp["epochs"]:
            write_status(f"Already trained {start_epoch} epochs (target={tp['epochs']}).")
            return
    else:
        optimizer = create_optimizer(model, tp["optimizer"], tp["lr"])

    early_stopper = None
    if tp["patience"] > 0:
        early_stopper = EarlyStopper(
            patience_evals=tp["patience"], min_delta=tp["min_delta"],
            min_epochs=tp["min_epochs"], window_size=tp["window_size"],
        )

    if tp["compile"]:
        write_status("torch.compile: Compiling model...")
        model = torch.compile(model)

    train_multimodal_loop(
        model, train_loader, val_loader, optimizer, device,
        mm_config, tp["epochs"],
        log_freq=tp["log_freq"], eval_freq=tp["eval_freq"], eval_iter=tp["eval_iter"],
        model_name=args.model_name,
        save_every_n_epochs=tp["save_every"],
        save_every_n_iters=tp["save_iters"],
        start_epoch=start_epoch,
        start_global_step=start_step,
        prev_train_losses=prev_tl,
        prev_val_losses=prev_vl,
        frozen_vqvae=frozen_vqvae,
        tokenizer=tokenizer,
        early_stopper=early_stopper,
        use_bf16=tp["bf16"],
        save_limit=tp["save_limit"],
    )
    
    elapsed = (time.time() - t0) / 60
    write_status(f"DONE multi-modal training completed in {elapsed:.2f} min")


if __name__ == "__main__":
    main()
