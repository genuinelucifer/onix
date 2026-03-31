#!/usr/bin/env python3
"""
YALLM Pretraining Script
Pretrain a GPT-2 model from scratch on a text file.

New run:
  python train.py --model-name my-gpt --data ../the-verdict.txt --epochs 20

Resume:
  python train.py --model-name my-gpt --resume
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import (
    GPT2, build_config, get_tokenizer, text_to_token_ids, token_ids_to_text,
    generate, write_status, set_status_file, get_model_dir, get_status_file,
    save_model_config, load_model_config, save_checkpoint, load_checkpoint,
    MODEL_CONFIGS,
)


# ---------------------------------------------------------------------------
#  Pretraining Dataset
# ---------------------------------------------------------------------------

class PretrainDataset(Dataset):
    def __init__(self, txt, tokenizer, window_length, stride):
        tokens = tokenizer.encode(txt)
        self.input_tokens = []
        self.output_tokens = []
        for i in range(0, len(tokens) - window_length, stride):
            self.input_tokens.append(torch.tensor(tokens[i:i + window_length]))
            self.output_tokens.append(torch.tensor(tokens[i + 1:i + 1 + window_length]))

    def __len__(self):
        return len(self.input_tokens)

    def __getitem__(self, idx):
        return self.input_tokens[idx], self.output_tokens[idx]


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

def train(model, train_loader, val_loader, optimizer, device,
          num_epochs, eval_freq, eval_iter, start_context, tokenizer,
          model_name, save_every_n_epochs,
          start_epoch=0, start_global_step=-1, start_tokens_seen=0,
          prev_train_losses=None, prev_val_losses=None):
    train_losses = list(prev_train_losses or [])
    val_losses = list(prev_val_losses or [])
    tokens_seen = start_tokens_seen
    global_step = start_global_step

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for inp, tgt in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(inp, tgt, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += inp.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                tl, vl = evaluate(model, train_loader, val_loader,
                                  device, eval_iter)
                train_losses.append(tl)
                val_losses.append(vl)
                write_status(
                    f"TRAIN epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"tokens={tokens_seen} train_loss={tl:.4f} val_loss={vl:.4f}"
                )

        # Generate a sample after each epoch
        model.eval()
        ctx_size = model.cfg["context_length"]
        enc = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            gen = generate(model, enc, 50, ctx_size)
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
    # Also update latest
    save_checkpoint(
        model_name, model, optimizer, num_epochs, global_step,
        tokens_seen, train_losses, val_losses)
    write_status(f"FINAL checkpoint saved -> {ckpt_path}")

    return train_losses, val_losses


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YALLM Pretrain",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New training run
  python train.py --model-name verdict-gpt --data ../the-verdict.txt --epochs 20

  # Resume from latest checkpoint
  python train.py --model-name verdict-gpt --resume

  # Resume and train for more epochs
  python train.py --model-name verdict-gpt --resume --epochs 50
""",
    )
    parser.add_argument("--model-name", required=True,
                        help="Name for this model (creates models/<name>/)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to training text file (required for new run)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--model-size", default="124M",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10,
                        help="Total epochs (not additional)")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--save-every", type=int, default=5,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval-freq", type=int, default=5,
                        help="Evaluate every N training steps")
    parser.add_argument("--eval-iter", type=int, default=5,
                        help="Number of batches per evaluation")
    parser.add_argument("--drop-rate", type=float, default=0.1)
    args = parser.parse_args()

    # ---- Setup status file ----
    status_file = get_status_file(args.model_name)
    set_status_file(status_file)
    if not args.resume:
        status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(status_file, "w") as f:
            f.write("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_status(f"START device={device} model_name={args.model_name} resume={args.resume}")

    tokenizer = get_tokenizer()

    if args.resume:
        # ---- Resume mode ----
        write_status("RESUME loading config and checkpoint...")
        full_cfg = load_model_config(args.model_name)
        model_cfg = full_cfg["model"]
        train_cfg = full_cfg["training"]

        # Allow overriding total epochs when resuming
        total_epochs = args.epochs if args.epochs != 10 else train_cfg["epochs"]
        data_path = train_cfg["data"]

        model = GPT2(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=train_cfg["lr"],
                                      weight_decay=0.1)
        ckpt_meta = load_checkpoint(args.model_name, model, optimizer)
        model.to(device)
        # Move optimizer state to device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        start_epoch = ckpt_meta["epoch"]
        start_step = ckpt_meta["global_step"]
        start_tokens = ckpt_meta["tokens_seen"]
        prev_tl = ckpt_meta["train_losses"]
        prev_vl = ckpt_meta["val_losses"]

        if start_epoch >= total_epochs:
            write_status(f"Already trained {start_epoch} epochs (target={total_epochs}). "
                         f"Increase --epochs to continue.")
            return

        write_status(f"RESUMED from epoch={start_epoch} step={start_step} "
                     f"tokens={start_tokens} -> training to epoch {total_epochs}")
    else:
        # ---- New run ----
        if args.data is None:
            parser.error("--data is required for a new training run")

        model_cfg = build_config(args.model_size,
                                 context_length=args.context_length,
                                 drop_rate=args.drop_rate)
        train_cfg = {
            "data": args.data,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "save_every": args.save_every,
            "eval_freq": args.eval_freq,
            "eval_iter": args.eval_iter,
        }
        total_epochs = args.epochs
        data_path = args.data

        # Save config
        full_cfg = {"model": model_cfg, "training": train_cfg}
        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")
        write_status(f"MODEL_CFG {model_cfg}")
        write_status(f"TRAIN_CFG {train_cfg}")

        torch.manual_seed(123)
        model = GPT2(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=0.1)
        start_epoch, start_step, start_tokens = 0, -1, 0
        prev_tl, prev_vl = [], []

    # ---- Load data ----
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()
    write_status(f"DATA loaded {len(text)} chars from {data_path}")

    split = int(0.9 * len(text))
    train_loader = DataLoader(
        PretrainDataset(text[:split], tokenizer,
                        model_cfg["context_length"], model_cfg["context_length"]),
        batch_size=train_cfg["batch_size"] if args.resume else args.batch_size,
        shuffle=True, drop_last=True)
    val_loader = DataLoader(
        PretrainDataset(text[split:], tokenizer,
                        model_cfg["context_length"], model_cfg["context_length"]),
        batch_size=train_cfg["batch_size"] if args.resume else args.batch_size,
        shuffle=False, drop_last=False)
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} batches")

    n_params = sum(p.numel() for p in model.parameters())
    write_status(f"MODEL params={n_params:,}")

    # ---- Train ----
    t0 = time.time()
    save_every = train_cfg.get("save_every", args.save_every) if args.resume else args.save_every
    eval_freq = train_cfg.get("eval_freq", args.eval_freq) if args.resume else args.eval_freq
    eval_iter = train_cfg.get("eval_iter", args.eval_iter) if args.resume else args.eval_iter

    train(model, train_loader, val_loader, optimizer, device,
          num_epochs=total_epochs,
          eval_freq=eval_freq,
          eval_iter=eval_iter,
          start_context="Every effort moves you",
          tokenizer=tokenizer,
          model_name=args.model_name,
          save_every_n_epochs=save_every,
          start_epoch=start_epoch,
          start_global_step=start_step,
          start_tokens_seen=start_tokens,
          prev_train_losses=prev_tl,
          prev_val_losses=prev_vl)

    elapsed = (time.time() - t0) / 60
    write_status(f"DONE training completed in {elapsed:.2f} min")


if __name__ == "__main__":
    main()
