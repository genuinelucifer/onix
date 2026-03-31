#!/usr/bin/env python3
"""
YALLM Pretraining Script
Pretrain a GPT-2 model from scratch on a text file.

Usage:
  python train.py --data the-verdict.txt --epochs 10
  python train.py --data the-verdict.txt --model-size 124M --epochs 20 --lr 4e-4
"""

import argparse
import time
from pathlib import Path

import tiktoken
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import GPT2, build_config, get_tokenizer, text_to_token_ids, token_ids_to_text, generate

_cfg = {"status_file": "train_status.txt"}


def write_status(msg):
    """Append a timestamped message to the status file for external monitoring."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(_cfg["status_file"], "a") as f:
        f.write(line)
    print(msg)


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
          num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses = [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
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
                    f"TRAIN epoch={epoch+1} step={global_step:06d} "
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

    return train_losses, val_losses


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YALLM Pretrain")
    parser.add_argument("--data", required=True, help="Path to training text file")
    parser.add_argument("--model-size", default="124M",
                        choices=["124M", "355M", "774M", "1558M"])
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--output", type=str, default="model_pretrained.pth",
                        help="Output checkpoint path")
    parser.add_argument("--status-file", type=str, default=_cfg["status_file"],
                        help="Path for status log (monitored by agent)")
    args = parser.parse_args()

    _cfg["status_file"] = args.status_file

    # Clear status file at start
    with open(_cfg["status_file"], "w") as f:
        f.write("")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    write_status(f"START device={device}")

    tokenizer = get_tokenizer()
    cfg = build_config(args.model_size,
                       context_length=args.context_length,
                       drop_rate=0.1)
    write_status(f"CONFIG {cfg}")

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
    write_status(f"DATA loaded {len(text)} chars from {args.data}")

    split = int(0.9 * len(text))
    train_loader = DataLoader(
        PretrainDataset(text[:split], tokenizer, cfg["context_length"],
                        cfg["context_length"]),
        batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(
        PretrainDataset(text[split:], tokenizer, cfg["context_length"],
                        cfg["context_length"]),
        batch_size=args.batch_size, shuffle=False, drop_last=False)
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} batches")

    torch.manual_seed(123)
    model = GPT2(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    write_status(f"MODEL {args.model_size} with {n_params:,} parameters")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.1)

    t0 = time.time()
    train(model, train_loader, val_loader, optimizer, device,
          args.epochs, eval_freq=5, eval_iter=5,
          start_context="Every effort moves you",
          tokenizer=tokenizer)
    elapsed = (time.time() - t0) / 60
    write_status(f"DONE training completed in {elapsed:.2f} min")

    torch.save(model.state_dict(), args.output)
    write_status(f"SAVED model to {args.output}")


if __name__ == "__main__":
    main()
