#!/usr/bin/env python3
"""
YALLM Instruction Fine-Tuning Script (SFT)
Fine-tune a pretrained GPT-2 model on Alpaca-style instruction data.

Usage:
  python finetune.py --data instruction-data.json --epochs 2
  python finetune.py --data instruction-data.json --model-size 355M --epochs 3 --lr 5e-5
"""

import argparse
import json
import time
from functools import partial

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import (
    GPT2, EOT_TOKEN_ID, build_config, download_and_load_pretrained,
    generate, get_tokenizer, load_weights_into_gpt, text_to_token_ids,
    token_ids_to_text,
)

_cfg = {"status_file": "finetune_status.txt"}


def write_status(msg):
    """Append a timestamped message to the status file for external monitoring."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    with open(_cfg["status_file"], "a") as f:
        f.write(line)
    print(msg)


# ---------------------------------------------------------------------------
#  Instruction dataset & collation
# ---------------------------------------------------------------------------

def format_input_alpaca(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text


class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.encoded_texts = []
        for entry in data:
            full = format_input_alpaca(entry) + f"\n\n### Response:\n{entry['output']}"
            self.encoded_texts.append(tokenizer.encode(full))

    def __getitem__(self, index):
        return self.encoded_texts[index]

    def __len__(self):
        return len(self.data)


def instruction_collate_fn(batch, pad_token_id=EOT_TOKEN_ID,
                           ignore_index=-100, allowed_max_length=None,
                           device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        new_item = item.copy()
        new_item += [pad_token_id]
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        inputs  = torch.tensor(padded[:-1])
        targets = torch.tensor(padded[1:])
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index
        if allowed_max_length is not None:
            inputs  = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]
        inputs_lst.append(inputs)
        targets_lst.append(targets)
    return torch.stack(inputs_lst).to(device), torch.stack(targets_lst).to(device)


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

def train_sft(model, train_loader, val_loader, optimizer, device,
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
                    f"SFT epoch={epoch+1} step={global_step:06d} "
                    f"tokens={tokens_seen} train_loss={tl:.4f} val_loss={vl:.4f}"
                )

        # Generate a sample after each epoch
        model.eval()
        ctx_size = model.cfg["context_length"]
        enc = text_to_token_ids(start_context, tokenizer).to(device)
        with torch.no_grad():
            gen = generate(model, enc, 50, ctx_size, eos_id=EOT_TOKEN_ID)
        sample = token_ids_to_text(gen, tokenizer).replace("\n", " ")
        write_status(f"SAMPLE epoch={epoch+1}: {sample}")
        model.train()

    return train_losses, val_losses


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="YALLM Instruction Fine-Tune")
    parser.add_argument("--data", required=True,
                        help="Path to instruction JSON (Alpaca format)")
    parser.add_argument("--model-size", default="355M",
                        choices=["124M", "355M", "774M", "1558M"])
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--output", type=str, default=None,
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
    cfg = build_config(args.model_size)
    write_status(f"CONFIG {cfg}")

    # Load instruction data
    with open(args.data, "r") as f:
        data = json.load(f)
    write_status(f"DATA loaded {len(data)} instruction entries from {args.data}")

    # Download / load pretrained OpenAI weights
    write_status(f"WEIGHTS downloading/loading pretrained {args.model_size}...")
    settings, params = download_and_load_pretrained(args.model_size)
    model = GPT2(cfg)
    load_weights_into_gpt(model, params)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    write_status(f"MODEL {args.model_size} with {n_params:,} parameters loaded on {device}")

    # Build data loaders
    n_train = int(len(data) * 0.85)
    n_test  = int(len(data) * 0.10)
    train_data = data[:n_train]
    test_data  = data[n_train:n_train + n_test]
    val_data   = data[n_train + n_test:]

    collate = partial(instruction_collate_fn, device=device, allowed_max_length=1024)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer),
                              batch_size=args.batch_size, collate_fn=collate,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer),
                            batch_size=args.batch_size, collate_fn=collate,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(InstructionDataset(test_data, tokenizer),
                             batch_size=args.batch_size, collate_fn=collate,
                             shuffle=False, drop_last=False)
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} "
                 f"test={len(test_loader)} batches")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=0.1)

    start_ctx = format_input_alpaca(data[0]) if data else "Hello"

    t0 = time.time()
    train_sft(model, train_loader, val_loader, optimizer, device,
              args.epochs, eval_freq=5, eval_iter=5,
              start_context=start_ctx, tokenizer=tokenizer)
    elapsed = (time.time() - t0) / 60
    write_status(f"DONE training completed in {elapsed:.2f} min")

    # Final test-set loss
    model.eval()
    with torch.no_grad():
        test_loss = calc_loss_loader(test_loader, model, device)
    write_status(f"TEST loss={test_loss:.4f}")

    out = args.output or f"gpt2-{args.model_size}-sft.pth"
    torch.save(model.state_dict(), out)
    write_status(f"SAVED model to {out}")


if __name__ == "__main__":
    main()
