#!/usr/bin/env python3
"""
YALLM Instruction Fine-Tuning Script (SFT)
Fine-tune a pretrained GPT-2 model on Alpaca-style instruction data.

New run:
  python finetune.py --model-name my-sft --data ../instruction-data.json --epochs 2

Resume:
  python finetune.py --model-name my-sft --resume
"""

import argparse
import json
import time
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model import (
    GPT2, EOT_TOKEN_ID, build_config, download_and_load_pretrained,
    generate, get_tokenizer, load_weights_into_gpt, text_to_token_ids,
    token_ids_to_text, write_status, set_status_file, get_status_file,
    save_model_config, load_model_config, save_checkpoint, load_checkpoint,
    MODEL_CONFIGS,
)


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
                    f"SFT epoch={epoch+1}/{num_epochs} step={global_step:06d} "
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
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="YALLM Instruction Fine-Tune",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # New SFT run
  python finetune.py --model-name my-sft --data ../instruction-data.json --epochs 3

  # Resume from latest checkpoint
  python finetune.py --model-name my-sft --resume

  # Resume and train for more epochs
  python finetune.py --model-name my-sft --resume --epochs 10
""",
    )
    parser.add_argument("--model-name", required=True,
                        help="Name for this model (creates models/<name>/)")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to instruction JSON (required for new run)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from latest checkpoint")
    parser.add_argument("--model-size", default="355M",
                        choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--epochs", type=int, default=2,
                        help="Total epochs (not additional)")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--save-every", type=int, default=1,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--eval-freq", type=int, default=5)
    parser.add_argument("--eval-iter", type=int, default=5)
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

        total_epochs = args.epochs if args.epochs != 2 else train_cfg["epochs"]
        data_path = train_cfg["data"]

        model = GPT2(model_cfg).to(device)
        optimizer = torch.optim.AdamW(model.parameters(),
                                      lr=train_cfg["lr"],
                                      weight_decay=0.1)
        ckpt_meta = load_checkpoint(args.model_name, model, optimizer)
        model.to(device)
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

        model_cfg = build_config(args.model_size)
        train_cfg = {
            "data": args.data,
            "base_model_size": args.model_size,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "save_every": args.save_every,
            "eval_freq": args.eval_freq,
            "eval_iter": args.eval_iter,
        }
        total_epochs = args.epochs
        data_path = args.data

        full_cfg = {"model": model_cfg, "training": train_cfg}
        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")

        # Download / load pretrained OpenAI weights
        write_status(f"WEIGHTS downloading/loading pretrained {args.model_size}...")
        settings, params = download_and_load_pretrained(args.model_size)
        model = GPT2(model_cfg)
        load_weights_into_gpt(model, params)
        model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                      weight_decay=0.1)
        start_epoch, start_step, start_tokens = 0, -1, 0
        prev_tl, prev_vl = [], []

    # ---- Load data ----
    with open(data_path, "r") as f:
        data = json.load(f)
    write_status(f"DATA loaded {len(data)} instruction entries from {data_path}")

    batch_size = train_cfg["batch_size"] if args.resume else args.batch_size
    n_train = int(len(data) * 0.85)
    n_test  = int(len(data) * 0.10)
    train_data = data[:n_train]
    test_data  = data[n_train:n_train + n_test]
    val_data   = data[n_train + n_test:]

    collate = partial(instruction_collate_fn, device=device, allowed_max_length=1024)
    train_loader = DataLoader(InstructionDataset(train_data, tokenizer),
                              batch_size=batch_size, collate_fn=collate,
                              shuffle=True, drop_last=True)
    val_loader = DataLoader(InstructionDataset(val_data, tokenizer),
                            batch_size=batch_size, collate_fn=collate,
                            shuffle=False, drop_last=False)
    test_loader = DataLoader(InstructionDataset(test_data, tokenizer),
                             batch_size=batch_size, collate_fn=collate,
                             shuffle=False, drop_last=False)
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} "
                 f"test={len(test_loader)} batches")

    n_params = sum(p.numel() for p in model.parameters())
    write_status(f"MODEL params={n_params:,}")

    start_ctx = format_input_alpaca(data[0]) if data else "Hello"

    # ---- Train ----
    t0 = time.time()
    save_every = train_cfg.get("save_every", args.save_every) if args.resume else args.save_every
    eval_freq = train_cfg.get("eval_freq", args.eval_freq) if args.resume else args.eval_freq
    eval_iter = train_cfg.get("eval_iter", args.eval_iter) if args.resume else args.eval_iter

    train_sft(model, train_loader, val_loader, optimizer, device,
              num_epochs=total_epochs,
              eval_freq=eval_freq,
              eval_iter=eval_iter,
              start_context=start_ctx,
              tokenizer=tokenizer,
              model_name=args.model_name,
              save_every_n_epochs=save_every,
              start_epoch=start_epoch,
              start_global_step=start_step,
              start_tokens_seen=start_tokens,
              prev_train_losses=prev_tl,
              prev_val_losses=prev_vl)

    # Final test-set loss
    model.eval()
    with torch.no_grad():
        test_loss = calc_loss_loader(test_loader, model, device)
    write_status(f"TEST loss={test_loss:.4f}")

    elapsed = (time.time() - t0) / 60
    write_status(f"DONE training completed in {elapsed:.2f} min")


if __name__ == "__main__":
    main()
