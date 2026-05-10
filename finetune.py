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
import gc
from functools import partial
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np
import warnings

# Suppress the warning about non-writable numpy arrays from mmap
warnings.filterwarnings("ignore", message="The given NumPy array is not writable")

from architecture import ModelConfig, CausalLM, PRESETS, get_preset
from architecture.generate import generate

from model import (
    EOT_TOKEN_ID, get_tokenizer, text_to_token_ids,
    token_ids_to_text, write_status, save_model_config, load_model_config,
    save_checkpoint, load_checkpoint,
)

from training_utils import (
    create_optimizer, setup_status_file, setup_device, setup_performance,
    migrate_optimizer_to_device, get_train_params, EarlyStopper,
    add_common_training_args, get_default_training_config, has_checkpoint
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
    def __init__(self, all_tokens, bounds):
        self.all_tokens = all_tokens
        self.bounds = bounds

    def __getitem__(self, index):
        # bounds is now a numpy array or torch tensor of shape [N, 2]
        start, end = self.bounds[index]
        return self.all_tokens[start:end]

    def __len__(self):
        return len(self.bounds)


def instruction_collate_fn(batch, pad_token_id=EOT_TOKEN_ID,
                           ignore_index=-100, allowed_max_length=None,
                           device="cpu"):
    batch_max_length = max(len(item) + 1 for item in batch)
    inputs_lst, targets_lst = [], []
    for item in batch:
        if isinstance(item, torch.Tensor):
            item = item.tolist()
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
    # Don't move to device here if we use num_workers > 0 and pin_memory
    # DataLoader handles this if num_workers > 0. CPU tensors are better for pin_memory.
    return torch.stack(inputs_lst), torch.stack(targets_lst)

# ---------------------------------------------------------------------------
#  Loss helpers
# ---------------------------------------------------------------------------

def calc_loss_batch(inp, tgt, model, device):
    inp, tgt = inp.to(device), tgt.to(device)
    logits = model(inp)
    return nn.functional.cross_entropy(logits.flatten(0, 1), tgt.flatten())


def calc_loss_loader(loader, model, device, num_batches=None, use_bf16=False):
    if len(loader) == 0:
        return float("nan")
    n = min(num_batches, len(loader)) if num_batches else len(loader)
    total = 0.0
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
        loader_iter = iter(loader)
        for _ in range(n):
            try:
                inp, tgt = next(loader_iter)
            except StopIteration:
                break
            total += calc_loss_batch(inp, tgt, model, device).item()
    return total / n


def evaluate(model, train_loader, val_loader, device, eval_iter, use_bf16=False):
    model.eval()
    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, eval_iter, use_bf16)
        vl = calc_loss_loader(val_loader, model, device, eval_iter, use_bf16)
    model.train()
    return tl, vl

# ---------------------------------------------------------------------------
#  Training loop
# ---------------------------------------------------------------------------

def train_sft(model, train_loader, val_loader, optimizer, device,
              num_epochs, log_freq, eval_freq, eval_iter, start_context, tokenizer,
              model_name, save_every_n_epochs, save_every_n_iters=None,
              start_epoch=0, start_global_step=-1, start_tokens_seen=0,
              prev_train_losses=None, prev_val_losses=None,
              early_stopper=None, use_bf16=False):
    train_losses = list(prev_train_losses or [])
    val_losses = list(prev_val_losses or [])
    tokens_seen = start_tokens_seen
    global_step = start_global_step

    completed_epochs = start_epoch
    ctx_size = model.config.context_length

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for i, (inp, tgt) in enumerate(train_loader):
            current_abs_step = epoch * len(train_loader) + i
            if current_abs_step <= start_global_step:
                continue

            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
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
                tl, vl = evaluate(model, train_loader, val_loader, device, eval_iter, use_bf16)
                train_losses.append(tl)
                val_losses.append(vl)
                write_status(
                    f"EVAL epoch={epoch+1}/{num_epochs} step={global_step:06d} "
                    f"tokens={tokens_seen} train_loss={tl:.4f} val_loss={vl:.4f}"
                )

                if early_stopper is not None:
                    if early_stopper.check(vl, global_step, completed_epochs):
                        if device.type == "cuda":
                            torch.cuda.synchronize()
                        
                        write_status(
                            f"EARLY_STOP triggered at step {global_step} "
                            f"({early_stopper.status_message()})"
                        )
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
                    ckpt_path = save_checkpoint(
                        model_name, model, optimizer, epoch, global_step,
                        tokens_seen, train_losses, val_losses, tag=f"step{global_step}")
                    write_status(f"CHECKPOINT saved at step {global_step} -> {ckpt_path}")

        completed_epochs = epoch + 1

        # Generate a sample after each epoch
        model.eval()
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
    )
    add_common_training_args(parser)
    parser.add_argument("--data", type=str, default=None,
                        help="Path to instruction JSON (required for new run)")
    parser.add_argument("--preset", default="gpt2-124m",
                        choices=list(PRESETS.keys()),
                        help="Architecture preset to use for fresh SFT")
    parser.add_argument("--base-model", type=str, default=None,
                        help="Base model to start SFT from (loads weights, ignores optimizer)")
    args = parser.parse_args()

    # ---- Setup ----
    setup_performance()
    device = setup_device()
    setup_status_file(args.model_name, resume=args.resume)
    write_status(f"START device={device} model_name={args.model_name} resume={args.resume}")

    tokenizer = get_tokenizer()

    if args.resume:
        checkpoint_exists = has_checkpoint(args.model_name)
        if checkpoint_exists:
            write_status("RESUME loading config and checkpoint...")
            full_cfg = load_model_config(args.model_name)
        else:
            raise ValueError("--resume specified but no checkpoint found.")

        train_cfg = full_cfg["training"]
        model_config = ModelConfig.from_dict(full_cfg["architecture"])
        model = CausalLM(model_config).to(device)

        tp = get_train_params("sft", train_cfg, args, has_checkpoint=checkpoint_exists)
        optimizer = create_optimizer(model, tp["optimizer"], lr=tp["lr"])
        
        ckpt_meta = load_checkpoint(args.model_name, model, optimizer)
        model.to(device)
        migrate_optimizer_to_device(optimizer, device)

        start_epoch = ckpt_meta["epoch"]
        start_step = ckpt_meta["global_step"]
        start_tokens = ckpt_meta["tokens_seen"]
        prev_tl = ckpt_meta["train_losses"]
        prev_vl = ckpt_meta["val_losses"]

        if start_epoch >= tp["epochs"]:
            write_status(f"Already trained {start_epoch} epochs (target={tp['epochs']}). "
                         f"Increase --epochs to continue.")
            return

        write_status(f"RESUMED from epoch={start_epoch} step={start_step} "
                     f"tokens={start_tokens} -> training to epoch {tp['epochs']}")
        data_path = train_cfg["data"]
    elif args.base_model:
        if args.data is None:
            parser.error("--data is required for a new training run")

        write_status(f"BASE_MODEL loading config and weights from {args.base_model}...")
        base_cfg = load_model_config(args.base_model)
        model_config = ModelConfig.from_dict(base_cfg["architecture"])
        
        train_cfg = get_default_training_config("sft", args)
        train_cfg["data"] = args.data
        train_cfg["base_model"] = args.base_model

        full_cfg = {"architecture": model_config.to_dict(), "training": train_cfg}
        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")

        model = CausalLM(model_config).to(device)
        
        tp = get_train_params("sft", train_cfg, args, has_checkpoint=False)
        optimizer = create_optimizer(model, tp["optimizer"], lr=tp["lr"])
        
        load_checkpoint(args.base_model, model)
        
        start_epoch, start_step, start_tokens = 0, -1, 0
        prev_tl, prev_vl = [], []
        data_path = args.data
    else:
        if args.data is None:
            parser.error("--data is required for a new training run")

        model_config = get_preset(args.preset)
        train_cfg = get_default_training_config("sft", args)
        train_cfg["data"] = args.data
        train_cfg["preset"] = args.preset

        full_cfg = {"architecture": model_config.to_dict(), "training": train_cfg}
        save_model_config(args.model_name, full_cfg)
        write_status(f"CONFIG saved to models/{args.model_name}/config.json")

        model = CausalLM(model_config).to(device)

        tp = get_train_params("sft", train_cfg, args, has_checkpoint=False)
        optimizer = create_optimizer(model, tp["optimizer"], lr=tp["lr"])
        
        start_epoch, start_step, start_tokens = 0, -1, 0
        prev_tl, prev_vl = [], []
        data_path = args.data

    # ---- Load data ----
    # We check if pre-tokenized binary shards exist to avoid expensive in-memory tokenization.
    tokens_path = Path(f"{data_path}_tokens.npy")
    bounds_path = Path(f"{data_path}_bounds.npy")
    
    if tokens_path.exists() and bounds_path.exists():
        write_status(f"DATA: Loading pre-tokenized binary shards from {data_path} (mmap)")
        # Memory-map the arrays to keep CPU RAM usage near zero
        tokens_np = np.load(tokens_path, mmap_mode="r")
        bounds = np.load(bounds_path, mmap_mode="r")
        
        # Move tokens to VRAM. 
        # Even with 700M tokens, this is only 2.8GB (int32) or 5.6GB (int64).
        all_tokens_tensor = torch.from_numpy(tokens_np).to(torch.long).to(device)
        
        # For bounds, we keep a copy in CPU RAM for fast indexing in the Dataset
        bounds = bounds.copy()
        
        write_status(f"DATA loaded {len(bounds)} entries into {len(tokens_np)/1e6:.1f}M tokens")
    else:
        import array
        tokens_array = array.array('i')
        # Use a flat array for bounds to avoid creating millions of Python tuple objects
        bounds_array = array.array('i') 
        current_idx = 0
        count = 0
        
        write_status(f"DATA loading and tokenizing from {data_path} (Streaming)")
        with open(data_path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                full = format_input_alpaca(entry) + f"\n\n### Response:\n{entry['output']}"
                tokens = tokenizer.encode(full)
                tokens_array.extend(tokens)
                bounds_array.extend([current_idx, current_idx + len(tokens)])
                current_idx += len(tokens)
                count += 1
                if count % 500000 == 0:
                    write_status(f"DATA tokenized {count} entries...")
                    
        write_status(f"DATA loaded {count} entries into {len(tokens_array)/1e6:.1f}M tokens")
        
        # Zero-copy conversion to numpy, then move to GPU. 
        # This avoids the "Python object explosion" that occurs with torch.tensor(array_array).
        tokens_np = np.frombuffer(tokens_array, dtype=np.int32)
        all_tokens_tensor = torch.from_numpy(tokens_np).to(torch.long).to(device)
        
        # Free the temporary CPU arrays
        del tokens_array
        del tokens_np
        
        # Convert flat bounds to [N, 2] numpy array
        bounds = np.frombuffer(bounds_array, dtype=np.int32).reshape(-1, 2).copy()
        del bounds_array
    
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()

    n_train = int(len(bounds) * 0.85)
    n_test  = int(len(bounds) * 0.10)
    train_bounds = bounds[:n_train]
    test_bounds  = bounds[n_train:n_train + n_test]
    val_bounds   = bounds[n_train + n_test:]

    # DataLoaders: If data is on GPU, we should use num_workers=0 to avoid IPC overhead
    # and potential ROCm forking issues.
    use_gpu_data = all_tokens_tensor.is_cuda
    num_workers = 0 if use_gpu_data else tp["num_workers"]
    pin_memory = False if use_gpu_data else tp["pin_memory"]
    pf = tp["prefetch_factor"] if num_workers > 0 else None

    if use_gpu_data:
        write_status("DATA: Utilizing VRAM for dataset storage. DataLoader workers set to 0.")

    collate = partial(instruction_collate_fn, device=device if use_gpu_data else "cpu", 
                      allowed_max_length=1024)
    
    train_loader = DataLoader(InstructionDataset(all_tokens_tensor, train_bounds),
                              batch_size=tp["batch_size"], collate_fn=collate,
                              shuffle=True, drop_last=True,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              prefetch_factor=pf)
    val_loader = DataLoader(InstructionDataset(all_tokens_tensor, val_bounds),
                            batch_size=tp["batch_size"], collate_fn=collate,
                            shuffle=False, drop_last=False,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            prefetch_factor=pf)
    test_loader = DataLoader(InstructionDataset(all_tokens_tensor, test_bounds),
                             batch_size=tp["batch_size"], collate_fn=collate,
                             shuffle=False, drop_last=False,
                             num_workers=num_workers,
                             pin_memory=pin_memory,
                             prefetch_factor=pf)
                             
    write_status(f"LOADERS train={len(train_loader)} val={len(val_loader)} test={len(test_loader)} batches")

    n_params = sum(p.numel() for p in model.parameters())
    write_status(f"MODEL params={n_params:,}")

    # ---- Early stopper ----
    early_stopper = None
    if tp["patience"] > 0:
        early_stopper = EarlyStopper(
            patience_evals=tp["patience"],
            min_delta=tp["min_delta"],
            min_epochs=tp["min_epochs"],
            window_size=tp["window_size"],
        )
        write_status(f"EARLY_STOP enabled: patience={tp['patience']} evals")

    # ---- Gradient Checkpointing ----
    if tp["checkpointing"]:
        write_status("GRADIENT_CHECKPOINTING: Enabled")
        model.config.grad_checkpointing = True

    # ---- Compile ----
    if tp["compile"]:
        write_status("torch.compile: Compiling model... (This will take a few minutes)")
        model = torch.compile(model)

    start_ctx = "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n### Instruction:\nTell me a story."
    try:
        with open(data_path, "r") as f:
            for line in f:
                if line.strip():
                    start_ctx = format_input_alpaca(json.loads(line))
                    break
    except (FileNotFoundError, IsADirectoryError, UnicodeDecodeError, json.JSONDecodeError):
        # Fallback for binary data or missing JSONL
        pass

    # ---- Train ----
    t0 = time.time()

    train_sft(model, train_loader, val_loader, optimizer, device,
              num_epochs=tp["epochs"],
              log_freq=tp["log_freq"],
              eval_freq=tp["eval_freq"],
              eval_iter=tp["eval_iter"],
              start_context=start_ctx,
              tokenizer=tokenizer,
              model_name=args.model_name,
              save_every_n_epochs=tp["save_every"],
              save_every_n_iters=tp["save_iters"],
              start_epoch=start_epoch,
              start_global_step=start_step,
              start_tokens_seen=start_tokens,
              prev_train_losses=prev_tl,
              prev_val_losses=prev_vl,
              early_stopper=early_stopper,
              use_bf16=tp["bf16"])

    # Final test-set loss
    model.eval()
    with torch.no_grad():
        test_loss = calc_loss_loader(test_loader, model, device, use_bf16=tp["bf16"])
    write_status(f"TEST loss={test_loss:.4f}")

    elapsed = (time.time() - t0) / 60
    write_status(f"DONE training completed in {elapsed:.2f} min")

if __name__ == "__main__":
    main()
