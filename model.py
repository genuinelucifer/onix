#!/usr/bin/env python3
"""
YALLM Shared Model & Utilities
Contains the GPT-2 architecture, configs, tokenizer helpers, weight loading,
and generation logic. Used by both train.py and finetune.py.
"""

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import tiktoken
import torch
import torch.nn as nn

EOT_TOKEN = "<" + "|endoftext|" + ">"
EOT_TOKEN_ID = 50256

MODELS_DIR = Path(os.environ.get("YALLM_MODELS_DIR", Path(__file__).parent / "models"))


# ===========================================================================
#  Model Architecture
# ===========================================================================

class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * ((x - mean) / torch.sqrt(var + self.eps)) + self.shift


class MultiHeadAttention(nn.Module):
    def __init__(self, din, dout, ctxlen, dropout, num_heads, kv_bias=False):
        super().__init__()
        assert dout % num_heads == 0
        self.dout = dout
        self.num_heads = num_heads
        self.head_dim = dout // num_heads
        self.Wq = nn.Linear(din, dout, bias=kv_bias)
        self.Wk = nn.Linear(din, dout, bias=kv_bias)
        self.Wv = nn.Linear(din, dout, bias=kv_bias)
        self.out_proj = nn.Linear(dout, dout)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("mask", torch.triu(torch.ones(ctxlen, ctxlen), diagonal=1))

    def forward(self, x):
        b, ntok, _ = x.shape
        q = self.Wq(x).view(b, ntok, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.Wk(x).view(b, ntok, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.Wv(x).view(b, ntok, self.num_heads, self.head_dim).transpose(1, 2)
        att = q @ k.transpose(2, 3)
        att.masked_fill_(self.mask.bool()[:ntok, :ntok], -torch.inf)
        ws = torch.softmax(att / k.shape[-1] ** 0.5, dim=-1)
        ws = self.dropout(ws)
        ctx = (ws @ v).transpose(1, 2).contiguous().view(b, ntok, self.dout)
        return self.out_proj(ctx)


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x.pow(3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(cfg["emb_dim"] * 4, cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layer(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.mha = MultiHeadAttention(
            cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
            cfg["drop_rate"], cfg["n_heads"], cfg["qkv_bias"],
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_sc = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = self.drop_sc(self.mha(self.norm1(x))) + x
        x = self.drop_sc(self.ff(self.norm2(x))) + x
        return x


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.tk_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.tf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, x):
        _, seq_len = x.shape
        x = self.drop_emb(
            self.tk_emb(x) + self.pos_emb(torch.arange(seq_len, device=x.device))
        )
        x = self.final_norm(self.tf_blocks(x))
        return self.out(x)


# ===========================================================================
#  Model Configs (base presets matching OpenAI GPT-2 checkpoints)
# ===========================================================================

BASE_CONFIG = {
    "vocab_size": 50257, "context_length": 1024,
    "drop_rate": 0.0, "qkv_bias": True,
}
MODEL_CONFIGS = {
    "124M":  {"emb_dim": 768,  "n_layers": 12, "n_heads": 12},
    "355M":  {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "774M":  {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "1558M": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}


def build_config(model_size, context_length=None, drop_rate=None):
    cfg = {**BASE_CONFIG, **MODEL_CONFIGS[model_size]}
    if context_length is not None:
        cfg["context_length"] = context_length
    if drop_rate is not None:
        cfg["drop_rate"] = drop_rate
    return cfg


# ===========================================================================
#  Model directory helpers  (models/<model_name>/)
# ===========================================================================

def get_model_dir(model_name):
    """Return the Path to models/<model_name>, creating it if needed."""
    d = MODELS_DIR / model_name
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_model_config(model_name, config):
    """Write the full model config dict to models/<model_name>/config.json."""
    d = get_model_dir(model_name)
    with open(d / "config.json", "w") as f:
        json.dump(config, f, indent=2)


def load_model_config(model_name):
    """Load the config dict from models/<model_name>/config.json."""
    d = get_model_dir(model_name)
    cfg_path = d / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"No config.json in {d}")
    with open(cfg_path) as f:
        return json.load(f)


def save_checkpoint(model_name, model, optimizer, epoch, global_step,
                    tokens_seen, train_losses, val_losses, tag=None):
    """Save a training checkpoint to models/<model_name>/."""
    d = get_model_dir(model_name)
    fname = f"checkpoint_epoch{epoch}.pt" if tag is None else f"checkpoint_{tag}.pt"
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "tokens_seen": tokens_seen,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, d / fname)
    # Also save a "latest" symlink/copy for easy resume
    latest = d / "checkpoint_latest.pt"
    if latest.exists() or latest.is_symlink():
        latest.unlink()
    os.symlink(fname, latest)
    
    # Keep only the latest 2 epoch checkpoints
    if tag is None:
        checkpoints = sorted(
            [p for p in d.glob("checkpoint_epoch*.pt")],
            key=lambda p: os.path.getmtime(p)
        )
        while len(checkpoints) > 2:
            old_ckpt = checkpoints.pop(0)
            if old_ckpt.name != fname:
                try:
                    old_ckpt.unlink()
                except OSError:
                    pass

    return d / fname


def load_checkpoint(model_name, model, optimizer=None, tag="latest"):
    """Load a checkpoint from models/<model_name>/. Returns the metadata dict."""
    d = get_model_dir(model_name)
    ckpt_path = d / f"checkpoint_{tag}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint_{tag}.pt found in {d}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return {
        "epoch": ckpt["epoch"],
        "global_step": ckpt["global_step"],
        "tokens_seen": ckpt["tokens_seen"],
        "train_losses": ckpt.get("train_losses", []),
        "val_losses": ckpt.get("val_losses", []),
    }


def get_status_file(model_name):
    """Return the path to the status log for this model."""
    d = get_model_dir(model_name)
    return d / "status.txt"


# ===========================================================================
#  Status logging (writes to models/<model_name>/status.txt)
# ===========================================================================

_status_path = None   # set by scripts at startup


def set_status_file(path):
    global _status_path
    _status_path = Path(path)


def write_status(msg):
    """Append a timestamped message to the status file for external monitoring."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}\n"
    if _status_path is not None:
        with open(_status_path, "a") as f:
            f.write(line)
    print(msg)


# ===========================================================================
#  Tokenizer helpers
# ===========================================================================

def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={EOT_TOKEN})
    return torch.tensor(encoded).unsqueeze(0)


def token_ids_to_text(tokens, tokenizer):
    return tokenizer.decode(tokens.squeeze(0).tolist())


# ===========================================================================
#  Text generation
# ===========================================================================

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val,
                                 torch.tensor(float("-inf")).to(logits.device), logits)
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        if eos_id is not None and idx_next.item() == eos_id:
            break
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


# ===========================================================================
#  Weight loading from OpenAI TF checkpoints
# ===========================================================================

def _assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch: {left.shape} vs {right.shape}")
    return nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, params["wpe"])
    gpt.tk_emb.weight  = _assign(gpt.tk_emb.weight,  params["wte"])
    for b in range(len(params["blocks"])):
        blk = params["blocks"][b]
        q_w, k_w, v_w = np.split(blk["attn"]["c_attn"]["w"], 3, axis=-1)
        gpt.tf_blocks[b].mha.Wq.weight = _assign(gpt.tf_blocks[b].mha.Wq.weight, q_w.T)
        gpt.tf_blocks[b].mha.Wk.weight = _assign(gpt.tf_blocks[b].mha.Wk.weight, k_w.T)
        gpt.tf_blocks[b].mha.Wv.weight = _assign(gpt.tf_blocks[b].mha.Wv.weight, v_w.T)
        q_b, k_b, v_b = np.split(blk["attn"]["c_attn"]["b"], 3, axis=-1)
        gpt.tf_blocks[b].mha.Wq.bias = _assign(gpt.tf_blocks[b].mha.Wq.bias, q_b)
        gpt.tf_blocks[b].mha.Wk.bias = _assign(gpt.tf_blocks[b].mha.Wk.bias, k_b)
        gpt.tf_blocks[b].mha.Wv.bias = _assign(gpt.tf_blocks[b].mha.Wv.bias, v_b)
        gpt.tf_blocks[b].mha.out_proj.weight = _assign(gpt.tf_blocks[b].mha.out_proj.weight, blk["attn"]["c_proj"]["w"].T)
        gpt.tf_blocks[b].mha.out_proj.bias   = _assign(gpt.tf_blocks[b].mha.out_proj.bias,   blk["attn"]["c_proj"]["b"])
        gpt.tf_blocks[b].ff.layer[0].weight = _assign(gpt.tf_blocks[b].ff.layer[0].weight, blk["mlp"]["c_fc"]["w"].T)
        gpt.tf_blocks[b].ff.layer[0].bias   = _assign(gpt.tf_blocks[b].ff.layer[0].bias,   blk["mlp"]["c_fc"]["b"])
        gpt.tf_blocks[b].ff.layer[2].weight = _assign(gpt.tf_blocks[b].ff.layer[2].weight, blk["mlp"]["c_proj"]["w"].T)
        gpt.tf_blocks[b].ff.layer[2].bias   = _assign(gpt.tf_blocks[b].ff.layer[2].bias,   blk["mlp"]["c_proj"]["b"])
        gpt.tf_blocks[b].norm1.scale = _assign(gpt.tf_blocks[b].norm1.scale, blk["ln_1"]["g"])
        gpt.tf_blocks[b].norm1.shift = _assign(gpt.tf_blocks[b].norm1.shift, blk["ln_1"]["b"])
        gpt.tf_blocks[b].norm2.scale = _assign(gpt.tf_blocks[b].norm2.scale, blk["ln_2"]["g"])
        gpt.tf_blocks[b].norm2.shift = _assign(gpt.tf_blocks[b].norm2.shift, blk["ln_2"]["b"])
    gpt.final_norm.scale = _assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, params["b"])
    gpt.out.weight       = _assign(gpt.out.weight,       params["wte"])


def download_and_load_pretrained(model_size, models_dir="gpt2"):
    # gpt_download.py lives in the parent dir (yallm/)
    parent = str(Path(__file__).resolve().parent.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    from gpt_download import download_and_load_gpt2
    return download_and_load_gpt2(model_size=model_size, models_dir=models_dir)
