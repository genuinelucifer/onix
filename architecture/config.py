"""
Model Configuration System
Fully describes a decoder-only transformer architecture via a single dataclass.
Supports presets for popular architectures and custom JSON configs.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, Any


@dataclass
class ModelConfig:
    """
    Complete model architecture specification.

    Every architectural choice is explicit:
      - Attention: MHA / MQA / GQA  (via n_kv_heads)
      - Position:  learned / RoPE / ALiBi / none
      - Norm:      LayerNorm / RMSNorm  (always pre-norm)
      - FFN:       GELU / SwiGLU / GeGLU / ReGLU
      - Block:     sequential / parallel (GPT-J style)
      - Embeddings: tied / untied, dropout
    """

    # --- Identity ---
    name: str = "custom"

    # --- Core dimensions ---
    vocab_size: int = 50257
    context_length: int = 2048
    emb_dim: int = 2048
    n_layers: int = 24
    n_heads: int = 32

    # --- Attention ---
    n_kv_heads: Optional[int] = None   # None → n_heads (MHA), 1 → MQA, else GQA
    attn_bias: bool = False
    attn_dropout: float = 0.0
    sliding_window: Optional[int] = None  # None → full attention

    # --- Position encoding ---
    pos_encoding: str = "rope"          # "learned" | "rope" | "alibi" | "none"
    rope_base: float = 10000.0

    # --- Normalization (always pre-norm) ---
    norm_type: str = "rmsnorm"          # "layernorm" | "rmsnorm"
    norm_eps: float = 1e-5

    # --- Feed-forward ---
    ffn_type: str = "swiglu"            # "gelu" | "swiglu" | "geglu" | "reglu"
    intermediate_size: Optional[int] = None  # auto if None
    ffn_bias: bool = False
    ffn_dropout: float = 0.0

    # --- Block structure ---
    block_type: str = "sequential"      # "sequential" | "parallel"

    # --- Embeddings ---
    tie_embeddings: bool = True
    emb_dropout: float = 0.0

    # --- Residual ---
    residual_dropout: float = 0.0

    def __post_init__(self):
        # Default n_kv_heads to n_heads (standard MHA)
        if self.n_kv_heads is None:
            self.n_kv_heads = self.n_heads

        # Auto-compute intermediate_size
        if self.intermediate_size is None:
            if self.ffn_type in ("swiglu", "geglu", "reglu"):
                # GLU variants: 8/3 × emb_dim, rounded to nearest multiple of 256
                raw = int(8 / 3 * self.emb_dim)
                self.intermediate_size = ((raw + 255) // 256) * 256
            else:
                self.intermediate_size = 4 * self.emb_dim

        self._validate()

    def _validate(self):
        assert self.emb_dim % self.n_heads == 0, \
            f"emb_dim ({self.emb_dim}) must be divisible by n_heads ({self.n_heads})"
        assert self.n_heads % self.n_kv_heads == 0, \
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        assert self.pos_encoding in ("learned", "rope", "alibi", "none"), \
            f"Unknown pos_encoding: {self.pos_encoding}"
        assert self.norm_type in ("layernorm", "rmsnorm"), \
            f"Unknown norm_type: {self.norm_type}"
        assert self.ffn_type in ("gelu", "swiglu", "geglu", "reglu"), \
            f"Unknown ffn_type: {self.ffn_type}"
        assert self.block_type in ("sequential", "parallel"), \
            f"Unknown block_type: {self.block_type}"

    # --- Derived properties ---

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.n_heads

    @property
    def kv_dim(self) -> int:
        return self.n_kv_heads * self.head_dim

    @property
    def attention_type(self) -> str:
        if self.n_kv_heads == self.n_heads:
            return "MHA"
        elif self.n_kv_heads == 1:
            return "MQA"
        return "GQA"

    # --- Serialization ---

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> ModelConfig:
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> ModelConfig:
        with open(path) as f:
            return cls.from_dict(json.load(f))

    # --- Param estimation ---

    def param_count_estimate(self) -> int:
        """Rough parameter count (excludes biases for simplicity)."""
        # Embeddings
        emb = self.vocab_size * self.emb_dim
        pos = self.context_length * self.emb_dim if self.pos_encoding == "learned" else 0

        # Per-layer
        attn = self.emb_dim * (self.emb_dim + 2 * self.kv_dim + self.emb_dim)
        if self.ffn_type in ("swiglu", "geglu", "reglu"):
            ffn = 3 * self.emb_dim * self.intermediate_size
        else:
            ffn = 2 * self.emb_dim * self.intermediate_size
        norms = 2 * self.emb_dim
        layer_total = attn + ffn + norms

        # Output head
        out = 0 if self.tie_embeddings else self.vocab_size * self.emb_dim

        return emb + pos + self.n_layers * layer_total + self.emb_dim + out

    def summary(self) -> str:
        est = self.param_count_estimate()
        size = f"{est / 1e9:.2f}B" if est > 1e9 else f"{est / 1e6:.1f}M"
        lines = [
            f"Model: {self.name} (~{size} params)",
            f"  Dims: emb={self.emb_dim}, layers={self.n_layers}, "
            f"heads={self.n_heads}, head_dim={self.head_dim}",
            f"  Attention: {self.attention_type} (kv_heads={self.n_kv_heads})"
            + (f", sliding_window={self.sliding_window}" if self.sliding_window else ""),
            f"  Position: {self.pos_encoding}"
            + (f" (base={self.rope_base})" if self.pos_encoding == "rope" else ""),
            f"  Norm: {self.norm_type} (eps={self.norm_eps})",
            f"  FFN: {self.ffn_type} (intermediate={self.intermediate_size})",
            f"  Block: {self.block_type}",
            f"  Embeddings: tied={self.tie_embeddings}, dropout={self.emb_dropout}",
            f"  Context: {self.context_length}, Vocab: {self.vocab_size}",
        ]
        return "\n".join(lines)


# ===========================================================================
#  Presets — ready-to-use configs for popular architectures
# ===========================================================================

PRESETS: Dict[str, ModelConfig] = {

    # --- GPT-2 family (original architecture) ---
    "gpt2-124m": ModelConfig(
        name="gpt2-124m",
        vocab_size=50257, context_length=1024, emb_dim=768,
        n_layers=12, n_heads=12, n_kv_heads=12,
        attn_bias=True, pos_encoding="learned",
        norm_type="layernorm", ffn_type="gelu",
        ffn_bias=True, tie_embeddings=True,
        emb_dropout=0.1, attn_dropout=0.1, ffn_dropout=0.1, residual_dropout=0.1,
    ),
    "gpt2-355m": ModelConfig(
        name="gpt2-355m",
        vocab_size=50257, context_length=1024, emb_dim=1024,
        n_layers=24, n_heads=16, n_kv_heads=16,
        attn_bias=True, pos_encoding="learned",
        norm_type="layernorm", ffn_type="gelu",
        ffn_bias=True, tie_embeddings=True,
        emb_dropout=0.1, attn_dropout=0.1, ffn_dropout=0.1, residual_dropout=0.1,
    ),
    "gpt2-774m": ModelConfig(
        name="gpt2-774m",
        vocab_size=50257, context_length=1024, emb_dim=1280,
        n_layers=36, n_heads=20, n_kv_heads=20,
        attn_bias=True, pos_encoding="learned",
        norm_type="layernorm", ffn_type="gelu",
        ffn_bias=True, tie_embeddings=True,
        emb_dropout=0.1, attn_dropout=0.1, ffn_dropout=0.1, residual_dropout=0.1,
    ),

    # --- LLaMA-style (modern best practices) ---
    "llama-1b": ModelConfig(
        name="llama-1b",
        vocab_size=50257, context_length=2048, emb_dim=2048,
        n_layers=22, n_heads=32, n_kv_heads=4,
        pos_encoding="rope", rope_base=10000.0,
        norm_type="rmsnorm", ffn_type="swiglu",
        intermediate_size=5632,
        tie_embeddings=True,
    ),
    "llama-3b": ModelConfig(
        name="llama-3b",
        vocab_size=50257, context_length=2048, emb_dim=3200,
        n_layers=28, n_heads=32, n_kv_heads=8,
        pos_encoding="rope", rope_base=10000.0,
        norm_type="rmsnorm", ffn_type="swiglu",
        intermediate_size=8640,
        tie_embeddings=False,
    ),

    # --- Mistral-style (sliding window attention) ---
    "mistral-1b": ModelConfig(
        name="mistral-1b",
        vocab_size=50257, context_length=4096, emb_dim=2048,
        n_layers=22, n_heads=32, n_kv_heads=8,
        pos_encoding="rope", rope_base=10000.0,
        norm_type="rmsnorm", ffn_type="swiglu",
        intermediate_size=5632, sliding_window=4096,
        tie_embeddings=True,
    ),

    # --- GPT-J style (parallel blocks) ---
    "gptj-1b": ModelConfig(
        name="gptj-1b",
        vocab_size=50257, context_length=2048, emb_dim=2048,
        n_layers=22, n_heads=16, n_kv_heads=16,
        pos_encoding="rope", rope_base=10000.0,
        norm_type="layernorm", ffn_type="gelu",
        block_type="parallel",
        tie_embeddings=True,
    ),
}


def get_preset(name: str) -> ModelConfig:
    """Get a preset config by name (case-insensitive)."""
    key = name.lower()
    if key not in PRESETS:
        available = ", ".join(sorted(PRESETS.keys()))
        raise ValueError(f"Unknown preset '{name}'. Available: {available}")
    # Return a fresh copy
    return ModelConfig.from_dict(PRESETS[key].to_dict())
