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
    use_sdpa: bool = True               # Use PyTorch Scaled Dot Product Attention (Flash/Mem-Eff)

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
    grad_checkpointing: bool = False    # Use gradient checkpointing to save memory

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
#  VQ-VAE Configuration
# ===========================================================================

@dataclass
class VQVAEConfig:
    """
    Complete VQ-VAE architecture specification for image tokenization.

    Describes a convolutional encoder-decoder with a discrete codebook.
    The encoder downsamples an image into a spatial grid of latent vectors,
    which are quantized to the nearest codebook entry. The decoder
    reconstructs the image from quantized vectors.
    """

    # --- Identity ---
    name: str = "vqvae-default"

    # --- Image ---
    image_size: int = 256           # Input image resolution (must be square)
    image_channels: int = 3         # RGB=3, RGBA=4

    # --- Encoder / Decoder architecture ---
    base_channels: int = 128        # First conv layer output channels
    channel_multipliers: tuple = (1, 2, 4, 8)   # Channel scaling per downsampling stage
    num_res_blocks: int = 2         # Residual blocks per stage

    # --- Codebook ---
    codebook_size: int = 8192       # Number of discrete visual "words"
    codebook_dim: int = 256         # Dimension of each codebook vector
    commitment_weight: float = 0.25 # Weight for commitment loss
    ema_decay: float = 0.99         # EMA decay for codebook updates (0 = no EMA, use gradient)

    # --- Loss ---
    loss_type: str = "mse"          # "mse" | "perceptual" (future extension)

    # --- Memory ---
    grad_checkpointing: bool = False

    def __post_init__(self):
        # Convert list to tuple if loaded from JSON
        if isinstance(self.channel_multipliers, list):
            self.channel_multipliers = tuple(self.channel_multipliers)
        self._validate()

    def _validate(self):
        # Check image_size is divisible by the total downsampling factor
        n_downsamples = len(self.channel_multipliers) - 1
        downsample_factor = 2 ** n_downsamples
        assert self.image_size % downsample_factor == 0, (
            f"image_size ({self.image_size}) must be divisible by "
            f"downsample factor ({downsample_factor} = 2^{n_downsamples})"
        )
        assert self.loss_type in ("mse", "perceptual"), \
            f"Unknown loss_type: {self.loss_type}"

    @property
    def latent_grid_size(self) -> int:
        """Spatial resolution of the latent grid after encoding."""
        n_downsamples = len(self.channel_multipliers) - 1
        return self.image_size // (2 ** n_downsamples)

    @property
    def num_visual_tokens(self) -> int:
        """Total number of tokens per image (latent_grid_size^2)."""
        return self.latent_grid_size ** 2

    # --- Serialization ---

    def to_dict(self) -> dict:
        d = asdict(self)
        # tuple → list for JSON serialization
        d["channel_multipliers"] = list(d["channel_multipliers"])
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "VQVAEConfig":
        known = {k for k in cls.__dataclass_fields__}
        return cls(**{k: v for k, v in d.items() if k in known})

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "VQVAEConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def summary(self) -> str:
        gs = self.latent_grid_size
        lines = [
            f"VQ-VAE: {self.name}",
            f"  Image: {self.image_size}x{self.image_size}x{self.image_channels}",
            f"  Latent grid: {gs}x{gs} = {self.num_visual_tokens} tokens",
            f"  Codebook: {self.codebook_size} entries, dim={self.codebook_dim}",
            f"  Channels: base={self.base_channels}, mults={self.channel_multipliers}",
            f"  Res blocks per stage: {self.num_res_blocks}",
            f"  Loss: {self.loss_type}, commitment_weight={self.commitment_weight}",
            f"  EMA decay: {self.ema_decay}",
        ]
        return "\n".join(lines)


# ===========================================================================
#  Multi-Modal Configuration (Transformer + frozen VQ-VAE)
# ===========================================================================

@dataclass
class MultiModalConfig:
    """
    Wraps a transformer ModelConfig with VQ-VAE reference for
    text-to-image autoregressive training (Phase 2).

    The transformer operates on a joint token space:
      [0, text_vocab_size)              → text BPE tokens
      [text_vocab_size, text+visual)    → visual codebook tokens
      text+visual, text+visual+1        → <IMG_START>
      text+visual+1, text+visual+2      → <IMG_END>
    """

    # --- Sub-configs ---
    transformer: Optional[ModelConfig] = None
    vqvae: Optional[VQVAEConfig] = None

    # --- References ---
    vqvae_checkpoint: str = ""      # Path to frozen VQ-VAE checkpoint

    # --- Token space ---
    text_vocab_size: int = 50257    # BPE vocab size
    max_text_tokens: int = 256      # Max text prompt length (padded/truncated)

    # --- Loss ---
    loss_mask_text: bool = True     # Only compute loss on visual token positions

    def __post_init__(self):
        if self.transformer is None:
            self.transformer = ModelConfig()
        if self.vqvae is None:
            self.vqvae = VQVAEConfig()

    @property
    def visual_vocab_size(self) -> int:
        return self.vqvae.codebook_size

    @property
    def num_visual_tokens(self) -> int:
        return self.vqvae.num_visual_tokens

    @property
    def img_start_id(self) -> int:
        return self.text_vocab_size + self.visual_vocab_size

    @property
    def img_end_id(self) -> int:
        return self.text_vocab_size + self.visual_vocab_size + 1

    @property
    def total_vocab_size(self) -> int:
        """text + visual + <IMG_START> + <IMG_END>"""
        return self.text_vocab_size + self.visual_vocab_size + 2

    @property
    def max_seq_length(self) -> int:
        """Max sequence: text_tokens + <IMG_START> + visual_tokens + <IMG_END>"""
        return self.max_text_tokens + 1 + self.num_visual_tokens + 1

    def to_dict(self) -> dict:
        return {
            "transformer": self.transformer.to_dict(),
            "vqvae": self.vqvae.to_dict(),
            "vqvae_checkpoint": self.vqvae_checkpoint,
            "text_vocab_size": self.text_vocab_size,
            "max_text_tokens": self.max_text_tokens,
            "loss_mask_text": self.loss_mask_text,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "MultiModalConfig":
        transformer = ModelConfig.from_dict(d["transformer"])
        vqvae = VQVAEConfig.from_dict(d["vqvae"])
        return cls(
            transformer=transformer,
            vqvae=vqvae,
            vqvae_checkpoint=d.get("vqvae_checkpoint", ""),
            text_vocab_size=d.get("text_vocab_size", 50257),
            max_text_tokens=d.get("max_text_tokens", 256),
            loss_mask_text=d.get("loss_mask_text", True),
        )

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "MultiModalConfig":
        with open(path) as f:
            return cls.from_dict(json.load(f))

    def build_transformer_config(self) -> ModelConfig:
        """
        Return a ModelConfig with vocab_size and context_length
        automatically set for the multi-modal token space.
        """
        cfg = ModelConfig.from_dict(self.transformer.to_dict())
        cfg.vocab_size = self.total_vocab_size
        cfg.context_length = self.max_seq_length
        return cfg

    def summary(self) -> str:
        t_cfg = self.build_transformer_config()
        lines = [
            f"Multi-Modal Config:",
            f"  Text vocab: {self.text_vocab_size}, Visual vocab: {self.visual_vocab_size}",
            f"  Total vocab: {self.total_vocab_size}",
            f"  Max seq length: {self.max_seq_length} "
            f"(text={self.max_text_tokens} + img_start + visual={self.num_visual_tokens} + img_end)",
            f"  Special tokens: <IMG_START>={self.img_start_id}, <IMG_END>={self.img_end_id}",
            f"  Loss mask text: {self.loss_mask_text}",
            f"  VQ-VAE checkpoint: {self.vqvae_checkpoint}",
            f"",
            f"--- Transformer ---",
            t_cfg.summary(),
            f"",
            f"--- VQ-VAE ---",
            self.vqvae.summary(),
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
