"""
Layer implementations for configurable transformer architectures.

Includes:
  - Normalization:  RMSNorm, LayerNorm
  - Position:       RotaryEmbedding (RoPE), ALiBiPositionBias
  - Attention:      GroupedQueryAttention (handles MHA/MQA/GQA)
  - FFN:            FeedForward (GELU / SwiGLU / GeGLU / ReGLU)
"""

from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .config import ModelConfig


# ===========================================================================
#  Normalization
# ===========================================================================

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA, Mistral)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        x_norm = x.float() / rms
        return (self.weight * x_norm).type_as(x)


class LayerNorm(nn.Module):
    """Standard Layer Normalization (GPT-2)."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))
        self.shift = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        return self.scale * ((x - mean) / torch.sqrt(var + self.eps)) + self.shift


def build_norm(config: ModelConfig) -> nn.Module:
    """Factory: create the norm layer specified by config."""
    if config.norm_type == "rmsnorm":
        return RMSNorm(config.emb_dim, config.norm_eps)
    return LayerNorm(config.emb_dim, config.norm_eps)


# ===========================================================================
#  Rotary Position Embeddings (RoPE)
# ===========================================================================

class RotaryEmbedding(nn.Module):
    """Rotary Position Embeddings (Su et al., 2021). Used by LLaMA, Mistral, Qwen."""

    def __init__(self, head_dim: int, max_seq_len: int = 8192, base: float = 10000.0):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)  # (seq_len, head_dim)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def forward(self, seq_len: int):
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)
            self.max_seq_len = seq_len
        return self.cos_cached[:seq_len], self.sin_cached[:seq_len]


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate the second half of the last dimension."""
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor, k: torch.Tensor,
    cos: torch.Tensor, sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply RoPE to query and key tensors.
    q, k: (batch, heads, seq, head_dim)
    cos, sin: (seq, head_dim)
    """
    cos = cos.unsqueeze(0).unsqueeze(0)  # (1, 1, seq, head_dim)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


# ===========================================================================
#  ALiBi Position Bias
# ===========================================================================

class ALiBiPositionBias(nn.Module):
    """Attention with Linear Biases (Press et al., 2021). Used by BLOOM."""

    def __init__(self, n_heads: int, max_seq_len: int = 8192):
        super().__init__()
        slopes = self._get_slopes(n_heads)
        # Build causal distance matrix: bias[h, i, j] = slope_h * -(i - j) for j <= i
        positions = torch.arange(max_seq_len)
        # distance[i, j] = j - i  (negative for causal positions)
        distance = positions.unsqueeze(1) - positions.unsqueeze(0)  # (seq, seq)
        # slopes: (n_heads,) → (n_heads, 1, 1)
        bias = slopes.unsqueeze(1).unsqueeze(1) * distance.unsqueeze(0).float()
        self.register_buffer("bias", bias, persistent=False)

    @staticmethod
    def _get_slopes(n_heads: int) -> torch.Tensor:
        def _slopes_power_of_2(n):
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            return [start * (start ** i) for i in range(n)]

        if math.log2(n_heads).is_integer():
            return torch.tensor(_slopes_power_of_2(n_heads), dtype=torch.float32)
        else:
            closest = 2 ** math.floor(math.log2(n_heads))
            base_slopes = _slopes_power_of_2(closest)
            extra_slopes = _slopes_power_of_2(2 * closest)
            extra_needed = [extra_slopes[i] for i in range(0, 2 * closest, 2)]
            return torch.tensor(
                base_slopes + extra_needed[:n_heads - closest], dtype=torch.float32
            )

    def forward(self, seq_len: int) -> torch.Tensor:
        """Returns bias of shape (n_heads, seq_len, seq_len)."""
        return self.bias[:, :seq_len, :seq_len]


# ===========================================================================
#  Grouped-Query Attention (MHA / MQA / GQA)
# ===========================================================================

class GroupedQueryAttention(nn.Module):
    """
    Unified attention layer supporting:
      - MHA  (n_kv_heads == n_heads)
      - MQA  (n_kv_heads == 1)
      - GQA  (1 < n_kv_heads < n_heads)

    Uses F.scaled_dot_product_attention (Flash Attention) when possible.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.n_rep = self.n_heads // self.n_kv_heads
        self.scale = self.head_dim ** -0.5
        self.attn_dropout_p = config.attn_dropout
        self.sliding_window = config.sliding_window
        self.pos_encoding_type = config.pos_encoding

        # Projections
        self.q_proj = nn.Linear(config.emb_dim, config.n_heads * self.head_dim,
                                bias=config.attn_bias)
        self.k_proj = nn.Linear(config.emb_dim, config.n_kv_heads * self.head_dim,
                                bias=config.attn_bias)
        self.v_proj = nn.Linear(config.emb_dim, config.n_kv_heads * self.head_dim,
                                bias=config.attn_bias)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.emb_dim,
                                bias=config.attn_bias)

        self.attn_dropout = nn.Dropout(config.attn_dropout)

        # Position encoding modules
        if config.pos_encoding == "rope":
            self.rotary = RotaryEmbedding(
                self.head_dim, config.context_length, config.rope_base
            )
        elif config.pos_encoding == "alibi":
            self.alibi = ALiBiPositionBias(self.n_heads, config.context_length)

        # Causal mask (for manual attention path)
        causal = torch.triu(
            torch.ones(config.context_length, config.context_length, dtype=torch.bool),
            diagonal=1,
        )
        self.register_buffer("causal_mask", causal, persistent=False)

        # Determine if we can use the fast SDPA path
        # SDPA doesn't natively support ALiBi bias or sliding window with custom masks easily,
        # so we fall back to manual attention for those.
        self._use_sdpa = (
            config.use_sdpa
            and hasattr(F, "scaled_dot_product_attention")
            and config.pos_encoding != "alibi"
            and config.sliding_window is None
        )

    def _repeat_kv(self, x: torch.Tensor) -> torch.Tensor:
        """Repeat KV heads to match query head count for GQA."""
        if self.n_rep == 1:
            return x
        B, H, T, D = x.shape
        return (
            x.unsqueeze(2)
            .expand(B, H, self.n_rep, T, D)
            .reshape(B, self.n_heads, T, D)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        # Project
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        if self.pos_encoding_type == "rope":
            cos, sin = self.rotary(T)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV for GQA
        k = self._repeat_kv(k)
        v = self._repeat_kv(v)

        # Compute attention
        if self._use_sdpa:
            dropout_p = self.attn_dropout_p if self.training else 0.0
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p,
            )
        else:
            # Manual attention path (needed for ALiBi, sliding window)
            attn = (q @ k.transpose(-2, -1)) * self.scale

            # ALiBi bias
            if self.pos_encoding_type == "alibi":
                attn = attn + self.alibi(T)

            # Causal mask
            attn = attn.masked_fill(self.causal_mask[:T, :T], float("-inf"))

            # Sliding window mask
            if self.sliding_window is not None:
                positions = torch.arange(T, device=x.device)
                # distance[i,j] = i - j;  mask where distance >= window
                dist = positions.unsqueeze(1) - positions.unsqueeze(0)
                window_mask = dist >= self.sliding_window
                attn = attn.masked_fill(window_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            out = attn @ v

        # Output projection
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.o_proj(out)


# ===========================================================================
#  Feed-Forward Network variants
# ===========================================================================

class FeedForward(nn.Module):
    """
    Configurable FFN supporting:
      - gelu:   W2(GELU(W1(x)))
      - swiglu: W_down(SiLU(W_gate(x)) * W_up(x))
      - geglu:  W_down(GELU(W_gate(x)) * W_up(x))
      - reglu:  W_down(ReLU(W_gate(x)) * W_up(x))
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ffn_type = config.ffn_type
        dim = config.emb_dim
        inter = config.intermediate_size

        if config.ffn_type == "gelu":
            self.fc1 = nn.Linear(dim, inter, bias=config.ffn_bias)
            self.fc2 = nn.Linear(inter, dim, bias=config.ffn_bias)
            self.act = nn.GELU()
        else:
            # GLU variants: gate + up + down
            self.gate_proj = nn.Linear(dim, inter, bias=config.ffn_bias)
            self.up_proj = nn.Linear(dim, inter, bias=config.ffn_bias)
            self.down_proj = nn.Linear(inter, dim, bias=config.ffn_bias)
            if config.ffn_type == "swiglu":
                self.act = nn.SiLU()
            elif config.ffn_type == "geglu":
                self.act = nn.GELU()
            elif config.ffn_type == "reglu":
                self.act = nn.ReLU()

        self.dropout = nn.Dropout(config.ffn_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.ffn_type == "gelu":
            return self.dropout(self.fc2(self.act(self.fc1(x))))
        else:
            return self.dropout(
                self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
            )
