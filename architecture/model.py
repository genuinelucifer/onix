"""
CausalLM — the main model class.
Assembles layers from config into a complete decoder-only transformer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.utils.checkpoint

from .layers import GroupedQueryAttention, FeedForward, build_norm

if TYPE_CHECKING:
    from .config import ModelConfig


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Supports two layouts:
      sequential: x → Norm → Attn → +residual → Norm → FFN → +residual   (LLaMA/GPT-2)
      parallel:   x → Norm → (Attn + FFN) → +residual                    (GPT-J/Phi)
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.block_type = config.block_type
        self.attn_norm = build_norm(config)
        self.attn = GroupedQueryAttention(config)
        self.ffn_norm = build_norm(config)
        self.ffn = FeedForward(config)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.block_type == "parallel":
            # GPT-J style: attention and FFN computed in parallel
            normed = self.attn_norm(x)
            attn_out = self.attn(normed)
            ffn_out = self.ffn(self.ffn_norm(x))
            x = x + self.resid_dropout(attn_out + ffn_out)
        else:
            # Standard sequential: Attn then FFN
            x = x + self.resid_dropout(self.attn(self.attn_norm(x)))
            x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))
        return x


class CausalLM(nn.Module):
    """
    Complete causal language model.

    Built entirely from a ModelConfig — supports GPT-2, LLaMA, Mistral,
    GPT-J, and custom architectures.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config

        # Token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.emb_dim)

        # Learned position embedding (only for pos_encoding == "learned")
        if config.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(config.context_length, config.emb_dim)
        else:
            self.pos_emb = None

        self.emb_dropout = nn.Dropout(config.emb_dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.n_layers)]
        )

        # Final norm
        self.final_norm = build_norm(config)

        # Output head
        self.lm_head = nn.Linear(config.emb_dim, config.vocab_size, bias=False)

        # Tie weights
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            idx: (batch, seq_len) token indices
        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, T = idx.shape

        # Token embeddings
        x = self.tok_emb(idx)

        # Add learned position embeddings if configured
        if self.pos_emb is not None:
            positions = torch.arange(T, device=idx.device)
            x = x + self.pos_emb(positions)

        x = self.emb_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            if self.config.grad_checkpointing and self.training:
                # use_reentrant=False is generally preferred in newer PyTorch
                x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)
        return logits

    def param_count(self) -> int:
        """Actual trainable parameter count."""
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        """Human-readable model summary."""
        n = self.param_count()
        size = f"{n / 1e9:.2f}B" if n > 1e9 else f"{n / 1e6:.1f}M"
        return (
            f"{self.config.summary()}\n"
            f"  Actual params: {n:,} (~{size})"
        )

    # ------ Backward compat: expose cfg dict like old GPT2 ------

    @property
    def cfg(self):
        """Compatibility shim: returns a dict similar to old GPT2.cfg."""
        return self.config.to_dict()
