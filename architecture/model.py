"""
CausalLM — the main model class.
Assembles layers from config into a complete decoder-only transformer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple, Optional

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

    def forward(
        self,
        x: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        new_cache = None
        if self.block_type == "parallel":
            # GPT-J style: attention and FFN computed in parallel
            normed = self.attn_norm(x)
            if use_cache or kv_cache is not None:
                attn_out, new_cache = self.attn(
                    normed,
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    use_cache=use_cache,
                )
            else:
                attn_out, _ = self.attn(normed, position_ids=position_ids, use_cache=False)
            ffn_out = self.ffn(self.ffn_norm(x))
            x = x + self.resid_dropout(attn_out + ffn_out)
        else:
            # Standard sequential: Attn then FFN
            if use_cache or kv_cache is not None:
                attn_out, new_cache = self.attn(
                    self.attn_norm(x),
                    position_ids=position_ids,
                    kv_cache=kv_cache,
                    use_cache=use_cache,
                )
            else:
                attn_out, _ = self.attn(self.attn_norm(x), position_ids=position_ids, use_cache=False)
            x = x + self.resid_dropout(attn_out)
            x = x + self.resid_dropout(self.ffn(self.ffn_norm(x)))

        if use_cache:
            return x, new_cache
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

    def forward(
        self,
        idx: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        kv_caches: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            idx: (batch, seq_len) token indices
            position_ids: (batch, seq_len) position indices
            kv_caches: list of cached (key, value) tuples per block
            use_cache: whether to compute and return new KV caches
        Returns:
            logits if use_cache is False, else (logits, new_kv_caches)
        """
        B, T = idx.shape

        # Token embeddings
        x = self.tok_emb(idx)

        # Add learned position embeddings if configured
        if self.pos_emb is not None:
            if position_ids is not None:
                x = x + self.pos_emb(position_ids)
            else:
                positions = torch.arange(T, device=idx.device)
                x = x + self.pos_emb(positions)

        x = self.emb_dropout(x)

        # Transformer blocks
        new_kv_caches = [] if use_cache else None
        for i, block in enumerate(self.blocks):
            block_cache = kv_caches[i] if kv_caches is not None else None

            if use_cache or block_cache is not None:
                x, new_cache = block(
                    x,
                    position_ids=position_ids,
                    kv_cache=block_cache,
                    use_cache=use_cache,
                )
                if use_cache:
                    new_kv_caches.append(new_cache)
            else:
                if self.config.grad_checkpointing and self.training:
                    # use_reentrant=False is generally preferred in newer PyTorch
                    x = torch.utils.checkpoint.checkpoint(block, x, use_reentrant=False)
                else:
                    x = block(x)

        # Final norm + LM head
        x = self.final_norm(x)
        logits = self.lm_head(x)

        if use_cache:
            return logits, new_kv_caches
        return logits

    def setup_caches(self, max_batch_size: int, dtype: torch.dtype):
        """Pre-allocate static KV caches for all attention blocks."""
        for block in self.blocks:
            if hasattr(block.attn, "setup_cache"):
                block.attn.setup_cache(max_batch_size, self.config.context_length, dtype)

    def reset_caches(self):
        """Reset all static KV caches back to zero."""
        for block in self.blocks:
            if hasattr(block.attn, "kv_cache") and block.attn.kv_cache is not None:
                block.attn.kv_cache.reset()

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
