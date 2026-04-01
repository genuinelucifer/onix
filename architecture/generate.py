"""
Text generation with configurable decoding strategies.

Supports: greedy, temperature sampling, top-k, top-p (nucleus),
and repetition penalty.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class GenerationConfig:
    """Controls how text is generated."""
    max_new_tokens: int = 100
    temperature: float = 0.0      # 0 = greedy
    top_k: Optional[int] = None
    top_p: Optional[float] = None  # nucleus sampling
    repetition_penalty: float = 1.0
    eos_id: Optional[int] = None


def generate(
    model,
    idx: torch.Tensor,
    max_new_tokens: int = 100,
    context_size: Optional[int] = None,
    temperature: float = 0.0,
    top_k: Optional[int] = None,
    top_p: Optional[float] = None,
    repetition_penalty: float = 1.0,
    eos_id: Optional[int] = None,
) -> torch.Tensor:
    """
    Generate tokens autoregressively.

    Args:
        model: CausalLM or GPT2 model
        idx: (1, seq_len) initial token ids
        max_new_tokens: number of tokens to generate
        context_size: max context window (auto-detected from model if None)
        temperature: sampling temperature (0 = greedy)
        top_k: keep only top-k logits
        top_p: keep only tokens with cumulative prob <= top_p
        repetition_penalty: penalize repeated tokens (> 1.0 = more penalty)
        eos_id: stop generation when this token is produced

    Returns:
        (1, seq_len + generated) token ids
    """
    # Auto-detect context size
    if context_size is None:
        if hasattr(model, "config"):
            context_size = model.config.context_length
        elif hasattr(model, "cfg"):
            context_size = model.cfg.get("context_length", model.cfg["context_length"])
        else:
            context_size = 1024

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Crop to context window
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)[:, -1, :]  # (B, vocab)

            # Repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(idx[0].tolist()):
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty

            # Top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = top_logits[:, -1].unsqueeze(-1)
                logits = torch.where(logits < min_val, torch.full_like(logits, float("-inf")), logits)

            # Top-p (nucleus) filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                # Remove tokens with cumulative prob above threshold
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                # Scatter back
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample or greedy
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Stop on EOS
            if eos_id is not None and idx_next.item() == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

    return idx


def generate_from_config(
    model,
    idx: torch.Tensor,
    gen_config: GenerationConfig,
    context_size: Optional[int] = None,
) -> torch.Tensor:
    """Generate using a GenerationConfig object."""
    return generate(
        model, idx,
        max_new_tokens=gen_config.max_new_tokens,
        context_size=context_size,
        temperature=gen_config.temperature,
        top_k=gen_config.top_k,
        top_p=gen_config.top_p,
        repetition_penalty=gen_config.repetition_penalty,
        eos_id=gen_config.eos_id,
    )
