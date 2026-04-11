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


# ===========================================================================
#  Image generation (multi-modal inference)
# ===========================================================================

def generate_image(
    model,
    vqvae,
    text_prompt: str,
    tokenizer,
    mm_config,
    temperature: float = 0.9,
    top_k: int = 100,
    top_p: float = None,
) -> tuple:
    """
    Generate an image from a text prompt using a trained multi-modal LLM
    and a frozen VQ-VAE decoder.

    Pipeline:
      1. Tokenize text prompt via BPE
      2. Append <IMG_START> token
      3. Autoregressively generate visual tokens
      4. Reshape visual tokens to 2D grid
      5. Decode via frozen VQ-VAE to pixel space

    Args:
        model: Trained CausalLM (multi-modal, joint vocab)
        vqvae: Frozen VQVAE model (for decoding tokens → pixels)
        text_prompt: Text description of the desired image
        tokenizer: BPE tokenizer (tiktoken)
        mm_config: MultiModalConfig with token space info
        temperature: Sampling temperature
        top_k: Top-k filtering
        top_p: Nucleus sampling threshold

    Returns:
        (image_tensor, visual_tokens)
        image_tensor: (1, C, H, W) reconstructed image in [-1, 1]
        visual_tokens: (1, num_visual_tokens) generated token indices
    """
    device = next(model.parameters()).device

    # Step 1: Tokenize text
    text_tokens = tokenizer.encode(text_prompt)
    # Truncate if needed
    if len(text_tokens) > mm_config.max_text_tokens:
        text_tokens = text_tokens[:mm_config.max_text_tokens]

    # Step 2: Prepare initial sequence: text_tokens + <IMG_START>
    sequence = text_tokens + [mm_config.img_start_id]
    idx = torch.tensor([sequence], dtype=torch.long, device=device)

    # Step 3: Autoregressively generate visual tokens
    num_visual = mm_config.num_visual_tokens
    context_size = mm_config.max_seq_length

    model.eval()
    visual_token_ids = []
    with torch.no_grad():
        for _ in range(num_visual):
            # Crop to context window
            idx_cond = idx[:, -context_size:]
            logits = model(idx_cond)[:, -1, :]  # (1, vocab_size)

            # Restrict sampling to visual token range only
            # Zero out logits for non-visual tokens
            mask = torch.ones_like(logits) * float("-inf")
            visual_start = mm_config.text_vocab_size
            visual_end = visual_start + mm_config.visual_vocab_size
            mask[:, visual_start:visual_end] = 0
            logits = logits + mask

            # Top-k filtering
            if top_k is not None:
                top_logits, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                min_val = top_logits[:, -1].unsqueeze(-1)
                logits = torch.where(
                    logits < min_val,
                    torch.full_like(logits, float("-inf")),
                    logits,
                )

            # Top-p filtering
            if top_p is not None and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )
                sorted_mask = (
                    cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                )
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            # Check for <IMG_END>
            if idx_next.item() == mm_config.img_end_id:
                break

            visual_token_ids.append(idx_next.item())
            idx = torch.cat((idx, idx_next), dim=1)

    # Step 4: Convert visual tokens to codebook indices
    # Visual tokens in the joint vocab are offset by text_vocab_size
    codebook_indices = [t - mm_config.text_vocab_size for t in visual_token_ids]

    # Pad or truncate to exact num_visual_tokens
    if len(codebook_indices) < num_visual:
        codebook_indices.extend([0] * (num_visual - len(codebook_indices)))
    codebook_indices = codebook_indices[:num_visual]

    indices_tensor = torch.tensor(
        [codebook_indices], dtype=torch.long, device=device
    )

    # Step 5: Decode via VQ-VAE
    image = vqvae.decode(indices_tensor)

    return image, indices_tensor
