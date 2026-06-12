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
    use_kv_cache: bool = True


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
    use_kv_cache: bool = True,
    metrics: Optional[dict] = None,
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
        use_kv_cache: whether to use key-value caching
        metrics: dict to collect execution times (ttft, decode_time)

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
    if hasattr(model, "reset_caches"):
        model.reset_caches()
    import time

    if not use_kv_cache:
        # Fallback to eager execution without KV cache (original behavior)
        with torch.no_grad():
            ttft = 0.0
            decode_start = time.perf_counter()
            for step in range(max_new_tokens):
                step_start = time.perf_counter()
                idx_cond = idx[:, -context_size:]
                raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
                logits = raw_model(idx_cond)
                if isinstance(logits, tuple):
                    logits = logits[0]
                logits = logits[:, -1, :]  # (B, vocab)

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
                    sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                    sorted_logits[sorted_mask] = float("-inf")
                    logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

                # Sample or greedy
                if temperature > 0.0:
                    logits = logits / temperature
                    probs = F.softmax(logits, dim=-1)
                    idx_next = torch.multinomial(probs, num_samples=1)
                else:
                    idx_next = torch.argmax(logits, dim=-1, keepdim=True)

                if step == 0:
                    ttft = time.perf_counter() - step_start
                    if metrics is not None:
                        metrics["ttft"] = ttft
                    decode_start = time.perf_counter()

                if eos_id is not None and idx_next.item() == eos_id:
                    break

                idx = torch.cat((idx, idx_next), dim=1)
            
            decode_time = time.perf_counter() - decode_start
            if metrics is not None:
                metrics["decode_time"] = max(0.0, decode_time)
        return idx

    # KV Cache mode
    with torch.no_grad():
        B, T_prompt = idx.shape
        kv_caches = None
        
        # Check if the model has static caches, unwrapping compiled model if necessary
        underlying_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        is_static = (
            hasattr(underlying_model, "blocks")
            and len(underlying_model.blocks) > 0
            and hasattr(underlying_model.blocks[0].attn, "kv_cache")
            and underlying_model.blocks[0].attn.kv_cache is not None
        )

        # Pre-allocate output buffer and position IDs to avoid dynamic allocations in the decode loop
        prefill_len = min(T_prompt, context_size)
        total_len = prefill_len + max_new_tokens
        
        out_idx = torch.empty((B, total_len), dtype=torch.long, device=idx.device)
        out_idx[:, :prefill_len] = idx[:, -prefill_len:]
        
        all_position_ids = torch.arange(total_len, device=idx.device).unsqueeze(0).expand(B, total_len)

        # Position ids and input for the prefill
        position_ids = all_position_ids[:, :prefill_len]
        idx_input = out_idx[:, :prefill_len]

        # Run first step to prefill cache (Measure TTFT)
        t0 = time.perf_counter()
        if is_static:
            raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
            res = raw_model(idx_input, position_ids=position_ids, use_cache=True)
            logits = res[0] if isinstance(res, tuple) else res
            kv_caches = None
        else:
            res = model(idx_input, position_ids=position_ids, use_cache=True)
            logits, kv_caches = res[0], res[1]
            
        logits = logits[:, -1, :]  # shape (B, vocab_size)
        ttft = time.perf_counter() - t0
        if metrics is not None:
            metrics["ttft"] = ttft

        decode_start = time.perf_counter()
        t_sample_total = 0.0
        t_overhead_total = 0.0
        t_forward_total = 0.0

        for step in range(max_new_tokens):
            t_step_start = time.perf_counter()
            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for token_id in set(out_idx[0, :prefill_len + step].tolist()):
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
                sorted_mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[sorted_mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

            # Sample or greedy
            if temperature > 0.0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
            else:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)

            if eos_id is not None and idx_next.item() == eos_id:
                t_sample_total += (time.perf_counter() - t_step_start)
                break
            t_sample_total += (time.perf_counter() - t_step_start)

            t_ov_start = time.perf_counter()
            # Write token in-place to pre-allocated output buffer
            curr_idx = prefill_len + step
            out_idx[:, curr_idx : curr_idx + 1] = idx_next

            # Crop the KV caches if they exceed context_size - 1 (only for dynamic cache)
            if not is_static and kv_caches is not None:
                cache_seq_len = kv_caches[0][0].shape[-2]
                if cache_seq_len >= context_size - 1:
                    keep = context_size - 1
                    kv_caches = [
                        (k[:, :, -keep:, :], v[:, :, -keep:, :])
                        for k, v in kv_caches
                    ]

            # Prepare inputs for the decode step (slice pre-allocated position IDs)
            position_ids = all_position_ids[:, curr_idx : curr_idx + 1]
            t_overhead_total += (time.perf_counter() - t_ov_start)

            t_fw_start = time.perf_counter()
            # Run decode step
            if is_static:
                res = model(idx_next, position_ids=position_ids, use_cache=True)
                logits = res[0] if isinstance(res, tuple) else res
            else:
                res = model(idx_next, position_ids=position_ids, kv_caches=kv_caches, use_cache=True)
                logits, kv_caches = res[0], res[1]
            logits = logits[:, -1, :]
            t_forward_total += (time.perf_counter() - t_fw_start)

        decode_time = time.perf_counter() - decode_start
        if metrics is not None:
            metrics["decode_time"] = decode_time
            metrics["forward_time"] = t_forward_total
            metrics["sample_time"] = t_sample_total
            metrics["overhead_time"] = t_overhead_total

        if max_new_tokens == 0:
            return out_idx[:, :prefill_len]

        final_len = prefill_len + step if (eos_id is not None and idx_next.item() == eos_id) else total_len
        return out_idx[:, :final_len]


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
        use_kv_cache=getattr(gen_config, "use_kv_cache", True),
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
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    device = next(raw_model.parameters()).device

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

    raw_model.eval()
    visual_token_ids = []
    with torch.no_grad():
        for _ in range(num_visual):
            # Crop to context window
            idx_cond = idx[:, -context_size:]
            logits = raw_model(idx_cond)[:, -1, :]  # (1, vocab_size)

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
