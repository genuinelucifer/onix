"""
AWQ (Activation-aware Weight Quantization) for the Onix model runner.

Implements a simplified but effective version of the AWQ algorithm:
  1. Calibrate: Run model on sample inputs to collect activation statistics
  2. Identify salient channels based on activation magnitudes
  3. Scale salient channels up before quantization (protecting accuracy)
  4. Quantize to INT4 with per-channel scaling
  5. During inference, dequantize and reverse the salient scaling

This produces significantly better INT4 quality than naive RTN (round-to-nearest)
quantization, typically saving 1-2 perplexity points.

Usage:
    from model_runner.awq import awq_quantize_model, calibrate_model
    scales = calibrate_model(model, tokenizer, device)
    awq_quantize_model(model, scales)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List


# ===========================================================================
#  AWQ INT4 Linear Layer
# ===========================================================================

class AWQInt4Linear(nn.Module):
    """
    Activation-aware Weight-only 4-bit Linear Layer.

    Unlike naive INT4 (RTN), this layer pre-scales salient weight channels
    before quantization. During inference, the scaling is reversed after
    dequantization, preserving accuracy for the most important weights.

    Memory layout:
      - qweight_packed: (out_features, in_features // 2) uint8 — packed 4-bit weights
      - scale:          (out_features, 1) — per-channel quantization scale
      - zero_point:     (out_features, 1) — per-channel zero point
      - salient_scale:  (1, in_features) — per-input-channel AWQ scaling factor
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        assert in_features % 2 == 0, "in_features must be even for 4-bit packing"

        self.register_buffer(
            "qweight_packed",
            torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device),
        )
        self.register_buffer(
            "scale",
            torch.zeros((out_features, 1), dtype=dtype, device=device),
        )
        self.register_buffer(
            "zero_point",
            torch.zeros((out_features, 1), dtype=dtype, device=device),
        )
        # AWQ salient channel scaling — applied to input activations
        self.register_buffer(
            "salient_scale",
            torch.ones((1, in_features), dtype=dtype, device=device),
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(
        cls,
        float_linear: nn.Linear,
        channel_scales: Optional[torch.Tensor] = None,
        dtype=torch.float16,
    ):
        """
        Create an AWQ-quantized linear layer from a float linear layer.

        Args:
            float_linear: Original nn.Linear to quantize
            channel_scales: (in_features,) tensor of per-channel importance scores.
                          Higher values = more salient = scaled up before quantization.
                          If None, falls back to RTN (no AWQ benefit).
            dtype: Compute dtype for scales
        """
        device = float_linear.weight.device
        w = float_linear.weight.detach().clone()  # (out, in)
        in_features = float_linear.in_features

        # Compute AWQ salient scaling
        if channel_scales is not None:
            # Normalize scales: mean=1, with salient channels getting higher values
            # This ensures the overall magnitude doesn't change drastically
            s = channel_scales.to(device).float()
            s = s / s.mean()
            # Clamp to prevent extreme scaling
            s = s.clamp(min=0.1, max=10.0)
            salient_scale = s.unsqueeze(0)  # (1, in_features)

            # Scale weights: multiply columns by salient_scale
            # This "protects" salient channels by amplifying them before quantization
            w = w * salient_scale
        else:
            salient_scale = torch.ones(1, in_features, device=device)

        # Quantize to 4-bit with per-channel min/max (same as RTN but on scaled weights)
        min_val = torch.min(w, dim=1, keepdim=True).values
        max_val = torch.max(w, dim=1, keepdim=True).values
        scale = (max_val - min_val).clamp(min=1e-5) / 15.0
        zero_point = min_val

        qweight = torch.clamp(torch.round((w - zero_point) / scale), 0, 15).to(torch.uint8)

        # Pack pairs into uint8
        w_even = qweight[:, 0::2]
        w_odd = qweight[:, 1::2]
        qweight_packed = w_even | (w_odd << 4)

        # Build the AWQ linear layer
        qlinear = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=(float_linear.bias is not None),
            device=device,
            dtype=dtype,
        )
        qlinear.qweight_packed.copy_(qweight_packed)
        qlinear.scale.copy_(scale.to(dtype))
        qlinear.zero_point.copy_(zero_point.to(dtype))
        qlinear.salient_scale.copy_(salient_scale.to(dtype))

        if float_linear.bias is not None:
            qlinear.bias.data.copy_(float_linear.bias.data.to(dtype))

        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with AWQ dequantization.

        The salient scaling is reversed by dividing the input activations,
        which is mathematically equivalent to dividing the weight columns
        but more efficient (done once on the smaller activation tensor).
        """
        # Reverse AWQ scaling on input: x_scaled = x / salient_scale
        x_scaled = x / self.salient_scale

        # Unpack 4-bit weights
        packed = self.qweight_packed
        w_even = packed & 0x0F
        w_odd = (packed >> 4) & 0x0F
        unpacked = torch.stack([w_even, w_odd], dim=2).view(
            self.out_features, self.in_features
        )

        # Dequantize
        dequantized_weight = unpacked.to(x.dtype) * self.scale + self.zero_point

        return F.linear(x_scaled, dequantized_weight, self.bias)


# ===========================================================================
#  Calibration: collect activation statistics
# ===========================================================================

_CALIBRATION_PROMPTS = [
    "The quick brown fox jumps over the lazy dog.",
    "In a hole in the ground there lived a hobbit.",
    "To be, or not to be, that is the question.",
    "It was the best of times, it was the worst of times.",
    "All happy families are alike; each unhappy family is unhappy in its own way.",
    "Call me Ishmael. Some years ago, never mind how long precisely.",
    "It is a truth universally acknowledged that a single man in possession of a good fortune must be in want of a wife.",
    "The sun shone, having no alternative, on the nothing new.",
    "Stately, plump Buck Mulligan came from the stairhead.",
    "Many years later, as he faced the firing squad.",
    "Once upon a time and a very good time it was there was a moocow coming down along the road.",
    "A screaming comes across the sky. It has happened before, but there is nothing to compare it to now.",
    "Ships at a distance have every man's wish on board.",
    "I am an invisible man. No, I am not a spook like those who haunted Edgar Allan Poe.",
    "The cold passed reluctantly from the earth, and the retiring fogs revealed an army stretched out on the hills.",
    "Mother died today. Or maybe yesterday; I can't be sure.",
]


def calibrate_model(
    model: nn.Module,
    tokenizer,
    device: torch.device,
    prompts: Optional[List[str]] = None,
    max_seq_len: int = 128,
) -> Dict[str, torch.Tensor]:
    """
    Run calibration samples through the model to collect per-layer
    activation statistics for AWQ.

    Returns a dict mapping layer names to per-input-channel activation
    magnitude tensors: { "blocks.0.attn.q_proj": tensor(in_features,), ... }
    """
    if prompts is None:
        prompts = _CALIBRATION_PROMPTS

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()

    # Register forward hooks to capture input activations
    activation_stats: Dict[str, torch.Tensor] = {}
    activation_counts: Dict[str, int] = {}
    hooks = []

    def _make_hook(name: str):
        def hook_fn(module, input, output):
            if len(input) == 0:
                return
            x = input[0].detach()
            if x.ndim == 3:
                # (B, T, D) → compute per-channel magnitude
                mag = x.abs().mean(dim=(0, 1))  # (D,)
            elif x.ndim == 2:
                mag = x.abs().mean(dim=0)
            else:
                return

            if name in activation_stats:
                activation_stats[name] += mag
                activation_counts[name] += 1
            else:
                activation_stats[name] = mag.clone()
                activation_counts[name] = 1
        return hook_fn

    for name, module in raw_model.named_modules():
        if isinstance(module, nn.Linear) and "lm_head" not in name:
            hooks.append(module.register_forward_hook(_make_hook(name)))

    # Run calibration
    with torch.no_grad():
        for prompt in prompts:
            tokens = tokenizer.encode(prompt)[:max_seq_len]
            input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
            raw_model(input_ids)

    # Remove hooks
    for h in hooks:
        h.remove()

    # Average the statistics
    for name in activation_stats:
        activation_stats[name] /= activation_counts[name]

    return activation_stats


# ===========================================================================
#  Model-level AWQ quantization
# ===========================================================================

def awq_quantize_model(
    model: nn.Module,
    activation_stats: Dict[str, torch.Tensor],
    target_dtype=torch.float16,
):
    """
    Recursively replace Linear layers with AWQ INT4 quantized versions.

    Skips lm_head (output projection) and layers with odd in_features.

    Args:
        model: The model to quantize in-place
        activation_stats: Dict from calibrate_model() mapping layer names
                         to per-channel activation magnitudes
        target_dtype: Dtype for quantization scales
    """
    _awq_replace_recursive(model, "", activation_stats, target_dtype)


def _awq_replace_recursive(
    module: nn.Module,
    prefix: str,
    activation_stats: Dict[str, torch.Tensor],
    target_dtype: torch.dtype,
):
    """Recursively replace nn.Linear with AWQInt4Linear."""
    for name, child in module.named_children():
        full_name = f"{prefix}.{name}" if prefix else name

        if isinstance(child, nn.Linear):
            # Skip lm_head and layers with odd in_features
            if name == "lm_head":
                continue
            if child.in_features % 2 != 0:
                continue

            # Look up activation statistics for this layer
            channel_scales = activation_stats.get(full_name, None)

            qlinear = AWQInt4Linear.from_float(
                child,
                channel_scales=channel_scales,
                dtype=target_dtype,
            )
            setattr(module, name, qlinear)
        else:
            _awq_replace_recursive(child, full_name, activation_stats, target_dtype)
