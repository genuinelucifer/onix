"""
YALLM Architecture Module
Expressive, config-driven transformer architectures for pretraining 1B–4B models.
"""

from .config import ModelConfig, PRESETS, get_preset
from .layers import (
    RMSNorm, LayerNorm, RotaryEmbedding, ALiBiPositionBias,
    GroupedQueryAttention, FeedForward,
    apply_rotary_pos_emb,
)
from .model import TransformerBlock, CausalLM
from .generate import generate, GenerationConfig

__all__ = [
    "ModelConfig", "PRESETS", "get_preset",
    "RMSNorm", "LayerNorm", "RotaryEmbedding", "ALiBiPositionBias",
    "GroupedQueryAttention", "FeedForward", "apply_rotary_pos_emb",
    "TransformerBlock", "CausalLM",
    "generate", "GenerationConfig",
]
