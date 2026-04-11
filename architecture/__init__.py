"""
YALLM Architecture Module
Expressive, config-driven transformer architectures for pretraining 1B–4B models.
Supports both text-only LLMs and multi-modal VQ-VAE + autoregressive pipelines.
"""

from .config import ModelConfig, VQVAEConfig, MultiModalConfig, PRESETS, get_preset
from .layers import (
    RMSNorm, LayerNorm, RotaryEmbedding, ALiBiPositionBias,
    GroupedQueryAttention, FeedForward,
    apply_rotary_pos_emb,
)
from .model import TransformerBlock, CausalLM
from .vqvae import VQVAE, Encoder, Decoder, VectorQuantizer
from .losses import vqvae_loss, masked_cross_entropy, calc_loss_batch_masked
from .generate import generate, GenerationConfig, generate_image

__all__ = [
    # Config
    "ModelConfig", "VQVAEConfig", "MultiModalConfig", "PRESETS", "get_preset",
    # Transformer layers
    "RMSNorm", "LayerNorm", "RotaryEmbedding", "ALiBiPositionBias",
    "GroupedQueryAttention", "FeedForward", "apply_rotary_pos_emb",
    # Models
    "TransformerBlock", "CausalLM",
    "VQVAE", "Encoder", "Decoder", "VectorQuantizer",
    # Losses
    "vqvae_loss", "masked_cross_entropy", "calc_loss_batch_masked",
    # Generation
    "generate", "GenerationConfig", "generate_image",
]
