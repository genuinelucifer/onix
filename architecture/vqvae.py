"""
VQ-VAE (Vector Quantized Variational AutoEncoder) for image tokenization.

Converts images into discrete token sequences that can be modeled
autoregressively by a transformer. Architecture:

  Encoder: Image (B,C,H,W) → Latent grid (B, codebook_dim, h, w)
  VectorQuantizer: Continuous → Discrete codebook indices + quantized vectors
  Decoder: Quantized (B, codebook_dim, h, w) → Reconstructed image (B,C,H,W)

References:
  - van den Oord et al. "Neural Discrete Representation Learning" (2017)
  - Esser et al. "Taming Transformers for High-Resolution Image Synthesis" (2021)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

if TYPE_CHECKING:
    from .config import VQVAEConfig


# ===========================================================================
#  Building blocks
# ===========================================================================

class ResidualBlock(nn.Module):
    """Conv residual block: Conv → GroupNorm → SiLU → Conv → GroupNorm → + residual."""

    def __init__(self, in_channels: int, out_channels: int, num_groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(min(num_groups, out_channels), out_channels)
        self.act = nn.SiLU()

        # Skip connection projection if channels change
        if in_channels != out_channels:
            self.skip_proj = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return self.act(h + self.skip_proj(x))


class Downsample(nn.Module):
    """2x spatial downsampling via strided convolution."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class Upsample(nn.Module):
    """2x spatial upsampling via nearest-neighbor interpolation + conv."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)


# ===========================================================================
#  Encoder
# ===========================================================================

class Encoder(nn.Module):
    """
    Convolutional encoder that downsamples an image into a spatial grid of
    continuous latent vectors.

    Architecture per stage:
      [ResBlock × num_res_blocks] → Downsample (except last stage)
    Final: project to codebook_dim.
    """

    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        ch = config.base_channels
        mults = config.channel_multipliers

        # Initial convolution
        self.conv_in = nn.Conv2d(config.image_channels, ch * mults[0], 3, padding=1)

        # Downsampling stages
        self.stages = nn.ModuleList()
        in_ch = ch * mults[0]
        for i, mult in enumerate(mults):
            out_ch = ch * mult
            stage = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                stage.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            # Downsample between stages (not after the last one)
            if i < len(mults) - 1:
                stage.append(Downsample(out_ch))
            self.stages.append(stage)

        # Final residual block + projection to codebook dimension
        self.final_res = ResidualBlock(in_ch, in_ch)
        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv_out = nn.Conv2d(in_ch, config.codebook_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W) input image, normalized to [-1, 1]
        Returns:
            (B, codebook_dim, h, w) continuous latent grid
        """
        h = self.conv_in(x)
        for stage in self.stages:
            for layer in stage:
                h = layer(h)
        h = self.final_res(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return h


# ===========================================================================
#  Vector Quantizer
# ===========================================================================

class VQOutput(NamedTuple):
    """Output of the vector quantizer."""
    quantized: torch.Tensor     # (B, codebook_dim, h, w) quantized latent vectors
    indices: torch.Tensor       # (B, h*w) codebook indices (integer tokens)
    commitment_loss: torch.Tensor  # Scalar commitment loss
    codebook_usage: float       # Fraction of codebook entries used in this batch


class VectorQuantizer(nn.Module):
    """
    Discrete codebook with nearest-neighbor lookup and EMA updates.

    During forward:
      1. Find the nearest codebook vector for each spatial position
      2. Replace continuous vectors with their quantized counterparts
      3. Use straight-through estimator for gradients (gradients flow through
         as if quantization didn't happen)

    Codebook update:
      - If ema_decay > 0: update codebook via Exponential Moving Average
        (no gradient to codebook needed)
      - If ema_decay == 0: codebook is updated via gradient descent
        (commitment loss gradient flows to codebook)
    """

    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.n_embed = config.codebook_size
        self.embed_dim = config.codebook_dim
        self.commitment_weight = config.commitment_weight
        self.ema_decay = config.ema_decay

        # Codebook embeddings
        self.embedding = nn.Embedding(self.n_embed, self.embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / self.n_embed, 1.0 / self.n_embed)

        # EMA tracking
        if self.ema_decay > 0:
            self.register_buffer("ema_cluster_size", torch.zeros(self.n_embed))
            self.register_buffer("ema_embed_sum", self.embedding.weight.data.clone())

    def forward(self, z: torch.Tensor) -> VQOutput:
        """
        Args:
            z: (B, D, h, w) continuous latent vectors from encoder
        Returns:
            VQOutput with quantized vectors, indices, and losses
        """
        B, D, h, w = z.shape

        # Reshape to (B*h*w, D) for distance computation
        z_flat = z.permute(0, 2, 3, 1).reshape(-1, D)  # (N, D)

        # Compute distances to all codebook entries
        # ||z - e||^2 = ||z||^2 + ||e||^2 - 2*z·e
        dist = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1, keepdim=False)
            - 2 * z_flat @ self.embedding.weight.t()
        )  # (N, n_embed)

        # Find nearest codebook entry
        indices = dist.argmin(dim=-1)  # (N,)
        quantized_flat = self.embedding(indices)  # (N, D)

        # Compute codebook usage (fraction of entries used in this batch)
        unique_codes = indices.unique().numel()
        usage = unique_codes / self.n_embed

        # EMA codebook update (during training only)
        if self.training and self.ema_decay > 0:
            with torch.no_grad():
                # Count how many vectors map to each codebook entry
                one_hot = F.one_hot(indices, self.n_embed).float()  # (N, n_embed)
                cluster_size = one_hot.sum(dim=0)  # (n_embed,)
                embed_sum = one_hot.t() @ z_flat  # (n_embed, D)

                self.ema_cluster_size.mul_(self.ema_decay).add_(
                    cluster_size, alpha=1 - self.ema_decay
                )
                self.ema_embed_sum.mul_(self.ema_decay).add_(
                    embed_sum, alpha=1 - self.ema_decay
                )

                # Laplace smoothing to avoid division by zero
                n = self.ema_cluster_size.sum()
                cluster_size_smooth = (
                    (self.ema_cluster_size + 1e-5)
                    / (n + self.n_embed * 1e-5) * n
                )
                self.embedding.weight.data.copy_(
                    self.ema_embed_sum / cluster_size_smooth.unsqueeze(1)
                )

        # Commitment loss: encourage encoder output to stay close to codebook
        commitment_loss = F.mse_loss(z_flat.detach(), quantized_flat) + \
                          self.commitment_weight * F.mse_loss(z_flat, quantized_flat.detach())

        # Straight-through estimator: copy gradients from quantized to z
        quantized_flat = z_flat + (quantized_flat - z_flat).detach()

        # Reshape back to spatial
        quantized = quantized_flat.reshape(B, h, w, D).permute(0, 3, 1, 2)  # (B, D, h, w)
        indices = indices.reshape(B, h * w)  # (B, h*w)

        return VQOutput(quantized, indices, commitment_loss, usage)


# ===========================================================================
#  Decoder
# ===========================================================================

class Decoder(nn.Module):
    """
    Convolutional decoder that upsamples quantized latent vectors back to
    a full-resolution image. Mirror of the Encoder.
    """

    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        ch = config.base_channels
        mults = config.channel_multipliers

        # Input projection from codebook dim
        top_ch = ch * mults[-1]
        self.conv_in = nn.Conv2d(config.codebook_dim, top_ch, 3, padding=1)
        self.initial_res = ResidualBlock(top_ch, top_ch)

        # Upsampling stages (reverse order of encoder)
        self.stages = nn.ModuleList()
        in_ch = top_ch
        for i, mult in enumerate(reversed(mults)):
            out_ch = ch * mult
            stage = nn.ModuleList()
            for _ in range(config.num_res_blocks):
                stage.append(ResidualBlock(in_ch, out_ch))
                in_ch = out_ch
            # Upsample between stages (not after the last one)
            if i < len(mults) - 1:
                stage.append(Upsample(out_ch))
            self.stages.append(stage)

        # Final output
        self.norm_out = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv_out = nn.Conv2d(in_ch, config.image_channels, 3, padding=1)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_q: (B, codebook_dim, h, w) quantized latent vectors
        Returns:
            (B, C, H, W) reconstructed image in [-1, 1]
        """
        h = self.conv_in(z_q)
        h = self.initial_res(h)
        for stage in self.stages:
            for layer in stage:
                h = layer(h)
        h = F.silu(self.norm_out(h))
        h = self.conv_out(h)
        return torch.tanh(h)


# ===========================================================================
#  Full VQ-VAE Model
# ===========================================================================

class VQVAE(nn.Module):
    """
    Complete Vector Quantized Variational AutoEncoder.

    Training:  images → encoder → quantizer → decoder → reconstruction
    Encoding:  images → encoder → quantizer → integer token indices
    Decoding:  integer token indices → codebook lookup → decoder → image

    Usage:
        config = VQVAEConfig(image_size=256, codebook_size=8192)
        model = VQVAE(config)

        # Training
        recon, vq_loss, indices = model(images)

        # Encode images to tokens
        tokens = model.encode(images)     # (B, num_visual_tokens) int64

        # Decode tokens to images
        images = model.decode(tokens)     # (B, C, H, W) float in [-1, 1]
    """

    def __init__(self, config: VQVAEConfig):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.quantizer = VectorQuantizer(config)
        self.decoder = Decoder(config)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full forward pass for training.

        Args:
            x: (B, C, H, W) input images, normalized to [-1, 1]
        Returns:
            recon: (B, C, H, W) reconstructed images
            vq_loss: scalar (commitment + codebook loss)
            indices: (B, num_visual_tokens) codebook indices
        """
        z = self.encoder(x)

        if self.config.grad_checkpointing and self.training:
            # Quantizer has non-trivial memory usage
            vq_out = torch.utils.checkpoint.checkpoint(
                self.quantizer, z, use_reentrant=False
            )
        else:
            vq_out = self.quantizer(z)

        recon = self.decoder(vq_out.quantized)
        return recon, vq_out.commitment_loss, vq_out.indices

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to discrete token indices (for LLM data preparation).

        Args:
            x: (B, C, H, W) input images, normalized to [-1, 1]
        Returns:
            (B, num_visual_tokens) integer token indices
        """
        self.eval()
        z = self.encoder(x)
        vq_out = self.quantizer(z)
        return vq_out.indices

    @torch.no_grad()
    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode integer token indices back to images (for inference).

        Args:
            indices: (B, num_visual_tokens) integer codebook indices
        Returns:
            (B, C, H, W) reconstructed images in [-1, 1]
        """
        self.eval()
        B = indices.shape[0]
        gs = self.config.latent_grid_size

        # Look up codebook vectors
        z_q = self.quantizer.embedding(indices)  # (B, N, D)
        z_q = z_q.reshape(B, gs, gs, -1).permute(0, 3, 1, 2)  # (B, D, h, w)

        return self.decoder(z_q)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def summary(self) -> str:
        n = self.param_count()
        size = f"{n / 1e6:.1f}M"
        return (
            f"{self.config.summary()}\n"
            f"  Actual params: {n:,} (~{size})"
        )
