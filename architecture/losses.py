"""
Loss functions for VQ-VAE and multi-modal training.

Includes:
  - vqvae_loss: MSE reconstruction + codebook commitment loss
  - masked_cross_entropy: cross-entropy with position masking (for visual-only loss)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def vqvae_loss(
    recon_images: torch.Tensor,
    target_images: torch.Tensor,
    vq_commitment_loss: torch.Tensor,
    recon_weight: float = 1.0,
    commitment_weight: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined VQ-VAE training loss.

    Args:
        recon_images: (B, C, H, W) reconstructed images from decoder
        target_images: (B, C, H, W) original input images
        vq_commitment_loss: scalar commitment loss from VectorQuantizer
        recon_weight: weight for reconstruction loss
        commitment_weight: weight for commitment loss

    Returns:
        total_loss: scalar combined loss
        loss_dict: breakdown of individual loss components
    """
    recon_loss = F.mse_loss(recon_images, target_images)
    total = recon_weight * recon_loss + commitment_weight * vq_commitment_loss

    loss_dict = {
        "recon_loss": recon_loss.item(),
        "commitment_loss": vq_commitment_loss.item(),
        "total_loss": total.item(),
    }
    return total, loss_dict


def masked_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Cross-entropy loss computed only at masked positions.

    Used in multi-modal training where we only want loss on visual token
    positions (not text prompt positions).

    Args:
        logits: (B, T, vocab_size) model predictions
        targets: (B, T) target token IDs
        loss_mask: (B, T) binary mask — 1 for positions to include in loss, 0 to ignore

    Returns:
        Scalar mean loss over masked positions
    """
    B, T, V = logits.shape

    # Compute per-position cross-entropy (no reduction)
    ce = F.cross_entropy(
        logits.reshape(-1, V),
        targets.reshape(-1),
        reduction="none",
    )
    ce = ce.reshape(B, T)  # (B, T)

    # Apply mask and average over non-zero positions
    masked_loss = (ce * loss_mask).sum()
    num_positions = loss_mask.sum().clamp(min=1)

    return masked_loss / num_positions


def calc_loss_batch_masked(
    inp: torch.Tensor,
    tgt: torch.Tensor,
    loss_mask: torch.Tensor,
    model: nn.Module,
    device: torch.device,
) -> torch.Tensor:
    """
    Convenience wrapper: forward pass + masked cross-entropy.

    Args:
        inp: (B, T) input token IDs
        tgt: (B, T) target token IDs
        loss_mask: (B, T) binary mask for loss positions
        model: CausalLM
        device: torch device
    """
    inp, tgt, loss_mask = inp.to(device), tgt.to(device), loss_mask.to(device)
    logits = model(inp)
    return masked_cross_entropy(logits, tgt, loss_mask)
