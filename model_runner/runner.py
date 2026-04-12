#!/usr/bin/env python3
"""
Model Runner — load trained YALLM checkpoints and run inference.

Supports three model types:
  - vqvae:      Image reconstruction (encode → decode)
  - multimodal: Text-to-image generation (text → visual tokens → VQ-VAE decode)
  - llm:        Text generation (multi-turn chat)
"""

import os
import json
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Tuple

# Enable Flash Attention on AMD consumer GPUs before importing PyTorch
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

import torch
import numpy as np
from PIL import Image

# Add parent to path so we can import architecture/model modules
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from architecture.config import ModelConfig, VQVAEConfig, MultiModalConfig
from architecture.model import CausalLM
from architecture.vqvae import VQVAE
from architecture.generate import generate, generate_image
from model import get_tokenizer, text_to_token_ids, token_ids_to_text


# ---------------------------------------------------------------------------
#  Data classes
# ---------------------------------------------------------------------------

@dataclass
class LoadedModel:
    """Container for a loaded model + metadata."""
    model_type: str                     # "vqvae", "multimodal", "llm"
    model: torch.nn.Module              # The main model (VQVAE, CausalLM)
    config: dict                        # Full config dict from config.json
    device: torch.device
    checkpoint_path: str

    # Type-specific extras
    vqvae_config: Optional[VQVAEConfig] = None
    model_config: Optional[ModelConfig] = None
    mm_config: Optional[MultiModalConfig] = None
    frozen_vqvae: Optional[VQVAE] = None  # For multimodal mode
    tokenizer: object = None


@dataclass
class GenerationParams:
    """Parameters for text/image generation."""
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.1
    eos_id: Optional[int] = None


# ---------------------------------------------------------------------------
#  Model detection and loading
# ---------------------------------------------------------------------------

def detect_model_type(config: dict) -> str:
    """Detect model type from config.json contents."""
    model_type = config.get("model_type", "")
    if model_type == "vqvae":
        return "vqvae"
    elif model_type == "multimodal":
        return "multimodal"
    elif "architecture" in config:
        return "llm"
    else:
        raise ValueError(
            f"Cannot detect model type from config. "
            f"Keys: {list(config.keys())}"
        )


def find_config_json(checkpoint_path: str) -> Path:
    """Find config.json in the same directory as the checkpoint."""
    # Use the original path's parent (don't resolve symlinks which could
    # change the directory).  Path.parent on a symlink still gives the
    # directory the symlink lives in, which is what we want.
    ckpt = Path(checkpoint_path)
    model_dir = ckpt.parent.resolve()

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found in {model_dir}. "
            f"Expected alongside checkpoint file."
        )
    return config_path


def _resolve_checkpoint(path: str) -> Path:
    """Resolve a checkpoint path that may be a directory or a file."""
    p = Path(path)
    if p.is_dir():
        # Look for best checkpoint inside the directory
        for candidate in ["checkpoint_final.pt", "checkpoint_latest.pt"]:
            cp = p / candidate
            if cp.exists():
                return cp.resolve()
        # Fall back to any checkpoint file
        pt_files = sorted(p.glob("checkpoint_*.pt"))
        if pt_files:
            return pt_files[-1].resolve()
        raise FileNotFoundError(
            f"No checkpoint files found in directory: {p}"
        )
    # It's a file path
    p = p.resolve()
    if not p.exists():
        raise FileNotFoundError(f"Checkpoint not found: {p}")
    return p


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
) -> LoadedModel:
    """
    Load a model from a checkpoint file or model directory.

    Auto-detects model type from config.json in the same directory.

    Args:
        checkpoint_path: Path to a .pt file OR a model directory
        device: Device to load onto ("cuda", "cpu", etc.)

    Returns:
        LoadedModel with the model ready for inference
    """
    ckpt_path = _resolve_checkpoint(checkpoint_path)

    dev = torch.device(device)

    # Load config
    config_path = find_config_json(str(ckpt_path))
    with open(config_path) as f:
        full_config = json.load(f)

    model_type = detect_model_type(full_config)
    tokenizer = get_tokenizer()

    if model_type == "vqvae":
        return _load_vqvae(ckpt_path, full_config, dev, tokenizer)
    elif model_type == "multimodal":
        return _load_multimodal(ckpt_path, full_config, dev, tokenizer)
    elif model_type == "llm":
        return _load_llm(ckpt_path, full_config, dev, tokenizer)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_vqvae(
    ckpt_path: Path, config: dict, device: torch.device, tokenizer
) -> LoadedModel:
    """Load a VQ-VAE model."""
    vqvae_config = VQVAEConfig.from_dict(config["vqvae"])
    model = VQVAE(vqvae_config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    return LoadedModel(
        model_type="vqvae",
        model=model,
        config=config,
        device=device,
        checkpoint_path=str(ckpt_path),
        vqvae_config=vqvae_config,
        tokenizer=tokenizer,
    )


def _load_multimodal(
    ckpt_path: Path, config: dict, device: torch.device, tokenizer
) -> LoadedModel:
    """Load a multimodal model (transformer + frozen VQ-VAE)."""
    mm_config = MultiModalConfig.from_dict(config["multimodal"])

    # Build transformer with correct vocab/context
    transformer_config = mm_config.build_transformer_config()
    model = CausalLM(transformer_config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    # Load frozen VQ-VAE for decoding
    vqvae_ckpt_path = mm_config.vqvae_checkpoint
    # Resolve relative paths from project root
    if not Path(vqvae_ckpt_path).is_absolute():
        vqvae_ckpt_path = str(_PROJECT_ROOT / vqvae_ckpt_path)

    frozen_vqvae = None
    if Path(vqvae_ckpt_path).exists():
        frozen_vqvae = VQVAE(mm_config.vqvae)
        vq_ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", weights_only=False)
        if "model_state_dict" in vq_ckpt:
            frozen_vqvae.load_state_dict(vq_ckpt["model_state_dict"])
        else:
            frozen_vqvae.load_state_dict(vq_ckpt)
        frozen_vqvae.to(device)
        frozen_vqvae.eval()
        for p in frozen_vqvae.parameters():
            p.requires_grad_(False)

    return LoadedModel(
        model_type="multimodal",
        model=model,
        config=config,
        device=device,
        checkpoint_path=str(ckpt_path),
        model_config=transformer_config,
        mm_config=mm_config,
        frozen_vqvae=frozen_vqvae,
        tokenizer=tokenizer,
    )


def _load_llm(
    ckpt_path: Path, config: dict, device: torch.device, tokenizer
) -> LoadedModel:
    """Load an LLM (CausalLM) model."""
    model_config = ModelConfig.from_dict(config["architecture"])
    model = CausalLM(model_config)

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    return LoadedModel(
        model_type="llm",
        model=model,
        config=config,
        device=device,
        checkpoint_path=str(ckpt_path),
        model_config=model_config,
        tokenizer=tokenizer,
    )


# ---------------------------------------------------------------------------
#  Inference functions
# ---------------------------------------------------------------------------

def run_vqvae_inference(
    loaded: LoadedModel,
    image: Image.Image,
) -> Tuple[Image.Image, int, float]:
    """
    Run VQ-VAE encode → decode on an image.

    Returns:
        (reconstructed_image, num_tokens, codebook_usage_fraction)
    """
    assert loaded.model_type == "vqvae"
    vqvae = loaded.model
    cfg = loaded.vqvae_config

    # Preprocess: resize to expected size, normalize to [-1, 1]
    img = image.convert("RGB").resize(
        (cfg.image_size, cfg.image_size), Image.LANCZOS
    )
    img_tensor = torch.tensor(
        np.array(img), dtype=torch.float32
    ).permute(2, 0, 1) / 127.5 - 1.0  # [0,255] → [-1,1]
    img_tensor = img_tensor.unsqueeze(0).to(loaded.device)

    # Encode → quantize → decode
    with torch.no_grad():
        indices = vqvae.encode(img_tensor)       # (1, num_tokens)
        recon_tensor = vqvae.decode(indices)      # (1, C, H, W) in [-1, 1]

    num_tokens = indices.shape[1]
    unique_codes = indices.cpu().unique().numel()
    usage = unique_codes / cfg.codebook_size

    # Convert back to PIL
    recon_np = ((recon_tensor[0].cpu().clamp(-1, 1) + 1) * 127.5).byte()
    recon_np = recon_np.permute(1, 2, 0).numpy()
    recon_image = Image.fromarray(recon_np)

    return recon_image, num_tokens, usage


def run_multimodal_inference(
    loaded: LoadedModel,
    text_prompt: str,
    params: GenerationParams,
) -> Tuple[Optional[Image.Image], str]:
    """
    Generate an image from a text prompt using the multimodal model.

    Returns:
        (generated_image_or_None, status_message)
    """
    assert loaded.model_type == "multimodal"

    if loaded.frozen_vqvae is None:
        return None, "Error: VQ-VAE decoder not loaded. Cannot generate images."

    try:
        image_tensor, visual_tokens = generate_image(
            model=loaded.model,
            vqvae=loaded.frozen_vqvae,
            text_prompt=text_prompt,
            tokenizer=loaded.tokenizer,
            mm_config=loaded.mm_config,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
        )

        # Convert [-1, 1] tensor to PIL
        img_np = ((image_tensor[0].cpu().clamp(-1, 1) + 1) * 127.5).byte()
        img_np = img_np.permute(1, 2, 0).numpy()
        pil_image = Image.fromarray(img_np)

        n_tokens = visual_tokens.shape[1]
        return pil_image, f"Generated {n_tokens} visual tokens"

    except Exception as e:
        return None, f"Generation failed: {e}"


def run_llm_inference(
    loaded: LoadedModel,
    conversation_text: str,
    params: GenerationParams,
) -> str:
    """
    Generate text continuation from the conversation so far.

    Args:
        loaded: Loaded LLM model
        conversation_text: Full conversation text to continue from
        params: Generation parameters

    Returns:
        Generated text (new tokens only, not the input)
    """
    assert loaded.model_type == "llm"

    tokenizer = loaded.tokenizer
    model = loaded.model
    ctx_size = loaded.model_config.context_length

    # Tokenize input
    input_ids = text_to_token_ids(conversation_text, tokenizer).to(loaded.device)

    # Truncate to fit context window (keep recent tokens)
    if input_ids.shape[1] > ctx_size - params.max_new_tokens:
        keep = ctx_size - params.max_new_tokens
        input_ids = input_ids[:, -keep:]

    input_len = input_ids.shape[1]

    # Generate
    with torch.no_grad():
        output_ids = generate(
            model=model,
            idx=input_ids,
            max_new_tokens=params.max_new_tokens,
            context_size=ctx_size,
            temperature=params.temperature,
            top_k=params.top_k,
            top_p=params.top_p,
            repetition_penalty=params.repetition_penalty,
            eos_id=params.eos_id,
        )

    # Extract only the new tokens
    new_tokens = output_ids[:, input_len:]
    generated_text = token_ids_to_text(new_tokens, tokenizer)

    return generated_text


# ---------------------------------------------------------------------------
#  Model info
# ---------------------------------------------------------------------------

def get_model_info(loaded: LoadedModel) -> dict:
    """Get a summary dict of the loaded model."""
    info = {
        "model_type": loaded.model_type,
        "checkpoint": loaded.checkpoint_path,
        "device": str(loaded.device),
    }

    n_params = sum(p.numel() for p in loaded.model.parameters())
    if n_params > 1e9:
        info["params"] = f"{n_params:,} (~{n_params/1e9:.2f}B)"
    else:
        info["params"] = f"{n_params:,} (~{n_params/1e6:.1f}M)"

    if loaded.model_type == "vqvae":
        cfg = loaded.vqvae_config
        info["image_size"] = cfg.image_size
        info["codebook_size"] = cfg.codebook_size
        info["codebook_dim"] = cfg.codebook_dim
        info["latent_grid"] = f"{cfg.latent_grid_size}×{cfg.latent_grid_size}"
        info["visual_tokens"] = cfg.num_visual_tokens

    elif loaded.model_type == "llm":
        cfg = loaded.model_config
        info["architecture"] = cfg.name
        info["context_length"] = cfg.context_length
        info["vocab_size"] = cfg.vocab_size
        info["emb_dim"] = cfg.emb_dim
        info["n_layers"] = cfg.n_layers

    elif loaded.model_type == "multimodal":
        mm = loaded.mm_config
        info["text_vocab"] = mm.text_vocab_size
        info["visual_vocab"] = mm.visual_vocab_size
        info["total_vocab"] = mm.total_vocab_size
        info["max_seq_length"] = mm.max_seq_length
        info["visual_tokens_per_image"] = mm.num_visual_tokens
        info["vqvae_loaded"] = loaded.frozen_vqvae is not None

    return info


def format_model_info(info: dict) -> str:
    """Format model info dict as a readable string."""
    lines = []
    type_labels = {
        "vqvae": "🖼️ VQ-VAE (Image Reconstruction)",
        "multimodal": "🎨 Multimodal (Text → Image)",
        "llm": "💬 LLM (Text Generation)",
    }
    lines.append(f"**Type:** {type_labels.get(info['model_type'], info['model_type'])}")
    lines.append(f"**Parameters:** {info['params']}")
    lines.append(f"**Device:** {info['device']}")
    lines.append(f"**Checkpoint:** `{Path(info['checkpoint']).name}`")

    # Type-specific info
    skip_keys = {"model_type", "params", "device", "checkpoint"}
    for k, v in info.items():
        if k in skip_keys:
            continue
        label = k.replace("_", " ").title()
        lines.append(f"**{label}:** {v}")

    return "\n".join(lines)
