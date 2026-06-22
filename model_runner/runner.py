#!/usr/bin/env python3
"""
Model Runner — load trained Onix checkpoints and run inference.

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

# Enable TunableOp for optimal GEMM kernel selection on AMD GPUs.
# Set PYTORCH_TUNABLEOP_TUNING=1 on first run to generate tuned kernels,
# then set to 0 for production use with the generated CSV.
if "PYTORCH_TUNABLEOP_ENABLED" not in os.environ:
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"
if "PYTORCH_TUNABLEOP_TUNING" not in os.environ:
    os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"

# Enable C++ wrapper mode for Inductor — reduces Python dispatch overhead
# in the compiled decode loop by generating C++ kernel dispatch code.
try:
    import torch
    # For ROCm/HIP builds, PyTorch requires CUDA_HOME to point to the ROCm SDK root
    # in order to locate HIP headers for compilation.
    if hasattr(torch.version, "hip") and torch.version.hip:
        if "CUDA_HOME" not in os.environ:
            rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
            if os.path.isdir(rocm_home):
                os.environ["CUDA_HOME"] = rocm_home

    # Only enable cpp_wrapper if CUDA_HOME is available in ROCm environments,
    # preventing compilation errors if the ROCm SDK is not installed.
    if not (hasattr(torch.version, "hip") and torch.version.hip and "CUDA_HOME" not in os.environ):
        import torch._inductor.config
        torch._inductor.config.cpp_wrapper = True
except (ImportError, AttributeError):
    pass

import torch
import torch.nn as nn
import torch.nn.functional as F
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
from model import get_tokenizer, text_to_token_ids, token_ids_to_text, EOT_TOKEN_ID


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
    use_kv_cache: bool = True


# ---------------------------------------------------------------------------
#  In-memory Weight-Only Quantization (INT8 & INT4)
# ---------------------------------------------------------------------------

class WeightOnlyInt8Linear(nn.Module):
    """In-memory Weight-only 8-bit Linear Layer (W8A16/W8A32)"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("qweight", torch.zeros((out_features, in_features), dtype=torch.int8, device=device))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, dtype=torch.float16):
        device = float_linear.weight.device
        w = float_linear.weight.detach()
        
        # Calculate channel-wise scales
        max_val = torch.max(torch.abs(w), dim=1, keepdim=True).values.clamp(min=1e-5)
        scale = max_val / 127.0
        
        # Quantize to int8
        qweight = torch.clamp(torch.round(w / scale), -128, 127).to(torch.int8)
        
        qlinear = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=(float_linear.bias is not None),
            device=device,
            dtype=dtype
        )
        qlinear.qweight.copy_(qweight)
        qlinear.scale.copy_(scale.to(dtype))
        if float_linear.bias is not None:
            qlinear.bias.data.copy_(float_linear.bias.data.to(dtype))
        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dequantized_weight = self.qweight.to(x.dtype) * self.scale
        return F.linear(x, dequantized_weight, self.bias)


class WeightOnlyInt4Linear(nn.Module):
    """In-memory Packed Weight-only 4-bit Linear Layer (W4A16/W4A32)"""
    def __init__(self, in_features: int, out_features: int, bias: bool = True, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        assert in_features % 2 == 0, "in_features must be even for 4-bit packing"
        self.register_buffer("qweight_packed", torch.zeros((out_features, in_features // 2), dtype=torch.uint8, device=device))
        self.register_buffer("scale", torch.zeros((out_features, 1), dtype=dtype, device=device))
        self.register_buffer("zero_point", torch.zeros((out_features, 1), dtype=dtype, device=device))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=dtype, device=device))
        else:
            self.register_parameter("bias", None)

    @classmethod
    def from_float(cls, float_linear: nn.Linear, dtype=torch.float16):
        device = float_linear.weight.device
        w = float_linear.weight.detach()
        
        # Calculate channel-wise scale and zero_point (range [0, 15])
        min_val = torch.min(w, dim=1, keepdim=True).values
        max_val = torch.max(w, dim=1, keepdim=True).values
        scale = (max_val - min_val).clamp(min=1e-5) / 15.0
        zero_point = min_val
        
        # Quantize to [0, 15]
        qweight = torch.clamp(torch.round((w - zero_point) / scale), 0, 15).to(torch.uint8)
        
        # Pack even and odd columns along in_features (dim 1)
        w_even = qweight[:, 0::2]
        w_odd = qweight[:, 1::2]
        qweight_packed = w_even | (w_odd << 4)
        
        qlinear = cls(
            in_features=float_linear.in_features,
            out_features=float_linear.out_features,
            bias=(float_linear.bias is not None),
            device=device,
            dtype=dtype
        )
        qlinear.qweight_packed.copy_(qweight_packed)
        qlinear.scale.copy_(scale.to(dtype))
        qlinear.zero_point.copy_(zero_point.to(dtype))
        if float_linear.bias is not None:
            qlinear.bias.data.copy_(float_linear.bias.data.to(dtype))
        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        packed = self.qweight_packed
        
        # Extract even and odd components
        w_even = packed & 0x0F
        w_odd = (packed >> 4) & 0x0F
        
        # Interleave columns
        unpacked = torch.stack([w_even, w_odd], dim=2).view(self.out_features, self.in_features)
        
        # Dequantize
        dequantized_weight = (unpacked.to(x.dtype) * self.scale) + self.zero_point
        return F.linear(x, dequantized_weight, self.bias)


def quantize_model(model: nn.Module, mode: str, target_dtype=torch.float16, tokenizer=None, device=None):
    """Recursively replace Linear layers (except lm_head) with quantized versions."""
    if mode == "awq_int4":
        # AWQ: Activation-aware Weight Quantization
        from model_runner.awq import calibrate_model, awq_quantize_model
        if tokenizer is None or device is None:
            raise ValueError("AWQ quantization requires tokenizer and device for calibration")
        print("[AWQ] Running calibration (collecting activation statistics)...")
        activation_stats = calibrate_model(model, tokenizer, device)
        print(f"[AWQ] Calibrated {len(activation_stats)} layers. Applying AWQ INT4 quantization...")
        awq_quantize_model(model, activation_stats, target_dtype=target_dtype)
        print("[AWQ] Quantization complete.")
        return

    for name, child in model.named_children():
        if isinstance(child, nn.Linear):
            if name == "lm_head":
                continue
            if mode == "int4" and child.in_features % 2 != 0:
                continue
                
            if mode == "int8":
                qlinear = WeightOnlyInt8Linear.from_float(child, dtype=target_dtype)
            elif mode == "int4":
                qlinear = WeightOnlyInt4Linear.from_float(child, dtype=target_dtype)
            else:
                continue
            setattr(model, name, qlinear)
        else:
            quantize_model(child, mode, target_dtype)


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


def _remove_dropout(model: nn.Module):
    """Replace all Dropout(p=0.0) with Identity for cleaner compiled graphs."""
    for name, child in model.named_children():
        if isinstance(child, nn.Dropout) and child.p == 0.0:
            setattr(model, name, nn.Identity())
        else:
            _remove_dropout(child)


def load_model(
    checkpoint_path: str,
    device: str = "cuda",
    config_path: Optional[str] = None,
    dtype: Optional[torch.dtype] = None,
    compile: bool = False,
    compile_mode: str = "default",
    context_size: Optional[int] = None,
    kv_quant_mode: str = "none",
    precompiled_path: Optional[str] = None,
    medusa_heads_path: Optional[str] = None,
) -> LoadedModel:
    """
    Load a model from a checkpoint file or model directory.

    Auto-detects model type from config.json in the same directory.

    Args:
        checkpoint_path: Path to a .pt file OR a model directory
        device: Device to load onto ("cuda", "cpu", etc.)
        config_path: Optional manual path to config.json
        dtype: Optional torch.dtype to cast model weights to
        compile: Whether to run torch.compile on the model
        compile_mode: torch.compile mode ("default", "reduce-overhead", "max-autotune")
        context_size: Override model context length
        kv_quant_mode: KV cache quantization ("none", "fp8", "turboquant", "kivi")
        precompiled_path: Path to AOTInductor .so file (skips torch.compile)
        medusa_heads_path: Path to trained Medusa heads checkpoint

    Returns:
        LoadedModel with the model ready for inference
    """
    ckpt_path = _resolve_checkpoint(checkpoint_path)

    dev = torch.device(device)

    # Load config
    if config_path and config_path.strip():
        actual_config_path = Path(config_path.strip())
    else:
        actual_config_path = find_config_json(str(ckpt_path))
        
    with open(actual_config_path) as f:
        full_config = json.load(f)

    model_type = detect_model_type(full_config)
    tokenizer = get_tokenizer()

    if model_type == "vqvae":
        return _load_vqvae(ckpt_path, full_config, dev, tokenizer, dtype, compile, compile_mode)
    elif model_type == "multimodal":
        return _load_multimodal(
            ckpt_path, full_config, dev, tokenizer, dtype, compile, compile_mode, context_size,
            kv_quant_mode=kv_quant_mode,
        )
    elif model_type == "llm":
        return _load_llm(
            ckpt_path, full_config, dev, tokenizer, dtype, compile, compile_mode, context_size,
            kv_quant_mode=kv_quant_mode,
            precompiled_path=precompiled_path,
            medusa_heads_path=medusa_heads_path,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _strip_orig_mod_prefix(state_dict: dict) -> dict:
    """Strip 'torch.compile' prefix (_orig_mod.) from state_dict keys if present."""
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        return {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    return state_dict



def _load_vqvae(
    ckpt_path: Path, config: dict, device: torch.device, tokenizer,
    dtype: Optional[torch.dtype] = None, compile: bool = False,
    compile_mode: str = "default",
) -> LoadedModel:
    """Load a VQ-VAE model."""
    vqvae_config = VQVAEConfig.from_dict(config["vqvae"])
    with torch.device(device):
        model = VQVAE(vqvae_config)

    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = _strip_orig_mod_prefix(state_dict)
    del ckpt

    model.load_state_dict(state_dict, strict=False)

    if dtype is not None and not isinstance(dtype, str):
        model = model.to(dtype)

    if compile:
        model = torch.compile(model, mode=compile_mode)

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
    ckpt_path: Path, config: dict, device: torch.device, tokenizer,
    dtype: Optional[torch.dtype] = None, compile: bool = False,
    compile_mode: str = "default",
    context_size: Optional[int] = None,
    kv_quant_mode: str = "none",
) -> LoadedModel:
    """Load a multimodal model (transformer + frozen VQ-VAE)."""
    mm_config = MultiModalConfig.from_dict(config["multimodal"])

    # Build transformer with correct vocab/context
    transformer_config = mm_config.build_transformer_config()
    with torch.device(device):
        model = CausalLM(transformer_config)

    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = _strip_orig_mod_prefix(state_dict)
    del ckpt

    model.load_state_dict(state_dict, strict=False)

    # Load frozen VQ-VAE for decoding
    vqvae_ckpt_path = mm_config.vqvae_checkpoint
    # Resolve relative paths from project root
    if not Path(vqvae_ckpt_path).is_absolute():
        vqvae_ckpt_path = str(_PROJECT_ROOT / vqvae_ckpt_path)

    frozen_vqvae = None
    if Path(vqvae_ckpt_path).exists():
        with torch.device(device):
            frozen_vqvae = VQVAE(mm_config.vqvae)
        vq_ckpt = torch.load(vqvae_ckpt_path, map_location="cpu", mmap=True, weights_only=True)
        vq_state_dict = vq_ckpt["model_state_dict"] if "model_state_dict" in vq_ckpt else vq_ckpt
        vq_state_dict = _strip_orig_mod_prefix(vq_state_dict)
        del vq_ckpt

        frozen_vqvae.load_state_dict(vq_state_dict, strict=False)

    quant_mode = None
    target_dtype = None
    if isinstance(dtype, str) and dtype in ("int8", "int4", "awq_int4"):
        quant_mode = dtype
        target_dtype = torch.float16
    elif dtype is not None:
        target_dtype = dtype

    if target_dtype is not None:
        model = model.to(target_dtype)
        if frozen_vqvae is not None:
            frozen_vqvae = frozen_vqvae.to(target_dtype)

    if quant_mode is not None:
        quantize_model(model, quant_mode, target_dtype=target_dtype, tokenizer=tokenizer, device=device)

    # Set up static KV cache with optional quantization
    cache_dtype = target_dtype if target_dtype is not None else torch.float16
    if hasattr(model, "setup_caches"):
        model.setup_caches(
            max_batch_size=1, dtype=cache_dtype, context_length=context_size,
            kv_quant_mode=kv_quant_mode,
        )

    if compile:
        _remove_dropout(model)
        if frozen_vqvae is not None:
            _remove_dropout(frozen_vqvae)
        model = torch.compile(model, mode=compile_mode)

    model.eval()
    if frozen_vqvae is not None:
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
    ckpt_path: Path, config: dict, device: torch.device, tokenizer,
    dtype: Optional[torch.dtype] = None, compile: bool = False,
    compile_mode: str = "default",
    context_size: Optional[int] = None,
    kv_quant_mode: str = "none",
    precompiled_path: Optional[str] = None,
    medusa_heads_path: Optional[str] = None,
) -> LoadedModel:
    """Load an LLM (CausalLM) model."""
    model_config = ModelConfig.from_dict(config["architecture"])

    # Init on target device: params + buffers (RoPE, causal mask) all on CUDA.
    # No CPU RAM used — only VRAM.
    with torch.device(device):
        model = CausalLM(model_config)

    # Load checkpoint lazily: map_location="cpu" + mmap=True keeps tensors
    # as lazy memory-mapped references — near-zero CPU RAM until accessed.
    ckpt = torch.load(ckpt_path, map_location="cpu", mmap=True, weights_only=True)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_dict = _strip_orig_mod_prefix(state_dict)
    del ckpt

    model.load_state_dict(state_dict, strict=False)

    quant_mode = None
    target_dtype = None
    if isinstance(dtype, str) and dtype in ("int8", "int4", "awq_int4"):
        quant_mode = dtype
        target_dtype = torch.float16
    elif dtype is not None:
        target_dtype = dtype

    if target_dtype is not None:
        model = model.to(target_dtype)

    if quant_mode is not None:
        quantize_model(model, quant_mode, target_dtype=target_dtype, tokenizer=tokenizer, device=device)

    # Set up static KV cache with optional quantization
    cache_dtype = target_dtype if target_dtype is not None else torch.float16
    if hasattr(model, "setup_caches"):
        model.setup_caches(
            max_batch_size=1, dtype=cache_dtype, context_length=context_size,
            kv_quant_mode=kv_quant_mode,
        )

    # Compilation: prefer AOTInductor pre-compiled model if available
    if precompiled_path:
        from model_runner.aot_compiler import load_precompiled
        precompiled = load_precompiled(precompiled_path)
        if precompiled is not None:
            # Store the precompiled function as an attribute for the decode loop
            model._precompiled_decode = precompiled
            compile = False  # Skip torch.compile since we have AOT

    if compile:
        _remove_dropout(model)
        model = torch.compile(model, mode=compile_mode)

    model.eval()

    # Wrap with Medusa heads if provided
    if medusa_heads_path:
        from architecture.medusa import MedusaModel
        medusa_model = MedusaModel(model)
        medusa_model.load_heads(medusa_heads_path, map_location=str(device))
        medusa_model.to(device)
        medusa_model.eval()
        # Store medusa model reference but keep base model as the primary
        model._medusa_model = medusa_model

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
) -> dict:
    """
    Generate text continuation from the conversation so far.

    Args:
        loaded: Loaded LLM model
        conversation_text: Full conversation text to continue from
        params: Generation parameters

    Returns:
        A dict containing:
          - "text": generated text
          - "tokens": number of tokens generated
          - "time": duration of generation in seconds
    """
    assert loaded.model_type == "llm"

    tokenizer = loaded.tokenizer
    model = loaded.model
    ctx_size = getattr(model, "active_context_length", loaded.model_config.context_length)

    # Tokenize input
    input_ids = text_to_token_ids(conversation_text, tokenizer).to(loaded.device)

    # Truncate to fit context window (keep recent tokens)
    if input_ids.shape[1] > ctx_size - params.max_new_tokens:
        keep = ctx_size - params.max_new_tokens
        input_ids = input_ids[:, -keep:]

    input_len = input_ids.shape[1]

    # Generate
    metrics = {}
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
            eos_id=params.eos_id if params.eos_id is not None else EOT_TOKEN_ID,
            use_kv_cache=params.use_kv_cache,
            metrics=metrics,
        )

    # Extract only the new tokens
    new_tokens = output_ids[:, input_len:]
    generated_text = token_ids_to_text(new_tokens, tokenizer)

    return {
        "text": generated_text,
        "tokens": new_tokens.shape[1],
        "ttft": metrics.get("ttft", 0.0),
        "decode_time": metrics.get("decode_time", 0.0),
    }


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
    """Format model info dict as an HTML snippet with 3 columns."""
    lines = []
    type_labels = {
        "vqvae": "🖼️ VQ-VAE (Image Reconstruction)",
        "multimodal": "🎨 Multimodal (Text → Image)",
        "llm": "💬 LLM (Text Generation)",
    }
    
    html = ["<div style='display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px;'>"]
    
    html.append(f"<div><strong>Type:</strong><br/>{type_labels.get(info['model_type'], info['model_type'])}</div>")
    html.append(f"<div><strong>Parameters:</strong><br/>{info['params']}</div>")
    html.append(f"<div><strong>Device:</strong><br/>{info['device']}</div>")
    html.append(f"<div><strong>Checkpoint:</strong><br/><code>{Path(info['checkpoint']).name}</code></div>")

    # Type-specific info
    skip_keys = {"model_type", "params", "device", "checkpoint"}
    for k, v in info.items():
        if k in skip_keys:
            continue
        label = k.replace("_", " ").title()
        html.append(f"<div><strong>{label}:</strong><br/>{v}</div>")

    html.append("</div>")
    return "\n".join(html)
