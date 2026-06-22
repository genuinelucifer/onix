#!/usr/bin/env python3
"""
AOTInductor — Ahead-of-Time compilation for Onix models.

Pre-compiles a model into a standalone shared library (.so) using
torch._inductor.aot_compile, eliminating the cold-start torch.compile
warmup (~30-60s on ROCm) on subsequent loads.

Workflow:
  1. Export:  python -m model_runner.aot_compiler export --checkpoint path/to/model
  2. Load:   In runner.py, use load_precompiled("model.so") instead of torch.compile()

The exported .so contains the fused, optimized GPU kernels for the decode step
(batch_size=1, seq_len=1) — the hot path during autoregressive generation.
Prefill still uses the eager model.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Optional

# Environment setup (must be before torch import)
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

import torch
import torch.nn as nn

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def export_model(
    model: nn.Module,
    output_path: str,
    device: str = "cuda",
    dtype: torch.dtype = torch.float16,
    context_size: int = 2048,
    compile_mode: str = "max-autotune",
) -> str:
    """
    Export a CausalLM model to an AOTInductor shared library.

    The exported model is optimized for the decode step (single token input).
    Prefill (variable-length input) should still use the eager model.

    Args:
        model: A CausalLM model (already loaded and on device)
        output_path: Path for the output .so file
        device: Target device
        dtype: Compute dtype
        context_size: Maximum context length for the static KV cache
        compile_mode: Inductor compilation mode

    Returns:
        Path to the exported .so file
    """
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    raw_model.eval()

    dev = torch.device(device)
    output_path = str(Path(output_path).resolve())

    # Ensure output directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Set up static KV caches if not already done
    if hasattr(raw_model, "setup_caches"):
        raw_model.setup_caches(max_batch_size=1, dtype=dtype, context_length=context_size)

    # Create example inputs for a single decode step
    # These define the static shapes that the compiled model will optimize for
    example_idx = torch.tensor([[1]], dtype=torch.long, device=dev)
    example_pos = torch.tensor([[0]], dtype=torch.long, device=dev)

    print(f"[AOT] Exporting model to: {output_path}")
    print(f"[AOT] Compile mode: {compile_mode}")
    print(f"[AOT] This may take several minutes on first run...")

    # Use torch.export + aot_compile for ahead-of-time compilation
    try:
        # Enable C++ wrapper for lower dispatch overhead
        torch._inductor.config.cpp_wrapper = True

        with torch.no_grad():
            # Run one forward pass to populate caches
            _ = raw_model(example_idx, position_ids=example_pos, use_cache=True)

            # Export using torch.export (captures the computation graph)
            exported = torch.export.export(
                raw_model,
                (example_idx,),
                kwargs={"position_ids": example_pos, "use_cache": True},
            )

            # AOT compile to shared library
            so_path = torch._inductor.aot_compile(
                exported.module(),
                (example_idx,),
                kwargs={"position_ids": example_pos, "use_cache": True},
                options={"aot_inductor.output_path": output_path},
            )

        print(f"[AOT] Export complete: {so_path}")
        return so_path

    except Exception as e:
        print(f"[AOT] Export failed: {e}")
        print("[AOT] Falling back to standard torch.compile at runtime.")
        raise


def load_precompiled(so_path: str) -> Optional[callable]:
    """
    Load a pre-compiled AOTInductor model from a .so file.

    Args:
        so_path: Path to the .so file from export_model()

    Returns:
        A callable that takes (idx, position_ids, use_cache) and returns logits,
        or None if loading fails.
    """
    so_path = str(Path(so_path).resolve())

    if not Path(so_path).exists():
        print(f"[AOT] Pre-compiled model not found: {so_path}")
        return None

    try:
        compiled_fn = torch._inductor.aot_load(so_path)
        print(f"[AOT] Loaded pre-compiled model from: {so_path}")
        return compiled_fn
    except Exception as e:
        print(f"[AOT] Failed to load pre-compiled model: {e}")
        return None


def find_precompiled(checkpoint_path: str) -> Optional[str]:
    """
    Look for a pre-compiled .so file alongside a checkpoint.

    Convention: the .so file is named "<checkpoint_dir>/compiled_decode.so"

    Returns the path if found, else None.
    """
    ckpt = Path(checkpoint_path)
    if ckpt.is_file():
        model_dir = ckpt.parent
    else:
        model_dir = ckpt

    so_path = model_dir / "compiled_decode.so"
    if so_path.exists():
        return str(so_path)
    return None


# ===========================================================================
#  CLI interface
# ===========================================================================

def main():
    """CLI for ahead-of-time model compilation."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Onix AOT Compiler — Pre-compile models for instant loading"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Export subcommand
    export_parser = subparsers.add_parser("export", help="Export model to .so")
    export_parser.add_argument(
        "--checkpoint", "-c", required=True,
        help="Path to checkpoint directory or file"
    )
    export_parser.add_argument(
        "--output", "-o", default=None,
        help="Output path for .so file (default: <checkpoint_dir>/compiled_decode.so)"
    )
    export_parser.add_argument(
        "--dtype", type=str, default="float16",
        choices=["float16", "bfloat16"],
        help="Compute dtype"
    )
    export_parser.add_argument(
        "--compile-mode", type=str, default="max-autotune",
        choices=["default", "reduce-overhead", "max-autotune"],
    )
    export_parser.add_argument(
        "--context-size", type=int, default=2048,
        help="Maximum context length"
    )
    export_parser.add_argument(
        "--device", type=str, default="cuda",
    )

    args = parser.parse_args()

    if args.command == "export":
        from model_runner.runner import load_model

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        dtype = dtype_map[args.dtype]

        # Load the model
        loaded = load_model(
            checkpoint_path=args.checkpoint,
            device=args.device,
            dtype=dtype,
            compile=False,
            context_size=args.context_size,
        )

        # Determine output path
        output = args.output
        if output is None:
            ckpt = Path(args.checkpoint)
            model_dir = ckpt.parent if ckpt.is_file() else ckpt
            output = str(model_dir / "compiled_decode.so")

        # Export
        export_model(
            model=loaded.model,
            output_path=output,
            device=args.device,
            dtype=dtype,
            context_size=args.context_size,
            compile_mode=args.compile_mode,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
