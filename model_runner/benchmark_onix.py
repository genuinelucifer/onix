#!/usr/bin/env python3
"""
Benchmark script to compare text generation performance (Tokens/sec and TTFT)
across Eager, Static KV cache, Compiled, and Quantized modes.

Optimizations included:
  - TunableOp for optimal GEMM kernel selection on AMD GPUs
  - Fused RMSNorm (F.rms_norm) for reduced memory bandwidth
  - Vectorized repetition penalty (no GPU→CPU sync)
  - GPU-side EOS check (no .item() sync)
  - Dropout(0.0) removal for cleaner compiled graphs
  - Max-autotune GEMM for aggressive kernel tuning
"""

import os
import sys
import time
import argparse
from pathlib import Path

# ---- Environment-level optimizations (must be set before importing torch) ----

# Flash Attention on AMD consumer GPUs
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# TunableOp: optimal GEMM kernel selection for AMD GPUs
# Enabled by default (uses cached tunings if available, no overhead if no cache exists)
if "PYTORCH_TUNABLEOP_ENABLED" not in os.environ:
    os.environ["PYTORCH_TUNABLEOP_ENABLED"] = "1"

# Aggressive inductor GEMM autotuning during compilation
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE_GEMM"] = "1"

import torch  # noqa: E402

# Ensure project root is in path
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent if _SCRIPT_DIR.name == "model_runner" else _SCRIPT_DIR
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from model_runner.runner import load_model  # noqa: E402
from architecture.generate import generate  # noqa: E402
from model_runner.runner import text_to_token_ids, token_ids_to_text  # noqa: E402


def benchmark():
    parser = argparse.ArgumentParser(description="Onix Performance Benchmark Suite")
    parser.add_argument(
        "--checkpoint-path", "-c",
        type=str,
        default=str(_PROJECT_ROOT / "models/llama1b-sft-natural"),
        help="Path to checkpoint directory or file"
    )
    parser.add_argument(
        "--prompt", "-p",
        type=str,
        default="Tell me a story about a boy.",
        help="Prompt to run benchmark with"
    )
    parser.add_argument(
        "--max-new-tokens", "-n",
        type=int,
        default=50,
        help="Number of new tokens to generate"
    )
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default="all",
        choices=[
            "eager", "static", "compiled",
            "bf16", "compiled_bf16",
            "int8", "compiled_int8",
            "int4", "compiled_int4",
            "all"
        ],
        help="Benchmark mode to run (default: all)"
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="PyTorch compilation mode (default: default). "
             "NOTE: 'reduce-overhead' uses HIP graphs but requires static tensor shapes; "
             "the current KV cache uses torch.narrow with growing seq_len, so "
             "'reduce-overhead' will record a new graph per decode step (slower)."
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Enable TunableOp GEMM tuning phase (run once to generate tuned kernels, "
             "then re-run without --tune for optimized performance)"
    )
    parser.add_argument(
        "--tune-file",
        type=str,
        default="onix_tuned_kernels.csv",
        help="Path to TunableOp tuning results CSV file"
    )
    parser.add_argument(
        "--diagnose", "-d",
        action="store_true",
        help="Run torch._dynamo.explain to diagnose compilation and check for graph breaks"
    )
    parser.add_argument(
        "--context-size",
        type=int,
        default=None,
        help="Override model's maximum context length during benchmark (default: use config value)"
    )
    args = parser.parse_args()

    # Resolve paths relative to project root
    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_absolute():
        checkpoint_path = _PROJECT_ROOT / checkpoint_path
    checkpoint_path = str(checkpoint_path)

    tune_file_base = Path(args.tune_file)
    if not tune_file_base.is_absolute():
        tune_file_base = _PROJECT_ROOT / tune_file_base

    # Configure TunableOp tuning
    if args.tune:
        os.environ["PYTORCH_TUNABLEOP_TUNING"] = "1"
        print("[TunableOp] TUNING ENABLED — benchmarking GEMM kernels (first run is slower)")
    else:
        os.environ["PYTORCH_TUNABLEOP_TUNING"] = "0"
    os.environ["PYTORCH_TUNABLEOP_FILENAME"] = str(tune_file_base)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    prompt = args.prompt
    max_new_tokens = args.max_new_tokens

    # Detect if the tune file exists (handling ROCm's automatic device index suffix)
    device_id = torch.cuda.current_device() if torch.cuda.is_available() else 0
    device_tune_file = tune_file_base.parent / f"{tune_file_base.stem}{device_id}{tune_file_base.suffix}"

    tune_file_exists = tune_file_base.exists() or device_tune_file.exists()
    detected_filename = device_tune_file if device_tune_file.exists() else tune_file_base

    print("=" * 60)
    print("           ONIX PERFORMANCE BENCHMARK SUITE           ")
    print("=" * 60)
    print(f"Model path:    {checkpoint_path}")
    print(f"Device:        {device}")
    print(f"Prompt:        \"{prompt}\"")
    print(f"Generating:    {max_new_tokens} tokens")
    print(f"Compile mode:  {args.compile_mode}")
    print(f"TunableOp:     {'TUNING' if args.tune else 'enabled (using cached)'}")
    if tune_file_exists:
        print(f"Tune file:     {detected_filename.name} (found)")
    else:
        print(f"Tune file:     {args.tune_file} (not found — run with --tune first)")
    print("-" * 60)

    # 1. Run Diagnose check if requested
    if args.diagnose:
        print("\n=== RUNNING COMPILER DIAGNOSTIC CHECK ===")
        print("Loading model for diagnostics...")
        loaded = load_model(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=torch.float16,
            compile=False
        )
        model = loaded.model
        tokenizer = loaded.tokenizer

        if hasattr(model, "setup_caches"):
            model.setup_caches(max_batch_size=1, dtype=torch.float16, context_length=args.context_size)

        print("Compiling model for diagnostics...")
        compiled_model = torch.compile(model, mode=args.compile_mode)

        # Prepare inputs for a single decode step
        input_ids = text_to_token_ids(prompt, tokenizer).to(device)
        position_ids = torch.tensor([[input_ids.shape[1]]], device=device)
        idx_next = torch.tensor([[1]], device=device)

        print("\n--- torch._dynamo.explain results for decode step ---")
        try:
            explanation = torch._dynamo.explain(compiled_model, idx_next, position_ids=position_ids, use_cache=True)
            print(explanation)
        except Exception as diag_err:
            print(f"Failed to explain compilation: {diag_err}")
        print("=" * 60)
        return

    # Helper to run a benchmark
    def run_benchmark(name, dtype_arg, compile_arg, use_kv_cache):
        print(f"\n[Benchmarking Mode: {name}]")
        print("Loading model...")
        t_load_start = time.perf_counter()

        loaded = load_model(
            checkpoint_path=checkpoint_path,
            device=device,
            dtype=dtype_arg,
            compile=compile_arg,
            compile_mode=args.compile_mode,
            context_size=args.context_size
        )
        t_load = time.perf_counter() - t_load_start
        print(f"Model loaded in {t_load:.2f}s")

        tokenizer = loaded.tokenizer
        model = loaded.model
        ctx_size = getattr(model, "active_context_length", loaded.model_config.context_length)

        # Tokenize prompt
        input_ids = text_to_token_ids(prompt, tokenizer).to(device)

        # Warmup if compiled
        if compile_arg:
            print("Warming up compiled graph (compiling)... This takes ~30-60 seconds on ROCm...")
            t_comp_start = time.perf_counter()
            _ = generate(
                model=model,
                idx=input_ids.clone(),
                max_new_tokens=5,
                context_size=ctx_size,
                temperature=0.0,
                use_kv_cache=use_kv_cache
            )
            t_comp = time.perf_counter() - t_comp_start
            print(f"Compilation finished in {t_comp:.2f}s")

        # Actual generation benchmark
        metrics = {}
        t_gen_start = time.perf_counter()
        output_ids = generate(
            model=model,
            idx=input_ids.clone(),
            max_new_tokens=max_new_tokens,
            context_size=ctx_size,
            temperature=0.0,
            use_kv_cache=use_kv_cache,
            metrics=metrics
        )
        t_gen = time.perf_counter() - t_gen_start

        new_tokens = output_ids[0, input_ids.shape[1]:]
        generated_text = token_ids_to_text(new_tokens.unsqueeze(0), tokenizer)

        ttft = metrics.get("ttft", 0.0)
        decode_time = metrics.get("decode_time", t_gen - ttft)
        tps = len(new_tokens) / decode_time if decode_time > 0 else 0.0

        fw_time = metrics.get("forward_time", 0.0)
        sm_time = metrics.get("sample_time", 0.0)
        ov_time = metrics.get("overhead_time", 0.0)

        pct_fw = (fw_time / decode_time * 100) if decode_time > 0 else 0.0
        pct_sm = (sm_time / decode_time * 100) if decode_time > 0 else 0.0
        pct_ov = (ov_time / decode_time * 100) if decode_time > 0 else 0.0

        print(f"Generated text: \"{generated_text}\"")
        print(f"Results for {name}:")
        print(f"  - TTFT:             {ttft * 1000:.1f} ms")
        print(f"  - Decode speed:     {tps:.2f} tokens/sec")
        print(f"  - Total decode time: {decode_time:.4f}s")
        print(f"    * Forward pass:   {fw_time:.4f}s ({pct_fw:.1f}%)")
        print(f"    * Sampling/Rules: {sm_time:.4f}s ({pct_sm:.1f}%)")
        print(f"    * Loop overhead:  {ov_time:.4f}s ({pct_ov:.1f}%)")
        print(f"  - Total time:       {t_gen:.2f}s")

        # Cleanup
        del model
        del loaded
        torch.cuda.empty_cache()

        return tps, ttft * 1000

    results = {}

    # Map selected mode to configurations
    all_modes = {
        "eager": ("Eager (No KV Cache)", torch.float16, False, False),
        "static": ("Static KV Cache", torch.float16, False, True),
        "compiled": ("Compiled Static KV Cache", torch.float16, True, True),
        "bf16": ("Static KV Cache (BF16)", torch.bfloat16, False, True),
        "compiled_bf16": ("Compiled Static KV Cache (BF16)", torch.bfloat16, True, True),
        "int8": ("Weight-Only INT8", "int8", False, True),
        "compiled_int8": ("Compiled Weight-Only INT8", "int8", True, True),
        "int4": ("Weight-Only INT4", "int4", False, True),
        "compiled_int4": ("Compiled Weight-Only INT4", "int4", True, True),
    }

    try:
        if args.mode == "all":
            modes_to_run = list(all_modes.keys())
        else:
            modes_to_run = [args.mode]

        for m_key in modes_to_run:
            name, dtype_arg, compile_arg, use_kv_cache = all_modes[m_key]
            results[name] = run_benchmark(name, dtype_arg, compile_arg, use_kv_cache)

        # Print final comparison table
        print("\n" + "=" * 60)
        print("                  SUMMARY OF RESULTS                  ")
        print("=" * 60)
        print(f"{'Execution Mode':<30} | {'Decode TPS':<12} | {'TTFT':<10}")
        print("-" * 60)
        for mode, (tps, ttft) in results.items():
            print(f"{mode:<30} | {tps:<12.2f} | {ttft:.1f} ms")
        print("=" * 60)

        if args.tune:
            print(f"\n[TunableOp] Tuning complete. Results saved to: {args.tune_file}")
            print("[TunableOp] Re-run WITHOUT --tune for optimized performance.")

    except Exception as e:
        print(f"Error during benchmark: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    benchmark()
