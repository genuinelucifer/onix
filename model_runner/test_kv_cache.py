#!/usr/bin/env python3
"""
Unit test and benchmark for KV Cache and Quantization implementations.
Verifies:
1. Dynamic KV cache matches eager evaluation.
2. Static KV cache matches eager evaluation.
3. Static KV cache works with torch.compile.
4. Quantization (INT8 and INT4) loads and runs without errors.
"""

import time
import torch
from pathlib import Path
import sys

# Ensure project root is in path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from architecture.config import ModelConfig
from architecture.model import CausalLM
from architecture.generate import generate
from model_runner.runner import quantize_model

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running comprehensive performance test on: {device}")

    # 1. Create a small model config
    config = ModelConfig(
        name="test-tiny-llama",
        vocab_size=1000,
        context_length=128,
        emb_dim=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        pos_encoding="rope",
        norm_type="rmsnorm",
        ffn_type="swiglu",
        use_sdpa=True,
    )

    # 2. Instantiate the model and set to eval mode
    print("Initializing test model...")
    torch.manual_seed(42)
    model = CausalLM(config).to(device)
    model.eval()

    # Save initial state dict to reset model later
    import copy
    initial_state_dict = copy.deepcopy(model.state_dict())

    # 3. Create dummy input sequence (prompt)
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
    max_new_tokens = 30

    print(f"Prompt token IDs: {prompt_ids.tolist()[0]}")

    # 4. Generate WITHOUT KV cache (Eager mode)
    print("\n--- 1. Generating WITHOUT KV cache (Eager mode) ---")
    start_time = time.perf_counter()
    output_eager = generate(
        model=model,
        idx=prompt_ids.clone(),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=False,
    )
    eager_duration = time.perf_counter() - start_time
    eager_tokens = output_eager.shape[1] - prompt_ids.shape[1]
    eager_tps = eager_tokens / eager_duration
    print(f"Eager generated {eager_tokens} tokens in {eager_duration:.4f}s ({eager_tps:.2f} tokens/sec)")
    print(f"Eager output: {output_eager.tolist()[0]}")

    # 5. Generate WITH Dynamic KV cache
    print("\n--- 2. Generating WITH Dynamic KV cache ---")
    start_time = time.perf_counter()
    output_kv_dynamic = generate(
        model=model,
        idx=prompt_ids.clone(),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=True,
    )
    kv_dynamic_duration = time.perf_counter() - start_time
    kv_dynamic_tokens = output_kv_dynamic.shape[1] - prompt_ids.shape[1]
    kv_dynamic_tps = kv_dynamic_tokens / kv_dynamic_duration
    print(f"Dynamic KV generated {kv_dynamic_tokens} tokens in {kv_dynamic_duration:.4f}s ({kv_dynamic_tps:.2f} tokens/sec)")
    
    # Verify Dynamic
    if torch.equal(output_eager, output_kv_dynamic):
        print("✅ SUCCESS: Eager and Dynamic KV cache outputs are identical!")
    else:
        print("❌ FAILURE: Eager and Dynamic KV cache outputs differ!")
        sys.exit(1)

    # 6. Generate WITH Static KV cache
    print("\n--- 3. Generating WITH Static KV cache ---")
    # Set up static caches on the model
    model.setup_caches(max_batch_size=1, dtype=torch.float32)
    
    start_time = time.perf_counter()
    output_kv_static = generate(
        model=model,
        idx=prompt_ids.clone(),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=True,
    )
    kv_static_duration = time.perf_counter() - start_time
    kv_static_tokens = output_kv_static.shape[1] - prompt_ids.shape[1]
    kv_static_tps = kv_static_tokens / kv_static_duration
    print(f"Static KV generated {kv_static_tokens} tokens in {kv_static_duration:.4f}s ({kv_static_tps:.2f} tokens/sec)")

    # Verify Static
    if torch.equal(output_eager, output_kv_static):
        print("✅ SUCCESS: Eager and Static KV cache outputs are identical!")
    else:
        print("❌ FAILURE: Eager and Static KV cache outputs differ!")
        # Print differing tokens
        for idx in range(min(output_eager.shape[1], output_kv_static.shape[1])):
            if output_eager[0, idx] != output_kv_static[0, idx]:
                print(f"Diff at index {idx}: Eager={output_eager[0, idx].item()}, Static KV={output_kv_static[0, idx].item()}")
                break
        sys.exit(1)

    # 7. Generate WITH Compile + Static KV cache
    print("\n--- 4. Generating WITH Compile + Static KV cache ---")
    compiled_model = torch.compile(model, mode="reduce-overhead")
    # Warmup
    print("Warming up compiled model (compiling graph)...")
    _ = generate(
        model=compiled_model,
        idx=prompt_ids.clone(),
        max_new_tokens=5,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=True,
    )
    print("Warmup complete. Benchmarking...")
    
    start_time = time.perf_counter()
    output_kv_compiled = generate(
        model=compiled_model,
        idx=prompt_ids.clone(),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=True,
    )
    kv_compiled_duration = time.perf_counter() - start_time
    kv_compiled_tokens = output_kv_compiled.shape[1] - prompt_ids.shape[1]
    kv_compiled_tps = kv_compiled_tokens / kv_compiled_duration
    print(f"Compiled KV generated {kv_compiled_tokens} tokens in {kv_compiled_duration:.4f}s ({kv_compiled_tps:.2f} tokens/sec)")

    # Verify Compiled
    if torch.equal(output_eager, output_kv_compiled):
        print("✅ SUCCESS: Eager and Compiled KV cache outputs are identical!")
    else:
        print("❌ FAILURE: Eager and Compiled KV cache outputs differ!")
        sys.exit(1)

    # 8. Test Quantization (INT8)
    print("\n--- 5. Testing Weight-Only INT8 Quantization ---")
    # Reset model to float
    model = CausalLM(config).to(device)
    model.load_state_dict(initial_state_dict)
    model.eval()
    
    print("Quantizing to INT8...")
    quantize_model(model, "int8", target_dtype=torch.float32)
    model.setup_caches(max_batch_size=1, dtype=torch.float32)
    
    output_int8 = generate(
        model=model,
        idx=prompt_ids.clone(),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=True,
    )
    print(f"INT8 output: {output_int8.tolist()[0]}")
    print("✅ SUCCESS: INT8 model generated output successfully.")

    # 9. Test Quantization (INT4)
    print("\n--- 6. Testing Weight-Only INT4 Quantization ---")
    # Reset model to float
    model = CausalLM(config).to(device)
    model.load_state_dict(initial_state_dict)
    model.eval()
    
    print("Quantizing to INT4...")
    quantize_model(model, "int4", target_dtype=torch.float32)
    model.setup_caches(max_batch_size=1, dtype=torch.float32)
    
    output_int4 = generate(
        model=model,
        idx=prompt_ids.clone(),
        max_new_tokens=max_new_tokens,
        context_size=config.context_length,
        temperature=0.0,
        top_k=None,
        top_p=None,
        repetition_penalty=1.1,
        eos_id=999,
        use_kv_cache=True,
    )
    print(f"INT4 output: {output_int4.tolist()[0]}")
    print("✅ SUCCESS: INT4 model generated output successfully.")

    print(f"\nSummary of Speedups:")
    print(f"Eager TPS: {eager_tps:.2f}")
    print(f"Dynamic KV TPS: {kv_dynamic_tps:.2f} ({kv_dynamic_tps / eager_tps:.2f}x)")
    print(f"Static KV TPS: {kv_static_tps:.2f} ({kv_static_tps / eager_tps:.2f}x)")
    print(f"Compiled KV TPS: {kv_compiled_tps:.2f} ({kv_compiled_tps / eager_tps:.2f}x)")

if __name__ == "__main__":
    run_test()
