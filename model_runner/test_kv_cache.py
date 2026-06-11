#!/usr/bin/env python3
"""
Unit test and benchmark for KV Cache implementation.
Verifies that using KV Cache produces the exact same generated token IDs
as eager evaluation, and measures the speedup.
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

def run_test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running KV cache test on: {device}")

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
    model = CausalLM(config).to(device)
    model.eval()

    # 3. Create dummy input sequence (prompt)
    # Batch size 1, sequence length 10
    torch.manual_seed(42)
    prompt_ids = torch.randint(0, config.vocab_size, (1, 10), device=device)
    max_new_tokens = 30

    print(f"Prompt token IDs: {prompt_ids.tolist()[0]}")

    # 4. Generate WITHOUT KV cache (Eager mode)
    print("\n--- Generating WITHOUT KV cache (Eager mode) ---")
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
        eos_id=999,  # Some arbitrary EOS
        use_kv_cache=False,  # Should fall back to old behavior
    )
    eager_duration = time.perf_counter() - start_time
    eager_tokens = output_eager.shape[1] - prompt_ids.shape[1]
    eager_tps = eager_tokens / eager_duration
    print(f"Eager generated {eager_tokens} tokens in {eager_duration:.4f}s ({eager_tps:.2f} tokens/sec)")
    print(f"Eager output: {output_eager.tolist()[0]}")

    # 5. Generate WITH KV cache
    print("\n--- Generating WITH KV cache ---")
    start_time = time.perf_counter()
    output_kv = generate(
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
    kv_duration = time.perf_counter() - start_time
    kv_tokens = output_kv.shape[1] - prompt_ids.shape[1]
    kv_tps = kv_tokens / kv_duration
    print(f"KV cache generated {kv_tokens} tokens in {kv_duration:.4f}s ({kv_tps:.2f} tokens/sec)")
    print(f"KV cache output: {output_kv.tolist()[0]}")

    # 6. Verification
    print("\n--- Verification ---")
    match = torch.equal(output_eager, output_kv)
    if match:
        print("✅ SUCCESS: Eager and KV cache outputs are identical!")
    else:
        print("❌ FAILURE: Eager and KV cache outputs differ!")
        
        # Print first differing index
        min_len = min(output_eager.shape[1], output_kv.shape[1])
        for idx in range(min_len):
            if output_eager[0, idx] != output_kv[0, idx]:
                print(f"First diff at token index {idx}: Eager={output_eager[0, idx].item()}, KV={output_kv[0, idx].item()}")
                break
        sys.exit(1)

    print(f"Performance Speedup: {kv_tps / eager_tps:.2f}x")

if __name__ == "__main__":
    run_test()
