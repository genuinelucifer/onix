#!/usr/bin/env python3
import argparse
import json
import numpy as np
import array
import sys
from pathlib import Path

# Add parent dir to path so we can import model.py
sys.path.append(str(Path(__file__).parent.parent))
from model import get_tokenizer

def format_input_alpaca(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text

def main():
    parser = argparse.ArgumentParser(description="Pre-tokenize SFT JSONL to binary shards")
    parser.add_argument("--input", type=str, required=True, help="Path to input JSONL file")
    parser.add_argument("--output", type=str, required=True, help="Prefix for output .npy files")
    args = parser.parse_args()

    tokenizer = get_tokenizer()
    tokens_out = array.array('i')
    bounds_out = array.array('i')
    curr = 0
    count = 0
    
    print(f"Tokenizing {args.input}...")
    with open(args.input, "r") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            
            entry = json.loads(line)
            full_text = format_input_alpaca(entry) + f"\n\n### Response:\n{entry['output']}"
            
            tokens = tokenizer.encode(full_text)
            tokens_out.extend(tokens)
            
            # Store [start, end]
            bounds_out.append(curr)
            bounds_out.append(curr + len(tokens))
            
            curr += len(tokens)
            count += 1
            
            if count % 100000 == 0:
                print(f"  Processed {count} entries...")

    print(f"Saving to {args.output}_tokens.npy and {args.output}_bounds.npy...")
    
    # Save tokens as flat int32 array
    np.save(f"{args.output}_tokens.npy", np.frombuffer(tokens_out, dtype=np.int32))
    
    # Save bounds as [N, 2] int32 array
    bounds_np = np.frombuffer(bounds_out, dtype=np.int32).reshape(-1, 2)
    np.save(f"{args.output}_bounds.npy", bounds_np)
    
    print(f"Done! Processed {count} entries, total {curr} tokens.")

if __name__ == "__main__":
    main()
