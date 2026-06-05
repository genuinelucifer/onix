#!/usr/bin/env python3
"""
Preprocess and Transform TinyStoriesInstruct Dataset
----------------------------------------------------
This script processes the raw `roneneldan/TinyStoriesInstruct` dataset.
It parses the raw HF JSON format, optionally transforms the prompt into natural language,
and either outputs an Alpaca-style JSONL file or tokenizes the data into binary shards.

Usage:
    # Convert raw data to natural language JSONL:
    python utils/preprocess_tinystories_instruct_data.py \
        --input datasets/tiny-stories-instruct/train.json \
        --output datasets/tiny-stories-instruct/natural-instruction-data.jsonl \
        --to-natural

    # Convert raw data to natural language AND tokenize it into binary shards:
    python utils/preprocess_tinystories_instruct_data.py \
        --input datasets/tiny-stories-instruct/train.json \
        --output datasets/tiny-stories-instruct/natural-instruction-data \
        --to-natural --to-tokenized
"""

import json
import argparse
import sys
import re
from pathlib import Path
import array
import numpy as np

# Add parent directory to path to import tokenizer
sys.path.append(str(Path(__file__).parent.parent))
from model import get_tokenizer

def transform_instruction(inst_text):
    """Transform structured feature prompts into natural language prompt sentences."""
    summary_match = re.search(r'Summary:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)
    words_match = re.search(r'Words:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)
    features_match = re.search(r'Features:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)
    sentence_match = re.search(r'Random sentence:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)

    summary = summary_match.group(1).strip() if summary_match else ""
    words = words_match.group(1).strip() if words_match else ""
    features = features_match.group(1).strip() if features_match else ""
    sentence = sentence_match.group(1).strip() if sentence_match else ""

    parts = []
    if summary:
        clean_summary = summary.rstrip('.')
        parts.append(f"Tell me a story about {clean_summary}.")
    else:
        parts.append("Tell me a story.")

    if words:
        parts.append(f"It should have the words {words}.")
    
    if features:
        parts.append(f"The style should be {features}.")
    
    if sentence:
        parts.append(f"And it should include the sentence: \"{sentence}\"")

    return " ".join(parts)

def format_input_alpaca(entry):
    """Format instruction into standard Alpaca SFT template."""
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry.get("input") else ""
    return instruction_text + input_text

def main():
    parser = argparse.ArgumentParser(description="Preprocess and transform TinyStoriesInstruct dataset")
    parser.add_argument("--input", required=True, help="Input file (e.g. datasets/tiny-stories-instruct/train.json)")
    parser.add_argument("--output", required=True, help="Output file path / prefix for .npy files")
    parser.add_argument("--to-natural", action="store_true", help="Transform feature prompts to natural language prompts")
    parser.add_argument("--to-tokenized", action="store_true", help="Binarize and pre-tokenize data to _tokens.npy and _bounds.npy shards")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} does not exist.")
        sys.exit(1)

    print(f"Loading tokenizer...")
    tokenizer = get_tokenizer()

    tokens_out = array.array('i')
    bounds_out = array.array('i')
    curr_token_offset = 0
    count = 0

    current_story_lines = []

    # If writing jsonl
    f_out = None
    if not args.to_tokenized:
        f_out = open(args.output, "w")
        print(f"Reading {input_path} and writing JSONL to {args.output}...")
    else:
        print(f"Reading {input_path} and tokenizing to binary shards...")

    with open(input_path, "r") as f_in:
        for line in f_in:
            line = line.strip()
            if line.startswith('"text":'):
                if line.endswith(','):
                    line = line[:-1]
                try:
                    obj = json.loads("{" + line + "}")
                    text_val = obj["text"]
                except json.JSONDecodeError:
                    continue
                
                if text_val == "<|endoftext|>":
                    full_story = "\n".join(current_story_lines).strip()
                    if full_story:
                        if "Story:" in full_story:
                            parts = full_story.split("Story:", 1)
                            prompt = parts[0].replace("Prompt:", "").strip()
                            text = parts[1].strip()
                        else:
                            prompt = ""
                            text = full_story

                        if prompt or text:
                            # Apply natural transformation if requested
                            if args.to_natural and prompt:
                                prompt = transform_instruction(prompt)

                            entry = {
                                "instruction": prompt,
                                "input": "",
                                "output": text
                            }

                            if args.to_tokenized:
                                full_text = format_input_alpaca(entry) + f"\n\n### Response:\n{entry['output']}"
                                tokens = tokenizer.encode(full_text)
                                tokens_out.extend(tokens)
                                
                                bounds_out.append(curr_token_offset)
                                bounds_out.append(curr_token_offset + len(tokens))
                                curr_token_offset += len(tokens)
                            else:
                                f_out.write(json.dumps(entry) + "\n")
                            
                            count += 1
                            if count % 50000 == 0:
                                print(f"  Processed {count} entries...")
                                
                    current_story_lines = []
                else:
                    current_story_lines.append(text_val)

    if not args.to_tokenized:
        f_out.close()
        print(f"Successfully processed {count} records and saved JSONL file.")
    else:
        tokens_path = f"{args.output}_tokens.npy"
        bounds_path = f"{args.output}_bounds.npy"
        print(f"Saving to {tokens_path} and {bounds_path}...")
        
        np.save(tokens_path, np.frombuffer(tokens_out, dtype=np.int32))
        
        bounds_np = np.frombuffer(bounds_out, dtype=np.int32).reshape(-1, 2)
        np.save(bounds_path, bounds_np)
        
        print(f"Successfully processed {count} records (total {curr_token_offset} tokens).")

if __name__ == "__main__":
    main()
