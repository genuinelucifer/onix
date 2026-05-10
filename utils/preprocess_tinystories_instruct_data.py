#!/usr/bin/env python3
"""
Preprocess TinyStoriesInstruct Dataset
--------------------------------------
This script is SPECIFIC to the `roneneldan/TinyStoriesInstruct` dataset. 
It converts the dataset's custom JSON/JSONL format (which contains 'prompt' and 'text' fields, 
or a combined 'text' field) into the standard Alpaca JSON format expected by `finetune.py`.

Usage:
    python utils/preprocess_tinystories_instruct_data.py \
        --input datasets/tiny-stories-instruct/train.json \
        --output instruction-data.json
"""

import json
import argparse
import sys
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Preprocess TinyStoriesInstruct dataset for fine-tuning")
    parser.add_argument("--input", required=True, help="Input file (e.g. datasets/tiny-stories-instruct/train.json)")
    parser.add_argument("--output", required=True, help="Output file (e.g. instruction-data.json)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} does not exist.")
        sys.exit(1)

    out_data = []
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: {args.input} does not exist.")
        sys.exit(1)

    current_story_lines = []
    count = 0
    
    print(f"Reading {input_path} and writing jsonl to {args.output}...")
    with open(input_path, "r") as f_in, open(args.output, "w") as f_out:
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
                            f_out.write(json.dumps({
                                "instruction": prompt,
                                "input": "",
                                "output": text
                            }) + "\n")
                            count += 1
                    current_story_lines = []
                else:
                    current_story_lines.append(text_val)
            
    print(f"Processed {count} records successfully.")
        
    print("Done!")

if __name__ == "__main__":
    main()
