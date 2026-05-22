#!/usr/bin/env python3
"""
Sample usage:
    python3 utils/transform_tinystories_instruct_to_natural.py \
        --input datasets/tiny-stories-instruct/instruction-data.jsonl \
        --output datasets/tiny-stories-instruct/natural-instruction-data.jsonl
"""

import json
import argparse
import re
from pathlib import Path

def transform_instruction(inst_text):
    # Extract components using regex
    summary_match = re.search(r'Summary:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)
    words_match = re.search(r'Words:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)
    features_match = re.search(r'Features:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)
    sentence_match = re.search(r'Random sentence:\s*(.*?)(?:\n|$)', inst_text, re.DOTALL)

    summary = summary_match.group(1).strip() if summary_match else ""
    words = words_match.group(1).strip() if words_match else ""
    features = features_match.group(1).strip() if features_match else ""
    sentence = sentence_match.group(1).strip() if sentence_match else ""

    # Build the natural language prompt
    parts = []
    if summary:
        # If summary ends with a period, remove it for the sentence construction
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

def main():
    parser = argparse.ArgumentParser(description="Transform TinyStories instruct data to natural language")
    parser.add_argument("--input", default="datasets/tiny-stories-instruct/instruction-data.jsonl", help="Input JSONL file")
    parser.add_argument("--output", default="datasets/tiny-stories-instruct/natural-instruction-data.jsonl", help="Output JSONL file")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: {input_path} not found.")
        return

    print(f"Transforming {input_path} -> {output_path}...")
    
    count = 0
    with open(input_path, 'r') as f_in, open(output_path, 'w') as f_out:
        for line in f_in:
            try:
                data = json.loads(line)
                original_inst = data.get("instruction", "")
                natural_inst = transform_instruction(original_inst)
                
                data["instruction"] = natural_inst
                f_out.write(json.dumps(data) + "\n")
                count += 1
            except Exception as e:
                print(f"Error processing line: {e}")
                continue

    print(f"Successfully transformed {count} entries.")

if __name__ == "__main__":
    main()
