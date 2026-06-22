#!/usr/bin/env python3
"""
Wrapper script to run lm_eval with the custom Onix model integration.

Run from anywhere:
    python evals/run_eval.py --model onix --model_args model_name=<name>,dtype=bfloat16 --tasks blimp --batch_size 32
"""

import os
import sys
from pathlib import Path

os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Add project root (parent of evals/) to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Also add evals/ itself so the sibling lm_eval_onix module is importable
evals_dir = Path(__file__).resolve().parent
if str(evals_dir) not in sys.path:
    sys.path.insert(0, str(evals_dir))

# Import the wrapper to trigger registration under the name "onix"
import lm_eval_onix  # noqa: F401

from lm_eval.__main__ import cli_evaluate

if __name__ == "__main__":
    cli_evaluate()
