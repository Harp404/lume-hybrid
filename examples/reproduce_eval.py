"""Reproduce the Lume-Hybrid evaluation suite against a locally-trained checkpoint.

This codebase ships code only — there are no pretrained weights to download.
Train first (see ../README.md), then point this script at the resulting checkpoint.

Usage:
    python examples/reproduce_eval.py
    python examples/reproduce_eval.py --checkpoint lume_hybrid_weights/lume_hybrid.safetensors
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys


DEFAULT_CHECKPOINT = "lume_hybrid_weights/lume_hybrid.safetensors"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluate.py against a locally-trained checkpoint.")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to a locally-trained .safetensors checkpoint.",
    )
    args, extra = parser.parse_known_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(
            f"Checkpoint not found at {args.checkpoint!r}.\n"
            "This codebase does not download pretrained weights. Train first:\n"
            "    python train.py train --path lume_hybrid.safetensors\n"
            "Then re-run with --checkpoint <path>."
        )

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    eval_script = os.path.join(repo_root, "evaluate.py")
    cmd = [sys.executable, eval_script, "--checkpoint", args.checkpoint, *extra]
    print(f"[run] {' '.join(cmd)}\n")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
