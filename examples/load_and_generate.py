"""Load a locally-trained Lume-Hybrid checkpoint and generate text.

This script assumes you have already trained Lume-Hybrid (or a smaller variant)
and have a `.safetensors` checkpoint on disk. No weights are downloaded —
the published codebase ships code only.

Usage:
    # Train first (see ../README.md):
    #     python train.py train --path lume_hybrid.safetensors
    #
    # Then generate:
    python examples/load_and_generate.py
    python examples/load_and_generate.py --path lume_hybrid.safetensors --prompt "Once upon a time"
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modeling import LumeConfig, LumeModel


DEFAULT_PROMPT = "The capital of France is"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained Lume-Hybrid checkpoint.")
    parser.add_argument(
        "--path", default="lume_hybrid.safetensors",
        help="Path to a locally-trained checkpoint (.safetensors).",
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT, help="Generation prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise SystemExit(
            f"Checkpoint not found at {args.path!r}.\n"
            "This codebase does not download pretrained weights. Train first:\n"
            "    python train.py train --path lume_hybrid.safetensors"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("[warn] No CUDA available — generation will be very slow on CPU.")

    print(f"[load] reading checkpoint: {args.path}")
    sd = load_file(args.path)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    print("[load] building model")
    cfg = LumeConfig()
    model = LumeModel(cfg).to(device).eval()
    model.load_state_dict(sd, strict=False)
    print(f"[load] {model.count_params() / 1e6:.1f}M params on {device}")

    print("[load] tokenizer")
    tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

    print(f"\nprompt: {args.prompt!r}\n")
    ids = tok.encode(args.prompt, return_tensors="pt").to(device)
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    else:
        out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    print(tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
