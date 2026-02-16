#!/usr/bin/env python3
"""Demo 4: NCU deep-dive on attention and GEMM kernels.

Runs a minimal inference with a small model specifically for
Nsight Compute profiling, which replays each kernel multiple times.

Usage with Colonel:
    colonel run --name llm-ncu --evaluator ncu -- \
        python demos/vllm/04_ncu_deep_dive.py

What to look for:
    - Attention kernels: occupancy, memory throughput, stall reasons
    - GEMM kernels (ampere_*gemm*): compute vs memory throughput
    - Kernel launch grid/block dimensions
    - Register pressure and shared memory usage
    - L2 cache hit rates across different kernel types

WARNING: NCU profiling is SLOW (replays each kernel ~5x). Keep max-tokens
LOW. TinyLlama + 8 tokens should complete in 2-5 minutes under ncu.
"""
from __future__ import annotations

import argparse
import os
import sys

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="NCU deep-dive on LLM kernels")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="HuggingFace model ID (keep small for ncu!)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=8,
        help="Max new tokens (keep LOW for ncu, default: 8)",
    )
    parser.add_argument(
        "--dtype", default="float16",
        help="Model dtype",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype, torch.float16)

    print(f"Loading model: {args.model} (for NCU deep-dive)", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda",
    )
    model.eval()

    # Single short prompt — ncu replays each kernel, so keep workload minimal
    inputs = tokenizer(
        "What is attention in transformers?",
        return_tensors="pt",
    ).to("cuda")

    # Warm up (critical for ncu — avoids profiling JIT/allocation kernels)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=2, do_sample=False)
    torch.cuda.synchronize()

    # Profiled run
    print(f"Generating {args.max_tokens} tokens (ncu will replay kernels)...", file=sys.stderr)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,  # Deterministic for reproducibility
        )
    torch.cuda.synchronize()

    new_tokens = output_ids.shape[1] - inputs["input_ids"].shape[1]
    print(f"Generated {new_tokens} tokens", file=sys.stderr)
    print("LLM NCU deep-dive OK", file=sys.stderr)


if __name__ == "__main__":
    main()
