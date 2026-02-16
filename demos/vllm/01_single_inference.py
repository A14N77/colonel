#!/usr/bin/env python3
"""Demo 1: Single LLM inference pass — shows prefill vs decode phases.

Uses HuggingFace Transformers directly (single-process) so nsys/ncu
can capture all CUDA kernels. The kernel profiles are the same as
vLLM under the hood — same GEMM, same attention, same model weights.

Usage with Colonel:
    colonel run --name llm-single-nsys --evaluator nsys -- \
        python demos/vllm/01_single_inference.py

    colonel run --name llm-single-ncu --evaluator ncu -- \
        python demos/vllm/01_single_inference.py --max-tokens 8

What to look for in Colonel output:
    - Prefill phase: large GEMM kernels (compute-bound, high SM throughput)
    - Decode phase: smaller kernels (memory-bound, high DRAM throughput)
    - Attention kernels (flash_attn or sdpa) and their occupancy
    - Memory transfers during KV cache allocation
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Single LLM inference")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="HuggingFace model ID (default: TinyLlama-1.1B)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64,
        help="Max new tokens to generate (default: 64)",
    )
    parser.add_argument(
        "--prompt",
        default="Explain how GPU memory bandwidth affects deep learning inference performance.",
        help="Input prompt",
    )
    parser.add_argument(
        "--dtype", default="float16",
        choices=["float16", "bfloat16", "float32"],
        help="Model dtype (default: float16)",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Loading model: {args.model} (dtype={args.dtype})", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=dtype,
        device_map="cuda",
    )
    model.eval()

    # Tokenize
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Warm up (first run has extra overhead from JIT, memory allocation, etc.)
    print("Warming up...", file=sys.stderr)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    # Actual profiled run
    print(f"Generating {args.max_tokens} tokens...", file=sys.stderr)
    start = time.perf_counter()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    new_tokens = output_ids.shape[1] - input_len
    tps = new_tokens / elapsed if elapsed > 0 else 0

    text = tokenizer.decode(output_ids[0][input_len:], skip_special_tokens=True)
    print(f"\n--- Generated {new_tokens} tokens in {elapsed:.2f}s ({tps:.1f} tok/s) ---", file=sys.stderr)
    print(text[:500], file=sys.stderr)
    print(f"\nLLM single inference OK", file=sys.stderr)


if __name__ == "__main__":
    main()
