#!/usr/bin/env python3
"""Demo 2: Batch size sweep — shows how GPU utilization scales with batching.

Profile each batch size separately for session compare:
    colonel run --name llm-b1 --evaluator nsys -- python demos/vllm/02_batch_sweep.py --batch-size 1
    colonel run --name llm-b4 --evaluator nsys -- python demos/vllm/02_batch_sweep.py --batch-size 4
    colonel run --name llm-b16 --evaluator nsys -- python demos/vllm/02_batch_sweep.py --batch-size 16

Then compare:
    colonel session compare <b1-id> <b16-id> --ai

What to look for:
    - batch=1: Low occupancy (20-40%), memory-bound decode dominates
    - batch=4: Better occupancy, GEMM kernels get wider
    - batch=16: Near-peak occupancy, compute-bound prefill dominates
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM batch size sweep")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1,
        help="Number of concurrent prompts (default: 1)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="Max new tokens per prompt (default: 32)",
    )
    parser.add_argument(
        "--dtype", default="float16",
        help="Model dtype (default: float16)",
    )
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, args.dtype, torch.float16)

    print(f"Loading model: {args.model}", file=sys.stderr)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map="cuda",
    )
    model.eval()

    # Diverse prompts for batching
    prompts = [
        "What is the capital of France?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
        "What are the benefits of exercise?",
        "Describe the process of photosynthesis.",
        "How does a transformer model work?",
        "What is the difference between TCP and UDP?",
        "Explain the concept of recursion.",
        "What causes earthquakes?",
        "How do neural networks learn?",
        "What is the speed of light?",
        "Describe the water cycle.",
        "How does encryption work?",
        "What is machine learning?",
        "Explain how compilers work.",
        "What is the theory of relativity?",
    ]
    batch = prompts[: args.batch_size]

    # Tokenize with left-padding for batch generation
    tokenizer.padding_side = "left"
    inputs = tokenizer(batch, return_tensors="pt", padding=True).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Warm up
    print("Warming up...", file=sys.stderr)
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    # Profiled run
    print(f"Running batch_size={args.batch_size}, max_tokens={args.max_tokens}", file=sys.stderr)
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

    total_new_tokens = (output_ids.shape[1] - input_len) * args.batch_size
    tps = total_new_tokens / elapsed if elapsed > 0 else 0

    print(f"\n--- Results ---", file=sys.stderr)
    print(f"Batch size:    {args.batch_size}", file=sys.stderr)
    print(f"Total tokens:  {total_new_tokens}", file=sys.stderr)
    print(f"Wall time:     {elapsed:.2f}s", file=sys.stderr)
    print(f"Throughput:    {tps:.1f} tokens/sec", file=sys.stderr)
    print(f"LLM batch sweep OK (batch={args.batch_size})", file=sys.stderr)


if __name__ == "__main__":
    main()
