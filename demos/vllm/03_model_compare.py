#!/usr/bin/env python3
"""Demo 3: Model size comparison — shows different kernel profiles across models.

Usage with Colonel:
    # Small model (fast, ~3GB VRAM)
    colonel run --name llm-tinyllama --evaluator nsys -- \
        python demos/vllm/03_model_compare.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Medium model (~6GB VRAM)
    colonel run --name llm-phi2 --evaluator nsys -- \
        python demos/vllm/03_model_compare.py --model microsoft/phi-2

    # Large model (~14GB VRAM, needs HF login for gated models)
    colonel run --name llm-mistral7b --evaluator nsys -- \
        python demos/vllm/03_model_compare.py --model mistralai/Mistral-7B-v0.1

Then compare:
    colonel session compare <tinyllama-id> <mistral-id> --ai

What to look for:
    - More layers = more kernel launches per decode step
    - Larger hidden dim = wider GEMM tiles, better GPU utilization
    - 7B models saturate DRAM bandwidth during decode (~80-90%)
    - 1B models leave GPU underutilized — room for larger batches
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM model comparison")
    parser.add_argument(
        "--model",
        default=os.environ.get("DEMO_MODEL", "TinyLlama/TinyLlama-1.1B-Chat-v1.0"),
        help="HuggingFace model ID",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=4,
        help="Number of prompts in batch",
    )
    parser.add_argument(
        "--dtype", default="float16",
        help="Model dtype",
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

    prompts = [
        "Explain how GPU memory bandwidth affects deep learning inference.",
        "What is the attention mechanism and why is it efficient?",
        "Describe the difference between prefill and decode phases in LLM inference.",
        "How does KV-cache management work in transformer models?",
    ][: args.batch_size]

    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
    input_len = inputs["input_ids"].shape[1]

    # Warm up
    with torch.no_grad():
        _ = model.generate(**inputs, max_new_tokens=4, do_sample=False)
    torch.cuda.synchronize()

    # Profiled run
    print(f"Generating batch={args.batch_size}, max_tokens={args.max_tokens}...", file=sys.stderr)
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

    total_new_tokens = (output_ids.shape[1] - input_len) * len(prompts)
    tps = total_new_tokens / elapsed if elapsed > 0 else 0

    print(f"\n--- {args.model} ---", file=sys.stderr)
    print(f"Total tokens:  {total_new_tokens}", file=sys.stderr)
    print(f"Wall time:     {elapsed:.2f}s", file=sys.stderr)
    print(f"Throughput:    {tps:.1f} tokens/sec", file=sys.stderr)
    print(f"LLM model compare OK ({args.model})", file=sys.stderr)


if __name__ == "__main__":
    main()
