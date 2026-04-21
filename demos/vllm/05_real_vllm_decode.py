#!/usr/bin/env python3
"""Demo 5: the first *actual* vLLM demo in this repo.

Unlike 01_*–04_* which run HuggingFace Transformers and document that
the kernels are 'the same shape as vLLM,' this one runs real vLLM V1.
It is designed to be invoked through::

    colonel run --flavor vllm --evaluator nsys -- python demos/vllm/05_real_vllm_decode.py
    colonel run --flavor vllm --evaluator ncu  -- python demos/vllm/05_real_vllm_decode.py

``--flavor vllm`` sets the environment and profiler knobs that let
nsys/ncu actually see kernels launched inside vLLM's EngineCore
subprocess. If you forget ``--flavor vllm``, the region below will
raise a clear error rather than producing an empty profile.

Model default (``Qwen/Qwen2.5-0.5B``) fits easily on a 16 GB A4000.
Override with ``--model`` for other tiers; bigger models may need
``--max-model-len`` lowered to fit.

Known limitation (Phase 1, being fixed in Phase 2):
    Under ``--evaluator ncu``, the default graph-captured decode
    (``enforce_eager=False``) can hang ncu's kernel-replay pass. Pass
    ``--eager`` to this demo to disable CUDA graphs when profiling
    with ncu. Under ``--evaluator nsys`` both modes work.
"""
from __future__ import annotations

import argparse
import os
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Real vLLM decode under Colonel")
    parser.add_argument(
        "--model",
        default=os.environ.get("COLONEL_VLLM_MODEL", "Qwen/Qwen2.5-0.5B"),
        help="HuggingFace model id (default: Qwen/Qwen2.5-0.5B).",
    )
    parser.add_argument(
        "--max-model-len", type=int, default=2048,
        help="vLLM max_model_len (default: 2048).",
    )
    parser.add_argument(
        "--gpu-memory-utilization", type=float, default=0.7,
        help="Fraction of GPU memory for the KV cache (default: 0.7).",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=32,
        help="Tokens per request in the profiled region (default: 32).",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=4,
        help="Batch size in the profiled region (default: 4).",
    )
    parser.add_argument(
        "--eager", action="store_true",
        help="Disable CUDA graphs (pass enforce_eager=True).",
    )
    args = parser.parse_args()

    # Local import so `python this_file.py --help` works without vllm.
    from vllm import LLM, SamplingParams

    from colonel.profiling.vllm import profile_region

    print(f"[demo] loading model: {args.model}", file=sys.stderr, flush=True)
    llm = LLM(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.eager,
    )

    # Warm the engine: first decode populates CUDA graphs, compile cache,
    # prefill shapes. We run twice because the first pass sometimes lazy-
    # initializes extra state.
    warmup_params = SamplingParams(max_tokens=8, temperature=0.0)
    for _ in range(2):
        llm.generate(["Hello, world."] * max(1, args.num_prompts // 2),
                     warmup_params)

    prompts = ["Write a single short sentence."] * args.num_prompts
    region_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=0.0,
        ignore_eos=True,
    )

    print(
        f"[demo] entering profile_region: {args.num_prompts} prompts × "
        f"{args.max_tokens} tokens",
        file=sys.stderr, flush=True,
    )
    with profile_region(llm):
        outs = llm.generate(prompts, region_params)

    if outs and outs[0].outputs:
        print(f"[demo] sample output: {outs[0].outputs[0].text[:80]!r}",
              file=sys.stderr, flush=True)
    print("[demo] done", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
