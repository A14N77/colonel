#!/usr/bin/env python3
"""Advanced GPU kernel showcase for Colonel profiling.

Demonstrates multiple kernel patterns that Colonel can analyze in depth:
1. Naive vs optimized matrix multiply (shows GEMM tile size impact)
2. Fused SiLU-Multiply activation (shows kernel fusion opportunity)
3. Reduction (shows memory-bound vs compute-bound classification)
4. Batched softmax (shows warp-level patterns)
5. Vector add baseline (shows bandwidth ceiling)

Each section is profiled separately so Colonel can identify bottlenecks
and suggest CUDA/Triton optimizations for each pattern.

Usage:
    colonel run --name kernel-showcase --evaluator nsys -- python gpu_smoke/kernel_showcase.py
    colonel run --name kernel-showcase-ncu --evaluator ncu -- python gpu_smoke/kernel_showcase.py --quick
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn.functional as F


def vector_add(n: int, iters: int = 50) -> None:
    """Pure memory-bandwidth workload — Colonel should flag this as memory-bound."""
    a = torch.randn(n, device="cuda", dtype=torch.float32)
    b = torch.randn(n, device="cuda", dtype=torch.float32)
    torch.cuda.synchronize()
    for _ in range(iters):
        c = a + b
    torch.cuda.synchronize()
    bytes_moved = n * 4 * 3 * iters  # read a, read b, write c
    print(f"  vector_add: {n:,} elements x {iters} iters, "
          f"~{bytes_moved / 1e9:.1f} GB moved", file=sys.stderr)


def naive_matmul(m: int, n: int, k: int, iters: int = 20) -> None:
    """Standard cuBLAS GEMM — Colonel should see high compute throughput."""
    a = torch.randn(m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(k, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(iters):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    flops = 2 * m * n * k * iters
    print(f"  matmul: [{m}x{k}] @ [{k}x{n}] x {iters} iters, "
          f"~{flops / 1e12:.2f} TFLOP", file=sys.stderr)


def batched_matmul(batch: int, m: int, n: int, k: int, iters: int = 20) -> None:
    """Batched GEMM — simulates multi-head attention's linear projections."""
    a = torch.randn(batch, m, k, device="cuda", dtype=torch.float16)
    b = torch.randn(batch, k, n, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(iters):
        c = torch.bmm(a, b)
    torch.cuda.synchronize()
    flops = 2 * batch * m * n * k * iters
    print(f"  batched_matmul: [{batch}x{m}x{k}] @ [{batch}x{k}x{n}] x {iters}, "
          f"~{flops / 1e12:.2f} TFLOP", file=sys.stderr)


def fused_silu_mul(n: int, d: int, iters: int = 50) -> None:
    """SiLU(x) * y — common in LLM FFN layers.

    PyTorch runs this as separate kernels (silu + multiply).
    Colonel should identify the fusion opportunity.
    """
    x = torch.randn(n, d, device="cuda", dtype=torch.float16)
    y = torch.randn(n, d, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(iters):
        out = F.silu(x) * y
    torch.cuda.synchronize()
    print(f"  silu_mul: [{n}x{d}] x {iters} iters (unfused — 2 kernels per call)",
          file=sys.stderr)


def rms_norm(n: int, d: int, iters: int = 50) -> None:
    """RMSNorm — used in every transformer layer.

    Colonel should see this as a reduction-heavy, memory-bound operation.
    """
    x = torch.randn(n, d, device="cuda", dtype=torch.float16)
    weight = torch.ones(d, device="cuda", dtype=torch.float16)
    eps = 1e-6
    torch.cuda.synchronize()
    for _ in range(iters):
        variance = x.pow(2).mean(-1, keepdim=True)
        out = x * torch.rsqrt(variance + eps) * weight
    torch.cuda.synchronize()
    print(f"  rms_norm: [{n}x{d}] x {iters} iters (multiple kernels per call)",
          file=sys.stderr)


def softmax_bench(n: int, seq_len: int, iters: int = 50) -> None:
    """Softmax over sequence dimension — key attention bottleneck.

    Colonel should identify memory access pattern and warp utilization.
    """
    x = torch.randn(n, seq_len, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(iters):
        out = F.softmax(x, dim=-1)
    torch.cuda.synchronize()
    print(f"  softmax: [{n}x{seq_len}] x {iters} iters", file=sys.stderr)


def flash_attention_bench(batch: int, heads: int, seq_len: int, head_dim: int, iters: int = 10) -> None:
    """Flash attention — the most important kernel in LLM inference.

    Uses PyTorch's scaled_dot_product_attention which dispatches to
    flash attention on supported hardware.
    """
    q = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(batch, heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    for _ in range(iters):
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    torch.cuda.synchronize()
    print(f"  flash_attn: batch={batch}, heads={heads}, seq={seq_len}, "
          f"head_dim={head_dim}, x {iters} iters", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU kernel showcase for Colonel")
    parser.add_argument(
        "--quick", action="store_true",
        help="Reduced iterations for NCU profiling (default: full)",
    )
    parser.add_argument(
        "--size", default="medium",
        choices=["small", "medium", "large"],
        help="Problem size (small/medium/large)",
    )
    args = parser.parse_args()

    # Scale iterations for NCU (which replays each kernel)
    scale = 1 if not args.quick else 0
    iters_mul = max(1, 5 if not args.quick else 1)

    sizes = {
        "small":  {"vec_n": 1_000_000, "mm": 512,  "batch": 4,  "seq": 128,  "heads": 8,  "hdim": 64,  "ffn_d": 2048},
        "medium": {"vec_n": 10_000_000, "mm": 2048, "batch": 8,  "seq": 512,  "heads": 16, "hdim": 64,  "ffn_d": 4096},
        "large":  {"vec_n": 50_000_000, "mm": 4096, "batch": 16, "seq": 1024, "heads": 32, "hdim": 128, "ffn_d": 8192},
    }
    s = sizes[args.size]

    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")

    gpu = torch.cuda.get_device_name(0)
    print(f"GPU Kernel Showcase ({args.size}, {'quick' if args.quick else 'full'})", file=sys.stderr)
    print(f"GPU: {gpu}\n", file=sys.stderr)

    # Warm up GPU
    _ = torch.randn(1000, 1000, device="cuda") @ torch.randn(1000, 1000, device="cuda")
    torch.cuda.synchronize()

    print("1. Vector Add (memory bandwidth ceiling)", file=sys.stderr)
    vector_add(s["vec_n"], iters=10 * iters_mul)

    print("2. GEMM — Matrix Multiply (compute throughput)", file=sys.stderr)
    naive_matmul(s["mm"], s["mm"], s["mm"], iters=5 * iters_mul)

    print("3. Batched GEMM (multi-head attention projections)", file=sys.stderr)
    batched_matmul(s["batch"], s["seq"], s["hdim"], s["mm"], iters=5 * iters_mul)

    print("4. SiLU * x (unfused activation — fusion opportunity)", file=sys.stderr)
    fused_silu_mul(s["batch"] * s["seq"], s["ffn_d"], iters=10 * iters_mul)

    print("5. RMSNorm (reduction-heavy, memory-bound)", file=sys.stderr)
    rms_norm(s["batch"] * s["seq"], s["mm"], iters=10 * iters_mul)

    print("6. Softmax (warp-level reduction)", file=sys.stderr)
    softmax_bench(s["batch"] * s["heads"], s["seq"], iters=10 * iters_mul)

    print("7. Flash Attention (the money kernel)", file=sys.stderr)
    flash_attention_bench(s["batch"], s["heads"], s["seq"], s["hdim"], iters=3 * iters_mul)

    torch.cuda.synchronize()
    print(f"\nKernel showcase complete", file=sys.stderr)


if __name__ == "__main__":
    main()
