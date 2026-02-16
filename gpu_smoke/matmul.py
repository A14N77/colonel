#!/usr/bin/env python3
"""Minimal GPU workload for Colonel smoke tests: small matrix multiply on CUDA.
Run with: python matmul.py
"""
from __future__ import annotations

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


def main() -> None:
    n = 1024
    if HAS_TORCH and torch.cuda.is_available():
        a = torch.randn(n, n, device="cuda", dtype=torch.float32)
        b = torch.randn(n, n, device="cuda", dtype=torch.float32)
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        print("PyTorch matmul 1024x1024 OK")
        return
    if HAS_CUPY:
        a = cp.random.randn(n, n, dtype=cp.float32)
        b = cp.random.randn(n, n, dtype=cp.float32)
        c = cp.dot(a, b)
        cp.cuda.Stream.null.synchronize()
        print("CuPy matmul 1024x1024 OK")
        return
    raise SystemExit("Need PyTorch or CuPy with CUDA. Install with: pip install torch")


if __name__ == "__main__":
    main()
