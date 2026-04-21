"""Bracket a vLLM generation call with cudaProfilerStart/Stop.

vLLM V1 runs the GPU in an ``EngineCore`` subprocess. A naive
``cudaProfilerStart()`` in the parent process does nothing for ncu/nsys
because the profiler-capture flag never reaches the worker. The spike
at ``~/smoke/inproc_decode_rpc.py`` proved that
``llm.collective_rpc(fn)`` routes ``fn(worker)`` into the worker
subprocess, where ``torch.cuda.cudart().cudaProfilerStart()`` has the
intended effect. This module packages that pattern.

Typical use inside a vLLM script under ``colonel run --flavor vllm``:

    from vllm import LLM, SamplingParams
    from colonel.profiling.vllm import profile_region

    llm = LLM(model="Qwen/Qwen2.5-0.5B", enforce_eager=False)
    llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8))

    with profile_region(llm):
        outs = llm.generate(prompts, SamplingParams(max_tokens=128))

``--flavor vllm`` sets ``VLLM_ALLOW_INSECURE_SERIALIZATION=1`` and
``-c cudaProfilerApi`` on the profiler side. Without both pieces the
region will not appear in the capture.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

_START_STOP_ENV_OPT_OUT = "COLONEL_DISABLE_PROFILE_REGION"
_INSECURE_ENV = "VLLM_ALLOW_INSECURE_SERIALIZATION"


def _start(worker: Any) -> None:  # pragma: no cover - runs in worker
    """Called inside the vLLM EngineCore worker via collective_rpc."""
    import torch

    torch.cuda.cudart().cudaProfilerStart()


def _stop(worker: Any) -> None:  # pragma: no cover - runs in worker
    """Called inside the vLLM EngineCore worker via collective_rpc."""
    import torch

    torch.cuda.cudart().cudaProfilerStop()


@contextmanager
def profile_region(llm: Any, *, enable_nvtx_tracing: bool = True) -> Iterator[None]:
    """Bracket the enclosed block with cudaProfilerStart/Stop *inside the worker*.

    Args:
        llm: A ``vllm.LLM`` (V1). The helper calls
            ``llm.collective_rpc`` to run start/stop in the worker.
        enable_nvtx_tracing: If True, also enable vLLM's layerwise NVTX
            tracing the first time this is entered in a process.

    Opt-out:
        Setting ``COLONEL_DISABLE_PROFILE_REGION=1`` makes this a no-op,
        so the same user script can run uninstrumented in production
        and instrumented under ``colonel run --flavor vllm`` with no
        code change.

    Raises:
        RuntimeError: if ``VLLM_ALLOW_INSECURE_SERIALIZATION`` is not
            set. vLLM requires the env var to pass arbitrary callables
            through ``collective_rpc``. We fail loud rather than
            silently producing an empty profile.
    """
    if os.environ.get(_START_STOP_ENV_OPT_OUT) == "1":
        yield
        return

    if os.environ.get(_INSECURE_ENV) != "1":
        raise RuntimeError(
            "profile_region(): VLLM_ALLOW_INSECURE_SERIALIZATION=1 must be set "
            "in the environment so vLLM can ship the profiler start/stop "
            "closures through collective_rpc to the EngineCore worker. "
            "`colonel run --flavor vllm ...` sets this automatically; if you "
            "are invoking the script directly, export it yourself."
        )

    if enable_nvtx_tracing:
        # Imported locally so `import colonel.profiling.vllm` stays light.
        from colonel.profiling.vllm.nvtx import enable_nvtx

        enable_nvtx()

    llm.collective_rpc(_start)
    try:
        yield
    finally:
        llm.collective_rpc(_stop)
