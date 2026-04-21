"""vLLM-aware profiling helpers.

These are for use *inside* user scripts that are being profiled by
Colonel with ``--flavor vllm``. The CLI flag sets environment and
evaluator knobs; this package provides the in-process glue the user's
script needs (mainly ``profile_region``).

Minimal public surface:

- :func:`profile_region` — context manager that gates
  ``cudaProfilerStart/Stop`` inside vLLM V1's EngineCore subprocess
  via ``llm.collective_rpc``.
- :func:`enable_nvtx` — flip
  ``ObservabilityConfig.enable_layerwise_nvtx_tracing`` before LLM
  construction so nsys captures per-layer NVTX ranges.

Heavy imports (``torch``, ``vllm``) are deferred to call sites so
``import colonel`` stays cheap.
"""

from colonel.profiling.vllm.nvtx import enable_nvtx
from colonel.profiling.vllm.region import profile_region

__all__ = ["profile_region", "enable_nvtx"]
