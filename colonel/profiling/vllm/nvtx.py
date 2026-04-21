"""Enable vLLM's layerwise NVTX tracing from outside vLLM's config.

vLLM's ``ObservabilityConfig.enable_layerwise_nvtx_tracing`` defaults
to ``False``. Flipping it yields per-layer NVTX ranges that show up in
``nsys stats --report nvtx_sum`` and make layer-level timelines useful.

The flag lives on a dataclass inside vLLM's config; the public way to
set it is via ``VllmConfig`` construction. Because users construct
``LLM(...)`` themselves, the simplest intercept is to set the
dataclass field default *before* they construct the LLM. This module
does that once per process, idempotently.

If vLLM is not importable (e.g. Colonel installed on a non-GPU box),
this is a no-op — callers need not guard.
"""

from __future__ import annotations

import os

_ENV_FLAG = "COLONEL_VLLM_ENABLE_NVTX"
_APPLIED = False


def enable_nvtx() -> bool:
    """Flip ``enable_layerwise_nvtx_tracing`` on the vLLM ObservabilityConfig default.

    Returns:
        True if the flag is now (or was already) enabled; False if vLLM
        is not importable or the config surface changed and we could
        not locate the attribute.
    """
    global _APPLIED
    if _APPLIED:
        return True

    # Allow env opt-out even though colonel --flavor vllm sets this =1.
    if os.environ.get(_ENV_FLAG) == "0":
        return False

    try:
        from vllm.config import ObservabilityConfig
    except Exception:
        return False

    # ObservabilityConfig is a dataclass; mutating the class-level
    # default affects instances constructed afterward. Guard in case
    # upstream renames the field.
    if not hasattr(ObservabilityConfig, "enable_layerwise_nvtx_tracing"):
        return False

    try:
        ObservabilityConfig.enable_layerwise_nvtx_tracing = True  # type: ignore[attr-defined]
    except Exception:
        return False

    _APPLIED = True
    return True
