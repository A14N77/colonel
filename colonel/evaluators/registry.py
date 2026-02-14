"""Evaluator registry: auto-detect and select profiling evaluators."""

from __future__ import annotations

from colonel.config.settings import ColonelSettings, get_settings
from colonel.evaluators.base import BaseEvaluator
from colonel.evaluators.nsight_compute import NsightComputeEvaluator
from colonel.evaluators.nsight_systems import NsightSystemsEvaluator
from colonel.targets.base import BaseTarget


def get_all_evaluators(settings: ColonelSettings | None = None) -> list[BaseEvaluator]:
    """Return all known evaluator instances.

    Args:
        settings: Optional settings to configure evaluators.

    Returns:
        List of all evaluator instances.
    """
    if settings is None:
        settings = get_settings()

    return [
        NsightSystemsEvaluator(nsys_path=settings.nsys_path),
        NsightComputeEvaluator(ncu_path=settings.ncu_path),
    ]


def get_evaluator(
    name: str,
    target: BaseTarget,
    settings: ColonelSettings | None = None,
) -> BaseEvaluator:
    """Get an evaluator by name, or auto-detect the best available one.

    Args:
        name: Evaluator name ("nsys", "ncu", or "auto").
        target: The target to check tool availability on.
        settings: Optional settings.

    Returns:
        The selected evaluator.

    Raises:
        RuntimeError: If no suitable evaluator is found.
    """
    evaluators = get_all_evaluators(settings)

    if name != "auto":
        for ev in evaluators:
            if ev.name == name:
                if not ev.is_available(target):
                    raise RuntimeError(
                        f"Evaluator '{name}' is not available on target. "
                        f"Ensure {name} is installed and in PATH."
                    )
                return ev
        available = [ev.name for ev in evaluators]
        raise RuntimeError(
            f"Unknown evaluator '{name}'. Available: {available}"
        )

    # Auto-detect: prefer nsys for system-level overview
    preference_order = ["nsys", "ncu"]
    for pref_name in preference_order:
        for ev in evaluators:
            if ev.name == pref_name and ev.is_available(target):
                return ev

    raise RuntimeError(
        "No GPU profiling tools found. Install NVIDIA Nsight Systems (nsys) "
        "or Nsight Compute (ncu) and ensure they are in your PATH."
    )


def detect_available(target: BaseTarget) -> list[str]:
    """Detect which profiling tools are available on a target.

    Args:
        target: The execution target to check.

    Returns:
        List of available evaluator names.
    """
    available = []
    for ev in get_all_evaluators():
        if ev.is_available(target):
            available.append(ev.name)
    return available
