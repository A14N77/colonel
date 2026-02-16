"""Executor: orchestrates profiling by combining targets and evaluators.

The executor is the central coordination point:
1. Resolves the target (local or SSH)
2. Selects the evaluator (nsys, ncu, or auto-detect)
3. Optionally collects hardware info
4. Runs the profiler via the evaluator
5. Returns a complete ProfileResult
"""

from __future__ import annotations

import csv
import io
import uuid
from datetime import datetime, timezone

from colonel.config.settings import ColonelSettings, get_settings
from colonel.core.context import ProfileContext
from colonel.core.result import HardwareInfo, ProfileResult
from colonel.evaluators.registry import get_evaluator
from colonel.targets.base import BaseTarget
from colonel.targets.local import LocalTarget
from colonel.targets.ssh import SSHTarget
from colonel.utils.parsers import safe_int


def _resolve_target(ctx: ProfileContext) -> BaseTarget:
    """Create the appropriate target from the context.

    Args:
        ctx: Profile context with target specification.

    Returns:
        A BaseTarget instance (LocalTarget or SSHTarget).
    """
    if ctx.is_remote:
        return SSHTarget(ctx.target)
    return LocalTarget()


def _collect_hardware_info(target: BaseTarget) -> HardwareInfo:
    """Collect GPU and system hardware info from the target.

    Runs nvidia-smi and system commands to gather hardware details.

    Args:
        target: The execution target.

    Returns:
        Populated HardwareInfo.
    """
    info = HardwareInfo()

    # GPU info via nvidia-smi
    gpu_result = target.run_command(
        "nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap "
        "--format=csv,noheader,nounits",
        timeout=15.0,
    )
    if gpu_result.success and gpu_result.stdout.strip():
        # Parse as CSV so GPU names containing commas don't break
        reader = csv.reader(io.StringIO(gpu_result.stdout.strip()))
        row = next(reader, None)
        if row and len(row) >= 4:
            info.gpu_name = row[0].strip()
            info.gpu_memory_mb = safe_int(row[1].strip())
            info.driver_version = row[2].strip()
            info.compute_capability = row[3].strip()

    # CUDA version
    cuda_result = target.run_command(
        "nvcc --version 2>/dev/null | tail -1",
        timeout=10.0,
    )
    if cuda_result.success and cuda_result.stdout.strip():
        info.cuda_version = cuda_result.stdout.strip()

    # CPU info
    cpu_result = target.run_command(
        "lscpu 2>/dev/null | head -20 || sysctl -n machdep.cpu.brand_string 2>/dev/null",
        timeout=10.0,
    )
    if cpu_result.success and cpu_result.stdout.strip():
        info.cpu_name = cpu_result.stdout.strip().splitlines()[0]

    # System memory
    mem_result = target.run_command(
        "free -m 2>/dev/null | awk '/^Mem:/{print $2}' || "
        "sysctl -n hw.memsize 2>/dev/null | awk '{print int($1/1048576)}'",
        timeout=10.0,
    )
    if mem_result.success and mem_result.stdout.strip():
        info.system_memory_mb = safe_int(mem_result.stdout.strip())

    return info


class Executor:
    """Orchestrates profiling runs.

    Usage:
        executor = Executor()
        result = executor.run(context)
    """

    def __init__(self, settings: ColonelSettings | None = None) -> None:
        """Initialize the executor.

        Args:
            settings: Optional settings override.
        """
        self._settings = settings or get_settings()

    def run(
        self,
        ctx: ProfileContext,
        *,
        collect_hardware: bool = True,
    ) -> ProfileResult:
        """Execute a profiling run.

        This is the main entry point for profiling. It:
        1. Connects to the target
        2. Detects/selects the profiler
        3. Collects hardware info
        4. Runs the profiler
        5. Returns the result

        Args:
            ctx: What to profile and how.
            collect_hardware: Whether to gather hardware info.

        Returns:
            Complete ProfileResult.
        """
        session_id = uuid.uuid4().hex[:12]

        with _resolve_target(ctx) as target:
            # Select evaluator
            evaluator = get_evaluator(
                ctx.evaluator,
                target,
                self._settings,
            )

            # Collect hardware info
            hardware = HardwareInfo()
            if collect_hardware:
                hardware = _collect_hardware_info(target)

            # Run the profiler
            result = evaluator.collect(ctx, target)

            # Enrich the result
            result.session_id = session_id
            result.timestamp = datetime.now(timezone.utc).isoformat()
            result.hardware = hardware

        return result

    def detect_tools(self, ctx: ProfileContext) -> list[str]:
        """Detect which profiling tools are available on the target.

        Args:
            ctx: Context specifying the target.

        Returns:
            List of available evaluator names.
        """
        from colonel.evaluators.registry import detect_available

        with _resolve_target(ctx) as target:
            return detect_available(target)
