"""ProfileResult and supporting data models for profiling output.

These models capture the structured output from GPU profilers (nsys, ncu)
in a tool-agnostic format that the analysis agent and reports can consume.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class MetricCategory(str, Enum):
    """Categories for profiling metrics."""

    TIMING = "timing"
    MEMORY = "memory"
    COMPUTE = "compute"
    OCCUPANCY = "occupancy"
    TRANSFER = "transfer"
    API = "api"
    OTHER = "other"


class BottleneckSeverity(str, Enum):
    """Severity levels for identified bottlenecks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


@dataclass
class Metric:
    """A single profiling metric.

    Attributes:
        name: Metric name (e.g. "kernel_time_us", "memory_throughput_pct").
        value: Numeric value.
        unit: Unit of measurement (e.g. "us", "%", "GB/s").
        category: Which category this metric belongs to.
        kernel_name: If per-kernel, the kernel this metric applies to.
        metadata: Any extra info about this metric.
    """

    name: str
    value: float
    unit: str = ""
    category: MetricCategory = MetricCategory.OTHER
    kernel_name: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "name": self.name,
            "value": self.value,
            "unit": self.unit,
            "category": self.category.value,
            "kernel_name": self.kernel_name,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Metric:
        """Deserialize from a plain dict."""
        data = dict(data)
        if "category" in data and isinstance(data["category"], str):
            data["category"] = MetricCategory(data["category"])
        return cls(**data)


@dataclass
class KernelSummary:
    """Summary of a single GPU kernel invocation.

    Attributes:
        name: Kernel function name.
        duration_us: Total duration in microseconds.
        invocations: Number of times this kernel was launched.
        avg_duration_us: Average duration per invocation.
        grid: Grid dimensions as a string (e.g. "256,1,1").
        block: Block dimensions as a string (e.g. "128,1,1").
        registers_per_thread: Register usage per thread.
        shared_memory_bytes: Shared memory per block in bytes.
        occupancy_pct: Achieved occupancy as a percentage.
        metrics: Additional per-kernel metrics.
    """

    name: str
    duration_us: float = 0.0
    invocations: int = 1
    avg_duration_us: float = 0.0
    grid: str = ""
    block: str = ""
    registers_per_thread: int = 0
    shared_memory_bytes: int = 0
    occupancy_pct: float = 0.0
    metrics: list[Metric] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "name": self.name,
            "duration_us": self.duration_us,
            "invocations": self.invocations,
            "avg_duration_us": self.avg_duration_us,
            "grid": self.grid,
            "block": self.block,
            "registers_per_thread": self.registers_per_thread,
            "shared_memory_bytes": self.shared_memory_bytes,
            "occupancy_pct": self.occupancy_pct,
            "metrics": [m.to_dict() for m in self.metrics],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> KernelSummary:
        """Deserialize from a plain dict."""
        data = dict(data)
        if "metrics" in data:
            data["metrics"] = [Metric.from_dict(m) for m in data["metrics"]]
        return cls(**data)


@dataclass
class MemoryTransfer:
    """A host-device or device-host memory transfer event.

    Attributes:
        direction: "HtoD", "DtoH", or "DtoD".
        size_bytes: Transfer size in bytes.
        duration_us: Transfer duration in microseconds.
        throughput_gbps: Throughput in GB/s.
    """

    direction: str
    size_bytes: int = 0
    duration_us: float = 0.0
    throughput_gbps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "direction": self.direction,
            "size_bytes": self.size_bytes,
            "duration_us": self.duration_us,
            "throughput_gbps": self.throughput_gbps,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryTransfer:
        """Deserialize from a plain dict."""
        return cls(**data)


@dataclass
class Bottleneck:
    """An identified performance bottleneck.

    Attributes:
        title: Short description of the bottleneck.
        description: Detailed explanation.
        severity: How impactful this bottleneck is.
        category: Which metric category it relates to.
        affected_kernels: Kernel names affected by this bottleneck.
        evidence: Supporting metric values.
    """

    title: str
    description: str = ""
    severity: BottleneckSeverity = BottleneckSeverity.MEDIUM
    category: MetricCategory = MetricCategory.OTHER
    affected_kernels: list[str] = field(default_factory=list)
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "category": self.category.value,
            "affected_kernels": self.affected_kernels,
            "evidence": self.evidence,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Bottleneck:
        """Deserialize from a plain dict."""
        data = dict(data)
        if "severity" in data and isinstance(data["severity"], str):
            data["severity"] = BottleneckSeverity(data["severity"])
        if "category" in data and isinstance(data["category"], str):
            data["category"] = MetricCategory(data["category"])
        return cls(**data)


@dataclass
class HardwareInfo:
    """GPU and system hardware information.

    Attributes:
        gpu_name: GPU device name (e.g. "NVIDIA RTX 4090").
        gpu_memory_mb: GPU memory in MB.
        compute_capability: CUDA compute capability (e.g. "8.9").
        driver_version: NVIDIA driver version.
        cuda_version: CUDA toolkit version.
        cpu_name: CPU model name.
        system_memory_mb: System RAM in MB.
        extra: Any additional hardware info.
    """

    gpu_name: str = ""
    gpu_memory_mb: int = 0
    compute_capability: str = ""
    driver_version: str = ""
    cuda_version: str = ""
    cpu_name: str = ""
    system_memory_mb: int = 0
    extra: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "gpu_name": self.gpu_name,
            "gpu_memory_mb": self.gpu_memory_mb,
            "compute_capability": self.compute_capability,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "cpu_name": self.cpu_name,
            "system_memory_mb": self.system_memory_mb,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HardwareInfo:
        """Deserialize from a plain dict."""
        return cls(**data)


@dataclass
class ProfileResult:
    """Complete result of a profiling run.

    This is the main data object that flows through the system:
    evaluators produce it, the analysis agent consumes it,
    and sessions store it.

    Attributes:
        session_id: Unique ID for this profiling session.
        timestamp: When the profiling run occurred.
        evaluator_name: Which evaluator produced this result ("nsys", "ncu").
        wall_time_s: Total wall-clock time of the profiled application.
        gpu_time_us: Total GPU time (kernel + transfer) in microseconds.
        kernel_time_us: Total kernel execution time in microseconds.
        transfer_time_us: Total memory transfer time in microseconds.
        api_time_us: Total CUDA API overhead in microseconds.
        kernels: Per-kernel summaries.
        transfers: Memory transfer events.
        metrics: Flat list of all collected metrics.
        bottlenecks: Identified bottlenecks (may be populated by agent).
        hardware: Hardware info from the profiled system.
        raw_output: Raw profiler output for reference.
        exit_code: Exit code of the profiled application.
        errors: Any errors encountered during profiling.
    """

    session_id: str = ""
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    evaluator_name: str = ""
    wall_time_s: float = 0.0
    gpu_time_us: float = 0.0
    kernel_time_us: float = 0.0
    transfer_time_us: float = 0.0
    api_time_us: float = 0.0
    kernels: list[KernelSummary] = field(default_factory=list)
    transfers: list[MemoryTransfer] = field(default_factory=list)
    metrics: list[Metric] = field(default_factory=list)
    bottlenecks: list[Bottleneck] = field(default_factory=list)
    hardware: HardwareInfo = field(default_factory=HardwareInfo)
    raw_output: str = ""
    exit_code: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        """Return True if the profiling run completed without errors."""
        return self.exit_code == 0 and len(self.errors) == 0

    @property
    def kernel_count(self) -> int:
        """Total number of unique kernels observed."""
        return len(self.kernels)

    @property
    def total_invocations(self) -> int:
        """Total kernel invocations across all kernels."""
        return sum(k.invocations for k in self.kernels)

    @property
    def top_kernels(self) -> list[KernelSummary]:
        """Return kernels sorted by total duration (descending)."""
        return sorted(self.kernels, key=lambda k: k.duration_us, reverse=True)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full result to a plain dict for JSON storage."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "evaluator_name": self.evaluator_name,
            "wall_time_s": self.wall_time_s,
            "gpu_time_us": self.gpu_time_us,
            "kernel_time_us": self.kernel_time_us,
            "transfer_time_us": self.transfer_time_us,
            "api_time_us": self.api_time_us,
            "kernels": [k.to_dict() for k in self.kernels],
            "transfers": [t.to_dict() for t in self.transfers],
            "metrics": [m.to_dict() for m in self.metrics],
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "hardware": self.hardware.to_dict(),
            "raw_output": self.raw_output,
            "exit_code": self.exit_code,
            "errors": self.errors,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileResult:
        """Deserialize from a plain dict."""
        data = dict(data)
        if "kernels" in data:
            data["kernels"] = [KernelSummary.from_dict(k) for k in data["kernels"]]
        if "transfers" in data:
            data["transfers"] = [MemoryTransfer.from_dict(t) for t in data["transfers"]]
        if "metrics" in data:
            data["metrics"] = [Metric.from_dict(m) for m in data["metrics"]]
        if "bottlenecks" in data:
            data["bottlenecks"] = [Bottleneck.from_dict(b) for b in data["bottlenecks"]]
        if "hardware" in data:
            data["hardware"] = HardwareInfo.from_dict(data["hardware"])
        return cls(**data)
