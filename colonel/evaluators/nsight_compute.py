"""Nsight Compute (ncu) evaluator.

Wraps `ncu` to collect detailed per-kernel hardware metrics including:
- Achieved occupancy
- Memory throughput (DRAM, L1, L2)
- Compute throughput (SM utilization)
- Stall reasons and instruction mix
- Roofline model data
"""

from __future__ import annotations

from colonel.core.context import ProfileContext
from colonel.core.result import (
    KernelSummary,
    Metric,
    MetricCategory,
    ProfileResult,
)
from colonel.evaluators.base import BaseEvaluator
from colonel.targets.base import BaseTarget
from colonel.utils.parsers import parse_duration_string, parse_ncu_csv, safe_float, safe_int


class NsightComputeEvaluator(BaseEvaluator):
    """Evaluator using NVIDIA Nsight Compute (ncu).

    Runs `ncu --csv` to collect detailed per-kernel hardware metrics.
    Best for deep-diving into individual kernel performance.
    """

    def __init__(self, ncu_path: str = "ncu") -> None:
        """Initialize with path to ncu binary.

        Args:
            ncu_path: Path to ncu executable.
        """
        self._ncu_path = ncu_path

    @property
    def name(self) -> str:
        return "ncu"

    @property
    def description(self) -> str:
        return (
            "NVIDIA Nsight Compute: detailed per-kernel hardware metrics "
            "including occupancy, memory throughput, and compute utilization."
        )

    def is_available(self, target: BaseTarget) -> bool:
        """Check if ncu is installed on the target."""
        result = target.run_command(f"{self._ncu_path} --version", timeout=10.0)
        return result.success

    def build_command(self, ctx: ProfileContext) -> str:
        """Build the ncu profiling command.

        Uses --csv for machine-readable output and collects a standard
        set of metrics for comprehensive analysis.
        """
        cmd_parts = [
            self._ncu_path,
            "--csv",
            "--log-file", "/dev/stdout",
            "--set", "full",
            "--target-processes", "all",
            ctx.full_command,
        ]
        return " ".join(cmd_parts)

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        *,
        raw_report: str = "",
    ) -> ProfileResult:
        """Parse ncu CSV output into ProfileResult.

        ncu --csv outputs one row per kernel launch with columns for
        each collected metric.

        Args:
            stdout: ncu CSV output.
            stderr: Stderr output.
            raw_report: Optional additional report content.

        Returns:
            Populated ProfileResult.
        """
        result = ProfileResult(evaluator_name=self.name)
        combined = stdout + "\n" + (raw_report or "")
        result.raw_output = combined

        rows = parse_ncu_csv(stdout)
        if not rows:
            return result

        # Aggregate kernel data: ncu may have multiple rows per kernel
        # (one per launch). Group by kernel name.
        kernel_data: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            name = row.get("Kernel Name", row.get("kernel_name", ""))
            if not name:
                continue
            kernel_data.setdefault(name, []).append(row)

        for name, launches in kernel_data.items():
            kernel = self._build_kernel_summary(name, launches)
            result.kernels.append(kernel)

            # Collect per-kernel metrics
            for metric in kernel.metrics:
                result.metrics.append(metric)

        # Compute aggregates
        result.kernel_time_us = sum(k.duration_us for k in result.kernels)
        result.gpu_time_us = result.kernel_time_us

        # Add summary metrics
        result.metrics.append(Metric(
            name="total_kernel_time_us",
            value=result.kernel_time_us,
            unit="us",
            category=MetricCategory.TIMING,
        ))
        result.metrics.append(Metric(
            name="kernel_count",
            value=float(len(result.kernels)),
            unit="",
            category=MetricCategory.OTHER,
        ))

        return result

    def _build_kernel_summary(
        self,
        name: str,
        launches: list[dict[str, str]],
    ) -> KernelSummary:
        """Build a KernelSummary from ncu rows for a single kernel.

        Args:
            name: Kernel function name.
            launches: List of ncu CSV rows for this kernel.

        Returns:
            KernelSummary with aggregated metrics.
        """
        total_duration_us = 0.0
        metrics: list[Metric] = []

        # Common ncu metric column names
        duration_col = "Duration"
        occupancy_col = "Achieved Occupancy"
        mem_throughput_col = "Memory Throughput"
        compute_throughput_col = "Compute (SM) Throughput"
        registers_col = "Registers Per Thread"
        shared_mem_col = "Static SMem Per Block"
        grid_size_col = "Grid Size"
        block_size_col = "Block Size"

        # Alternate column names (ncu versions vary)
        alt_names = {
            duration_col: ["gpu__time_duration.sum", "Duration"],
            occupancy_col: [
                "sm__warps_active.avg.pct_of_peak_sustained_active",
                "Achieved Occupancy",
            ],
            mem_throughput_col: [
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
                "Memory Throughput",
            ],
            compute_throughput_col: [
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
                "Compute (SM) Throughput",
            ],
        }

        occupancy_sum = 0.0
        mem_throughput_sum = 0.0
        compute_throughput_sum = 0.0
        registers = 0
        shared_mem = 0
        grid = ""
        block = ""

        for row in launches:
            # Duration: parse unit (ns/us/ms/s) and convert to µs
            dur = self._find_metric(row, duration_col, alt_names.get(duration_col, []))
            dur_us = parse_duration_string(str(dur))
            total_duration_us += dur_us

            # Occupancy
            occ = self._find_metric(row, occupancy_col, alt_names.get(occupancy_col, []))
            occupancy_sum += safe_float(occ)

            # Memory throughput
            mem = self._find_metric(
                row, mem_throughput_col, alt_names.get(mem_throughput_col, [])
            )
            mem_throughput_sum += safe_float(mem)

            # Compute throughput
            comp = self._find_metric(
                row, compute_throughput_col, alt_names.get(compute_throughput_col, [])
            )
            compute_throughput_sum += safe_float(comp)

            # Static info (take from first launch)
            if not grid:
                grid = row.get(grid_size_col, row.get("grid_size", ""))
                block = row.get(block_size_col, row.get("block_size", ""))
                registers = safe_int(
                    row.get(registers_col, row.get("registers_per_thread", "0"))
                )
                shared_mem = safe_int(
                    row.get(shared_mem_col, row.get("shared_memory", "0"))
                )

        n = len(launches)
        avg_occupancy = occupancy_sum / n if n > 0 else 0.0
        avg_mem_throughput = mem_throughput_sum / n if n > 0 else 0.0
        avg_compute_throughput = compute_throughput_sum / n if n > 0 else 0.0

        # Build per-kernel metrics
        metrics.extend([
            Metric(
                name="occupancy_pct",
                value=avg_occupancy,
                unit="%",
                category=MetricCategory.OCCUPANCY,
                kernel_name=name,
            ),
            Metric(
                name="memory_throughput_pct",
                value=avg_mem_throughput,
                unit="%",
                category=MetricCategory.MEMORY,
                kernel_name=name,
            ),
            Metric(
                name="compute_throughput_pct",
                value=avg_compute_throughput,
                unit="%",
                category=MetricCategory.COMPUTE,
                kernel_name=name,
            ),
        ])

        return KernelSummary(
            name=name,
            duration_us=total_duration_us,
            invocations=n,
            avg_duration_us=total_duration_us / n if n > 0 else 0.0,
            grid=grid,
            block=block,
            registers_per_thread=registers,
            shared_memory_bytes=shared_mem,
            occupancy_pct=avg_occupancy,
            metrics=metrics,
        )

    def _find_metric(
        self,
        row: dict[str, str],
        primary: str,
        alternates: list[str],
    ) -> str:
        """Find a metric value in a row, trying primary and alternate column names.

        Args:
            row: CSV row dict.
            primary: Primary column name.
            alternates: Alternative column names to try.

        Returns:
            The metric value as a string, or "0" if not found.
        """
        if primary in row and row[primary].strip():
            return row[primary]
        for alt in alternates:
            if alt in row and row[alt].strip():
                return row[alt]
        return "0"
