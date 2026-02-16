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
from colonel.utils.parsers import parse_duration_string, parse_ncu_csv, pivot_ncu_rows, safe_float, safe_int


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

        ncu ``--csv --set full`` outputs a long-format CSV with one row
        per metric per kernel launch.  We pivot that into per-launch
        dicts grouped by kernel name, then build summaries.

        Args:
            stdout: ncu CSV output (may include app stdout).
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

        # Pivot long-format rows into {kernel_name: [launch_dict, …]}
        kernel_data = pivot_ncu_rows(rows)
        if not kernel_data:
            return result

        for name, launches in kernel_data.items():
            kernel = self._build_kernel_summary(name, launches)
            result.kernels.append(kernel)

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
        """Build a KernelSummary from pivoted ncu dicts for a single kernel.

        Each *launch* dict maps metric names (e.g. ``"Duration"``,
        ``"Achieved Occupancy"``) directly to string values, plus
        ``"Block Size"`` / ``"Grid Size"`` from the CSV columns.

        Args:
            name: Kernel function name.
            launches: Pivoted dicts, one per kernel launch.

        Returns:
            KernelSummary with aggregated metrics.
        """
        total_duration_us = 0.0
        metrics: list[Metric] = []

        occupancy_sum = 0.0
        mem_throughput_sum = 0.0
        compute_throughput_sum = 0.0
        sm_busy_sum = 0.0
        l2_hit_sum = 0.0
        registers = 0
        shared_mem = 0
        grid = ""
        block = ""

        for row in launches:
            # Duration – ncu reports in nanoseconds; attach unit for parser
            dur_val = self._find_metric(row, "Duration", [
                "GPU Speed Of Light Throughput::Duration",
                "gpu__time_duration.sum",
            ])
            dur_unit = row.get("Duration__unit", "ns")
            dur_us = parse_duration_string(f"{dur_val} {dur_unit}")
            total_duration_us += dur_us

            # Achieved occupancy (%)
            occ = self._find_metric(row, "Achieved Occupancy", [
                "Occupancy::Achieved Occupancy",
                "sm__warps_active.avg.pct_of_peak_sustained_active",
            ])
            occupancy_sum += safe_float(occ)

            # Memory throughput (% of peak) – from Speed Of Light section
            mem = self._find_metric(row, "DRAM Throughput", [
                "GPU Speed Of Light Throughput::Memory Throughput",
                "GPU Speed Of Light Throughput::DRAM Throughput",
                "Memory Throughput",
                "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed",
            ])
            mem_throughput_sum += safe_float(mem)

            # Compute (SM) throughput (%)
            comp = self._find_metric(row, "Compute (SM) Throughput", [
                "GPU Speed Of Light Throughput::Compute (SM) Throughput",
                "sm__throughput.avg.pct_of_peak_sustained_elapsed",
            ])
            compute_throughput_sum += safe_float(comp)

            # Extra useful metrics
            sm_busy_sum += safe_float(row.get("SM Busy", "0"))
            l2_hit_sum += safe_float(row.get("L2 Hit Rate", "0"))

            # Static info (take from first launch)
            if not grid:
                grid = row.get("Grid Size", "")
                block = row.get("Block Size", "")
                # Launch Statistics metrics
                registers = safe_int(self._find_metric(
                    row, "Registers Per Thread", ["registers_per_thread"],
                ))
                shared_mem = safe_int(self._find_metric(
                    row, "Static Shared Memory Per Block", [
                        "Static SMem Per Block",
                        "shared_memory",
                    ],
                ))

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

        # Extra detail metrics when available
        if sm_busy_sum > 0:
            metrics.append(Metric(
                name="sm_busy_pct",
                value=sm_busy_sum / n,
                unit="%",
                category=MetricCategory.COMPUTE,
                kernel_name=name,
            ))
        if l2_hit_sum > 0:
            metrics.append(Metric(
                name="l2_hit_rate_pct",
                value=l2_hit_sum / n,
                unit="%",
                category=MetricCategory.MEMORY,
                kernel_name=name,
            ))

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
