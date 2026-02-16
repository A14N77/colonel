"""Nsight Systems (nsys) evaluator.

Wraps `nsys profile` to collect system-level GPU traces including:
- CUDA API call times
- GPU kernel execution times and launch counts
- Memory transfer times and sizes
- Overall GPU utilization timeline

After profiling, runs `nsys stats` to extract structured CSV data.
"""

from __future__ import annotations

import re
import shutil
from pathlib import Path

from colonel.core.context import ProfileContext
from colonel.core.result import (
    KernelSummary,
    MemoryTransfer,
    Metric,
    MetricCategory,
    ProfileResult,
)
from colonel.evaluators.base import BaseEvaluator
from colonel.targets.base import BaseTarget
from colonel.utils.parsers import parse_csv_output, safe_float, safe_int


class NsightSystemsEvaluator(BaseEvaluator):
    """Evaluator using NVIDIA Nsight Systems (nsys).

    Runs `nsys profile` to capture a trace, then `nsys stats --format csv`
    to extract structured metrics for kernel times, API overhead, and
    memory transfers.
    """

    def __init__(self, nsys_path: str = "nsys") -> None:
        """Initialize with path to nsys binary.

        Args:
            nsys_path: Path to nsys executable.
        """
        self._nsys_path = nsys_path
        self._resolved_nsys_path: str | None = None

    def _candidate_nsys_paths(self) -> list[str]:
        """Return paths to try for the nsys binary (e.g. Ubuntu puts it off PATH)."""
        candidates = [self._nsys_path]
        if "/" not in self._nsys_path:
            which_nsys = shutil.which("nsys")
            if which_nsys:
                candidates.append(which_nsys)
            candidates.extend([
                "/usr/local/bin/nsys",
                "/usr/lib/nsight-systems/bin/nsys",
                "/usr/bin/nsys",
            ])
            # Apt nsight-systems-2025.x installs under /opt/nvidia/nsight-systems/<version>/bin/nsys
            # Use known version paths to avoid listing the dir (can raise PermissionError).
            for ver in ("2025.5.2", "2025.3.2", "2024.6.2"):
                try:
                    bin_nsys = Path(f"/opt/nvidia/nsight-systems/{ver}/bin/nsys")
                    if bin_nsys.is_file():
                        candidates.append(str(bin_nsys))
                except OSError:
                    pass
            # If we can list the dir, add any other versions (newest first)
            try:
                opt_nsys = Path("/opt/nvidia/nsight-systems")
                if opt_nsys.is_dir():
                    for version_dir in sorted(opt_nsys.iterdir(), reverse=True):
                        if version_dir.name in ("2025.5.2", "2025.3.2", "2024.6.2"):
                            continue  # already added
                        bin_nsys = version_dir / "bin" / "nsys"
                        if bin_nsys.is_file():
                            candidates.append(str(bin_nsys))
            except OSError:
                pass  # Permission denied or missing dir; static paths already tried
        return candidates

    @property
    def name(self) -> str:
        return "nsys"

    @property
    def description(self) -> str:
        return (
            "NVIDIA Nsight Systems: system-level GPU trace including "
            "kernel times, API overhead, and memory transfers."
        )

    def is_available(self, target: BaseTarget) -> bool:
        """Check if nsys is installed on the target (tries known paths if not on PATH)."""
        for path in self._candidate_nsys_paths():
            result = target.run_command(f"{path} --version", timeout=10.0)
            if result.success:
                self._resolved_nsys_path = path
                return True
        return False

    def _report_base_path(self, ctx: ProfileContext) -> str:
        """Compute the base path for nsys report files (without extension)."""
        safe_name = re.sub(r"[^\w\-]", "_", (ctx.name or "profile").strip())[:64]
        return f"/tmp/colonel_nsys_{safe_name or 'profile'}"

    def build_command(self, ctx: ProfileContext) -> str:
        """Build the nsys profile command.

        Generates a command that profiles the application and writes
        a .nsys-rep file to a temp location.
        """
        nsys_bin = self._resolved_nsys_path or self._nsys_path
        report_file = self._report_base_path(ctx)
        cmd_parts = [
            nsys_bin,
            "profile",
            "--output", report_file,
            "--force-overwrite", "true",
            "--trace", "cuda,nvtx,osrt",
            "--stats", "true",
            "--export", "none",
            "--wait", "all",
            ctx.full_command,
        ]
        return " ".join(cmd_parts)

    def _build_stats_command(self, report_base: str) -> str:
        """Build the nsys stats command to extract CSV from a report."""
        nsys_bin = self._resolved_nsys_path or self._nsys_path
        return f"{nsys_bin} stats --format csv --force-export true {report_base}.nsys-rep"

    def parse_output(
        self,
        stdout: str,
        stderr: str,
        *,
        raw_report: str = "",
    ) -> ProfileResult:
        """Parse nsys profile + stats output into ProfileResult.

        nsys with --stats true prints CSV tables to stdout after profiling.
        We parse these tables for kernel summaries, API times, and transfers.

        Args:
            stdout: Combined stdout from nsys profile (includes stats).
            stderr: Stderr output.
            raw_report: Optional additional report content.

        Returns:
            Populated ProfileResult.
        """
        result = ProfileResult(evaluator_name=self.name)
        combined = stdout + "\n" + (raw_report or "")
        result.raw_output = combined

        # Parse the different CSV tables from nsys stats output.
        # nsys stats outputs multiple tables separated by headers like:
        #   CUDA Kernel Statistics:
        #   CUDA API Statistics:
        #   CUDA Memory Operation Statistics:
        kernels = self._parse_kernel_table(combined)
        api_metrics = self._parse_api_table(combined)
        transfers = self._parse_memory_table(combined)

        result.kernels = kernels
        result.transfers = transfers

        # Compute aggregate times
        result.kernel_time_us = sum(k.duration_us for k in kernels)
        result.transfer_time_us = sum(t.duration_us for t in transfers)
        result.api_time_us = sum(
            m.value for m in api_metrics if m.category == MetricCategory.API
        )
        result.gpu_time_us = result.kernel_time_us + result.transfer_time_us

        # Add all API metrics to the flat list
        result.metrics.extend(api_metrics)

        # Add summary metrics
        result.metrics.append(Metric(
            name="total_kernel_time_us",
            value=result.kernel_time_us,
            unit="us",
            category=MetricCategory.TIMING,
        ))
        result.metrics.append(Metric(
            name="total_transfer_time_us",
            value=result.transfer_time_us,
            unit="us",
            category=MetricCategory.TRANSFER,
        ))
        result.metrics.append(Metric(
            name="total_gpu_time_us",
            value=result.gpu_time_us,
            unit="us",
            category=MetricCategory.TIMING,
        ))
        result.metrics.append(Metric(
            name="kernel_count",
            value=float(len(kernels)),
            unit="",
            category=MetricCategory.OTHER,
        ))
        result.metrics.append(Metric(
            name="total_invocations",
            value=float(sum(k.invocations for k in kernels)),
            unit="",
            category=MetricCategory.OTHER,
        ))

        return result

    def _parse_kernel_table(self, text: str) -> list[KernelSummary]:
        """Extract kernel summaries from nsys stats CSV output."""
        kernels: list[KernelSummary] = []

        # Look for the CUDA Kernel Statistics table
        # nsys stats format: Time(%),Total Time (ns),Instances,Avg (ns),Med (ns),
        # Min (ns),Max (ns),StdDev (ns),Name
        section = self._extract_csv_section(text, "CUDA Kernel Statistics")
        if not section:
            # Try alternate: look for lines with kernel-like data
            section = self._extract_csv_section(text, "cuda_gpu_kern_sum")

        rows = parse_csv_output(section)
        for row in rows:
            name = row.get("Name", row.get("name", ""))
            if not name:
                continue

            total_ns = safe_float(row.get("Total Time (ns)", row.get("total_time", "0")))
            instances = safe_int(row.get("Instances", row.get("instances", "1")))
            avg_ns = safe_float(row.get("Avg (ns)", row.get("avg_time", "0")))

            kernels.append(KernelSummary(
                name=name,
                duration_us=total_ns / 1000.0,
                invocations=instances,
                avg_duration_us=avg_ns / 1000.0,
            ))

        return kernels

    def _parse_api_table(self, text: str) -> list[Metric]:
        """Extract CUDA API call metrics from nsys stats output."""
        metrics: list[Metric] = []

        section = self._extract_csv_section(text, "CUDA API Statistics")
        if not section:
            section = self._extract_csv_section(text, "cuda_api_sum")

        rows = parse_csv_output(section)
        for row in rows:
            name = row.get("Name", row.get("name", ""))
            if not name:
                continue

            total_ns = safe_float(row.get("Total Time (ns)", row.get("total_time", "0")))
            metrics.append(Metric(
                name=f"api_{name}",
                value=total_ns / 1000.0,
                unit="us",
                category=MetricCategory.API,
            ))

        return metrics

    def _parse_memory_table(self, text: str) -> list[MemoryTransfer]:
        """Extract memory transfer info from nsys stats output."""
        transfers: list[MemoryTransfer] = []

        section = self._extract_csv_section(text, "CUDA Memory Operation Statistics")
        if not section:
            section = self._extract_csv_section(text, "cuda_gpu_mem_time_sum")

        rows = parse_csv_output(section)
        for row in rows:
            name = row.get("Name", row.get("name", row.get("Operation", "")))
            if not name:
                continue

            direction = "HtoD" if "HtoD" in name or "Host" in name else (
                "DtoH" if "DtoH" in name or "Device" in name else "DtoD"
            )
            total_ns = safe_float(row.get("Total Time (ns)", row.get("total_time", "0")))
            size_bytes = safe_int(row.get("Total (bytes)", row.get("total_bytes", "0")))

            duration_us = total_ns / 1000.0
            throughput = 0.0
            if duration_us > 0:
                throughput = (size_bytes / 1e9) / (duration_us / 1e6)

            transfers.append(MemoryTransfer(
                direction=direction,
                size_bytes=size_bytes,
                duration_us=duration_us,
                throughput_gbps=throughput,
            ))

        return transfers

    def _extract_csv_section(self, text: str, header_keyword: str) -> str:
        """Extract a CSV table section from nsys stats output.

        nsys stats output contains multiple tables, each preceded by a header.
        This finds the section matching the keyword and returns its content.

        Args:
            text: Full nsys stats output.
            header_keyword: Keyword to search for in table headers.

        Returns:
            The CSV content of the matching section, or empty string.
        """
        lines = text.splitlines()
        in_section = False
        section_lines: list[str] = []

        for line in lines:
            if header_keyword.lower() in line.lower():
                in_section = True
                continue

            if in_section:
                # End of section: blank line after we've collected data
                if not line.strip() and section_lines:
                    break
                # Skip separator lines (all dashes, equals)
                if line.strip() and not re.match(r"^[-=]+$", line.strip()):
                    section_lines.append(line)

        return "\n".join(section_lines)

    def collect(self, ctx: ProfileContext, target: BaseTarget) -> ProfileResult:
        """Run nsys profile, then nsys stats --format csv, and parse results.

        Two-step process:
        1. Run nsys profile (captures trace, --stats true gives text summary)
        2. Run nsys stats --format csv on the report for machine-readable output

        Args:
            ctx: What to profile.
            target: Where to run.

        Returns:
            ProfileResult with all collected metrics.
        """
        command = self.build_command(ctx)
        cmd_result = target.run_command(
            command,
            working_dir=ctx.working_dir,
            env=ctx.env,
        )

        # Store the raw text stats output (useful for debugging)
        raw_text = cmd_result.stdout + "\n" + cmd_result.stderr

        # Run nsys stats --format csv for machine-readable output
        report_base = self._report_base_path(ctx)
        stats_cmd = self._build_stats_command(report_base)
        stats_result = target.run_command(stats_cmd, timeout=60.0)

        # Parse the CSV stats output (preferred) or fall back to text
        csv_output = stats_result.stdout if stats_result.success else ""
        profile_result = self.parse_output(
            csv_output or cmd_result.stdout,
            cmd_result.stderr,
            raw_report=raw_text,
        )
        profile_result.wall_time_s = cmd_result.duration_s
        profile_result.exit_code = cmd_result.exit_code

        if not cmd_result.success:
            error_detail = (cmd_result.stderr or cmd_result.stdout or "")[:500]
            profile_result.errors.append(
                f"nsys exited with code {cmd_result.exit_code}: {error_detail}"
            )

        return profile_result
