"""Base evaluator interface for profiling tool wrappers."""

from __future__ import annotations

import abc

from colonel.core.context import ProfileContext
from colonel.core.result import ProfileResult
from colonel.targets.base import BaseTarget


class BaseEvaluator(abc.ABC):
    """Abstract base class for GPU profiling evaluators.

    Each evaluator wraps a specific profiling tool (nsys, ncu, etc.)
    and knows how to:
    1. Build the profiling command
    2. Parse the raw output into structured ProfileResult
    3. Detect whether the tool is available
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Short name of this evaluator (e.g. 'nsys', 'ncu')."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description of what this evaluator profiles."""
        ...

    @abc.abstractmethod
    def is_available(self, target: BaseTarget) -> bool:
        """Check if the profiling tool is installed on the target.

        Args:
            target: The execution target to check.

        Returns:
            True if the tool is available and usable.
        """
        ...

    @abc.abstractmethod
    def build_command(self, ctx: ProfileContext) -> str:
        """Build the full profiling command string.

        Args:
            ctx: The profile context describing what to profile.

        Returns:
            The complete shell command to invoke the profiler.
        """
        ...

    @abc.abstractmethod
    def parse_output(
        self,
        stdout: str,
        stderr: str,
        *,
        raw_report: str = "",
    ) -> ProfileResult:
        """Parse profiler output into a structured ProfileResult.

        Args:
            stdout: Standard output from the profiler command.
            stderr: Standard error from the profiler command.
            raw_report: Optional raw report content (e.g. CSV from nsys stats).

        Returns:
            A ProfileResult populated with parsed metrics.
        """
        ...

    def collect(self, ctx: ProfileContext, target: BaseTarget) -> ProfileResult:
        """Run the profiler and return parsed results.

        This is the main entry point. It builds the command, runs it on
        the target, then parses the output.

        Args:
            ctx: What to profile.
            target: Where to run.

        Returns:
            ProfileResult with all collected metrics.
        """
        command = self.build_command(ctx)
        result = target.run_command(
            command,
            working_dir=ctx.working_dir,
            env=ctx.env,
        )

        profile_result = self.parse_output(
            result.stdout,
            result.stderr,
        )
        profile_result.wall_time_s = result.duration_s
        profile_result.exit_code = result.exit_code
        profile_result.evaluator_name = self.name

        if not result.success:
            profile_result.errors.append(
                f"Profiler exited with code {result.exit_code}: {result.stderr[:500]}"
            )

        return profile_result
