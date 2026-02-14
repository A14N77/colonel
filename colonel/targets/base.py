"""Base target interface for local and remote execution."""

from __future__ import annotations

import abc
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CommandResult:
    """Result of running a command on a target.

    Attributes:
        exit_code: Process exit code.
        stdout: Standard output.
        stderr: Standard error.
        duration_s: Wall-clock duration in seconds.
        command: The command that was executed.
    """

    exit_code: int
    stdout: str = ""
    stderr: str = ""
    duration_s: float = 0.0
    command: str = ""

    @property
    def success(self) -> bool:
        """Return True if the command exited with code 0."""
        return self.exit_code == 0


class BaseTarget(abc.ABC):
    """Abstract base class for execution targets.

    A target knows how to run commands and transfer files.
    Implementations: LocalTarget (subprocess) and SSHTarget (paramiko).
    """

    @abc.abstractmethod
    def run_command(
        self,
        command: str,
        *,
        working_dir: str = ".",
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Execute a shell command on the target.

        Args:
            command: The shell command to run.
            working_dir: Working directory for execution.
            env: Extra environment variables.
            timeout: Maximum seconds to wait (None = no limit).

        Returns:
            CommandResult with exit code, stdout, stderr, and duration.
        """
        ...

    @abc.abstractmethod
    def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to the target.

        Args:
            local_path: Local file path.
            remote_path: Destination path on the target.
        """
        ...

    @abc.abstractmethod
    def download(self, remote_path: str, local_path: Path) -> None:
        """Download a file from the target to local.

        Args:
            remote_path: Path on the target.
            local_path: Local destination path.
        """
        ...

    @abc.abstractmethod
    def file_exists(self, path: str) -> bool:
        """Check if a file exists on the target."""
        ...

    @abc.abstractmethod
    def close(self) -> None:
        """Clean up any connections or resources."""
        ...

    def __enter__(self) -> BaseTarget:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()
