"""Local execution target using subprocess."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from pathlib import Path

from colonel.targets.base import BaseTarget, CommandResult


class LocalTarget(BaseTarget):
    """Execute commands and manage files on the local machine."""

    def run_command(
        self,
        command: str,
        *,
        working_dir: str = ".",
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a shell command locally via subprocess.

        Args:
            command: Shell command string.
            working_dir: Working directory.
            env: Extra environment variables (merged with current env).
            timeout: Max seconds to wait.

        Returns:
            CommandResult with captured output.
        """
        full_env = dict(os.environ)
        if env:
            full_env.update(env)

        cwd = Path(working_dir).resolve()
        if not cwd.is_dir():
            return CommandResult(
                exit_code=-1,
                stderr=f"Working directory does not exist: {cwd}",
                command=command,
            )

        start = time.monotonic()
        try:
            proc = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                cwd=str(cwd),
                env=full_env,
                timeout=timeout,
            )
            duration = time.monotonic() - start
            return CommandResult(
                exit_code=proc.returncode,
                stdout=proc.stdout,
                stderr=proc.stderr,
                duration_s=duration,
                command=command,
            )
        except subprocess.TimeoutExpired:
            duration = time.monotonic() - start
            return CommandResult(
                exit_code=-1,
                stderr=f"Command timed out after {timeout}s",
                duration_s=duration,
                command=command,
            )
        except OSError as exc:
            duration = time.monotonic() - start
            return CommandResult(
                exit_code=-1,
                stderr=f"OS error: {exc}",
                duration_s=duration,
                command=command,
            )

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Copy a local file to another local path (for interface consistency)."""
        dest = Path(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)

    def download(self, remote_path: str, local_path: Path) -> None:
        """Copy a local file to another local path (for interface consistency)."""
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(remote_path, local_path)

    def file_exists(self, path: str) -> bool:
        """Check if a file exists locally."""
        return Path(path).exists()

    def close(self) -> None:
        """No-op for local target."""
        pass
