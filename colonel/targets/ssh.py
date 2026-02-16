"""SSH execution target using paramiko."""

from __future__ import annotations

import getpass
import shlex
import time
from pathlib import Path
from urllib.parse import urlparse

import paramiko

from colonel.targets.base import BaseTarget, CommandResult


class SSHTarget(BaseTarget):
    """Execute commands and manage files on a remote machine via SSH.

    The target URI format is: ssh://user@host[:port]

    Authentication order:
    1. Explicit key file (--ssh-key)
    2. SSH agent (ssh-add)
    3. Default key files (~/.ssh/id_rsa, id_ed25519, etc.)
    4. Interactive passphrase prompt if key is encrypted

    Password auth is not supported for security reasons.
    """

    def __init__(
        self,
        uri: str,
        *,
        key_filename: str | None = None,
        connect_timeout: float = 30.0,
    ) -> None:
        """Initialize SSH connection.

        Args:
            uri: SSH URI like "ssh://user@host" or "ssh://user@host:2222".
            key_filename: Optional path to a private key file.
            connect_timeout: Connection timeout in seconds.
        """
        parsed = urlparse(uri)
        if parsed.scheme != "ssh":
            raise ValueError(f"Expected ssh:// URI, got: {uri}")

        self._host = parsed.hostname or "localhost"
        self._port = parsed.port or 22
        self._username = parsed.username or ""

        self._client = paramiko.SSHClient()
        self._client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            self._client.connect(
                hostname=self._host,
                port=self._port,
                username=self._username,
                key_filename=key_filename,
                timeout=connect_timeout,
                allow_agent=True,
                look_for_keys=True,
            )
        except paramiko.PasswordRequiredException:
            # Key file is encrypted — prompt for passphrase interactively
            self._connect_with_passphrase(
                key_filename=key_filename,
                connect_timeout=connect_timeout,
            )
        except paramiko.AuthenticationException as exc:
            raise ConnectionError(
                f"SSH authentication failed for {self._username}@{self._host}:{self._port}.\n"
                f"  Error: {exc}\n"
                f"  Try one of:\n"
                f"    1. Load your key into the agent: ssh-add\n"
                f"    2. Specify a key file: --ssh-key /path/to/key\n"
                f"    3. Ensure your public key is on the remote host"
            ) from exc

        self._sftp: paramiko.SFTPClient | None = None

    def _connect_with_passphrase(
        self,
        *,
        key_filename: str | None,
        connect_timeout: float,
    ) -> None:
        """Retry SSH connection after prompting for key passphrase.

        Args:
            key_filename: Optional path to a private key file.
            connect_timeout: Connection timeout in seconds.
        """
        passphrase = getpass.getpass(
            f"SSH key passphrase for {self._username}@{self._host}: "
        )
        try:
            self._client.connect(
                hostname=self._host,
                port=self._port,
                username=self._username,
                key_filename=key_filename,
                passphrase=passphrase,
                timeout=connect_timeout,
                allow_agent=False,
                look_for_keys=True,
            )
        except paramiko.AuthenticationException as exc:
            raise ConnectionError(
                f"SSH authentication failed (wrong passphrase or key rejected).\n"
                f"  Error: {exc}\n"
                f"  Tip: load your key into the agent to avoid passphrase prompts:\n"
                f"    ssh-add ~/.ssh/id_ed25519"
            ) from exc

    def _get_sftp(self) -> paramiko.SFTPClient:
        """Get or create an SFTP client."""
        if self._sftp is None:
            self._sftp = self._client.open_sftp()
        return self._sftp

    def run_command(
        self,
        command: str,
        *,
        working_dir: str = ".",
        env: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command on the remote host via SSH.

        Args:
            command: Shell command to execute.
            working_dir: Remote working directory.
            env: Extra environment variables.
            timeout: Max seconds to wait.

        Returns:
            CommandResult with captured output.
        """
        # Build the full command with cd and env (shell-escaped)
        parts: list[str] = []
        if env:
            exports = " ".join(
                f"{k}={shlex.quote(v)}" for k, v in env.items()
            )
            parts.append(f"export {exports};")
        if working_dir and working_dir != ".":
            parts.append(f"cd {shlex.quote(working_dir)} &&")
        parts.append(command)
        full_command = " ".join(parts)

        start = time.monotonic()
        try:
            _stdin, stdout_ch, stderr_ch = self._client.exec_command(
                full_command,
                timeout=timeout,
            )
            stdout = stdout_ch.read().decode("utf-8", errors="replace")
            stderr = stderr_ch.read().decode("utf-8", errors="replace")
            exit_code = stdout_ch.channel.recv_exit_status()
            duration = time.monotonic() - start

            return CommandResult(
                exit_code=exit_code,
                stdout=stdout,
                stderr=stderr,
                duration_s=duration,
                command=command,
            )
        except Exception as exc:
            duration = time.monotonic() - start
            return CommandResult(
                exit_code=-1,
                stderr=f"SSH error: {exc}",
                duration_s=duration,
                command=command,
            )

    def upload(self, local_path: Path, remote_path: str) -> None:
        """Upload a local file to the remote host via SFTP."""
        sftp = self._get_sftp()
        # Ensure remote directory exists
        remote_dir = str(Path(remote_path).parent)
        try:
            sftp.stat(remote_dir)
        except (FileNotFoundError, OSError):
            self.run_command(f"mkdir -p {shlex.quote(remote_dir)}")
        sftp.put(str(local_path), remote_path)

    def download(self, remote_path: str, local_path: Path) -> None:
        """Download a file from the remote host via SFTP."""
        sftp = self._get_sftp()
        local_path.parent.mkdir(parents=True, exist_ok=True)
        sftp.get(remote_path, str(local_path))

    def file_exists(self, path: str) -> bool:
        """Check if a file exists on the remote host."""
        sftp = self._get_sftp()
        try:
            sftp.stat(path)
            return True
        except FileNotFoundError:
            return False

    def close(self) -> None:
        """Close SSH and SFTP connections."""
        if self._sftp is not None:
            try:
                self._sftp.close()
            except Exception:
                pass
            self._sftp = None
        try:
            self._client.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        return f"SSHTarget({self._username}@{self._host}:{self._port})"
