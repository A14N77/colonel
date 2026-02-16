"""ProfileContext: the central data object for a profiling run.

Modeled after PEAK's kernel context -- encapsulates everything needed
to run and profile a target application or GPU kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ProfileContext:
    """Immutable description of what to profile and how.

    Attributes:
        command: The executable or script to run (e.g. "./my_kernel", "python train.py").
        args: Command-line arguments passed to the command.
        env: Extra environment variables to set for the run.
        working_dir: Working directory for execution. Defaults to ".".
        target: Where to run -- "local" or "ssh://user@host[:port]".
        evaluator: Which profiler to use -- "nsys", "ncu", or "auto".
        name: Optional human-readable label for this run.
        metadata: Arbitrary key-value metadata attached to this context.
    """

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    working_dir: str = "."
    target: str = "local"
    evaluator: str = "auto"
    name: str | None = None
    ssh_key: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def full_command(self) -> str:
        """Return the full command string including arguments.

        Arguments containing shell metacharacters are quoted so the
        resulting string is safe for ``shell=True`` execution.
        """
        import shlex

        parts = [self.command] + [shlex.quote(a) for a in self.args]
        return " ".join(parts)

    @property
    def is_remote(self) -> bool:
        """Return True if this context targets a remote machine."""
        return self.target.startswith("ssh://")

    def with_overrides(self, **kwargs: Any) -> ProfileContext:
        """Return a new context with the given fields overridden."""
        from dataclasses import asdict

        current = asdict(self)
        current.update(kwargs)
        return ProfileContext(**current)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict for JSON storage."""
        from dataclasses import asdict

        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProfileContext:
        """Deserialize from a plain dict."""
        return cls(**data)
