"""Session Manager: PEAK-style workflows with checkpoints.

Each `colonel profile` run creates a checkpoint containing:
- ProfileContext (what was profiled)
- ProfileResult (metrics and data)
- Agent analysis (if performed)
- Timestamp and metadata

Sessions are stored in .colonel/sessions/<session_id>/ as JSON files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import orjson

from colonel.config.settings import get_settings
from colonel.core.context import ProfileContext
from colonel.core.result import ProfileResult


@dataclass
class Checkpoint:
    """A saved profiling session checkpoint.

    Attributes:
        session_id: Unique session identifier.
        timestamp: When the checkpoint was created.
        name: Human-readable label.
        context: The ProfileContext used for this run.
        result: The ProfileResult from profiling.
        analysis: Agent analysis text, if available.
        metadata: Arbitrary metadata.
    """

    session_id: str
    timestamp: str
    name: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    result: dict[str, Any] = field(default_factory=dict)
    analysis: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a plain dict."""
        return {
            "session_id": self.session_id,
            "timestamp": self.timestamp,
            "name": self.name,
            "context": self.context,
            "result": self.result,
            "analysis": self.analysis,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Checkpoint:
        """Deserialize from a plain dict."""
        return cls(**data)

    @property
    def display_name(self) -> str:
        """Return a display-friendly name."""
        return self.name or self.session_id[:8]

    @property
    def short_id(self) -> str:
        """Return a short session ID for display."""
        return self.session_id[:8]


class SessionManager:
    """Manages profiling session checkpoints.

    Stores and retrieves checkpoints from the .colonel/sessions/ directory.
    Provides listing, comparison, and lookup operations.
    """

    def __init__(self, sessions_dir: str | Path | None = None) -> None:
        """Initialize the session manager.

        Args:
            sessions_dir: Directory for storing sessions.
                         Defaults to settings value.
        """
        if sessions_dir is None:
            settings = get_settings()
            sessions_dir = settings.sessions_dir
        self._dir = Path(sessions_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    @property
    def sessions_dir(self) -> Path:
        """Return the sessions directory path."""
        return self._dir

    def save(
        self,
        context: ProfileContext,
        result: ProfileResult,
        *,
        name: str = "",
        analysis: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> Checkpoint:
        """Save a profiling run as a checkpoint.

        Args:
            context: The profile context.
            result: The profiling result.
            name: Optional human-readable label.
            analysis: Optional agent analysis text.
            metadata: Optional extra metadata.

        Returns:
            The saved Checkpoint.
        """
        checkpoint = Checkpoint(
            session_id=result.session_id,
            timestamp=result.timestamp,
            name=name or context.name or "",
            context=context.to_dict(),
            result=result.to_dict(),
            analysis=analysis,
            metadata=metadata or {},
        )

        # Write to disk
        session_dir = self._dir / checkpoint.session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = session_dir / "checkpoint.json"
        data = orjson.dumps(checkpoint.to_dict(), option=orjson.OPT_INDENT_2)
        checkpoint_path.write_bytes(data)

        # Also save raw profiler output separately if it's large
        if result.raw_output and len(result.raw_output) > 1000:
            raw_path = session_dir / "raw_output.txt"
            raw_path.write_text(result.raw_output)

        return checkpoint

    def load(self, session_id: str) -> Checkpoint:
        """Load a checkpoint by session ID.

        Supports both full and partial (prefix) session IDs.

        Args:
            session_id: Full or partial session ID.

        Returns:
            The loaded Checkpoint.

        Raises:
            FileNotFoundError: If no matching session is found.
            ValueError: If the session ID is ambiguous.
        """
        # Try exact match first
        exact_path = self._dir / session_id / "checkpoint.json"
        if exact_path.is_file():
            return self._read_checkpoint(exact_path)

        # Try prefix match
        matches = [
            d for d in self._dir.iterdir()
            if d.is_dir() and d.name.startswith(session_id)
        ]

        if len(matches) == 0:
            raise FileNotFoundError(f"No session found matching '{session_id}'")
        if len(matches) > 1:
            ids = [m.name[:8] for m in matches]
            raise ValueError(
                f"Ambiguous session ID '{session_id}'. Matches: {ids}"
            )

        checkpoint_path = matches[0] / "checkpoint.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Session directory exists but checkpoint.json is missing: {matches[0]}"
            )
        return self._read_checkpoint(checkpoint_path)

    def list_sessions(self, *, limit: int = 50) -> list[Checkpoint]:
        """List all saved sessions, most recent first.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of Checkpoints sorted by timestamp (newest first).
        """
        checkpoints: list[Checkpoint] = []

        for session_dir in self._dir.iterdir():
            if not session_dir.is_dir():
                continue
            checkpoint_path = session_dir / "checkpoint.json"
            if not checkpoint_path.is_file():
                continue
            try:
                cp = self._read_checkpoint(checkpoint_path)
                checkpoints.append(cp)
            except (json.JSONDecodeError, KeyError, orjson.JSONDecodeError):
                continue

        # Sort by timestamp descending
        checkpoints.sort(key=lambda c: c.timestamp, reverse=True)
        return checkpoints[:limit]

    def update_analysis(self, session_id: str, analysis: str) -> Checkpoint:
        """Update the analysis text for an existing checkpoint.

        Args:
            session_id: Session ID to update.
            analysis: New analysis text.

        Returns:
            Updated Checkpoint.
        """
        checkpoint = self.load(session_id)
        checkpoint.analysis = analysis

        session_dir = self._dir / checkpoint.session_id
        checkpoint_path = session_dir / "checkpoint.json"
        data = orjson.dumps(checkpoint.to_dict(), option=orjson.OPT_INDENT_2)
        checkpoint_path.write_bytes(data)

        # Also save analysis as standalone markdown
        analysis_path = session_dir / "analysis.md"
        analysis_path.write_text(analysis)

        return checkpoint

    def delete(self, session_id: str) -> bool:
        """Delete a session checkpoint.

        Args:
            session_id: Session ID to delete.

        Returns:
            True if deleted, False if not found.
        """
        import shutil

        checkpoint = self.load(session_id)
        session_dir = self._dir / checkpoint.session_id
        if session_dir.is_dir():
            shutil.rmtree(session_dir)
            return True
        return False

    def get_result(self, session_id: str) -> ProfileResult:
        """Load and return just the ProfileResult for a session.

        Args:
            session_id: Session ID.

        Returns:
            ProfileResult from the checkpoint.
        """
        checkpoint = self.load(session_id)
        return ProfileResult.from_dict(checkpoint.result)

    def get_context(self, session_id: str) -> ProfileContext:
        """Load and return just the ProfileContext for a session.

        Args:
            session_id: Session ID.

        Returns:
            ProfileContext from the checkpoint.
        """
        checkpoint = self.load(session_id)
        return ProfileContext.from_dict(checkpoint.context)

    def _read_checkpoint(self, path: Path) -> Checkpoint:
        """Read a checkpoint from a JSON file.

        Args:
            path: Path to checkpoint.json.

        Returns:
            Deserialized Checkpoint.
        """
        data = orjson.loads(path.read_bytes())
        return Checkpoint.from_dict(data)
