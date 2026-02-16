"""Colonel configuration via pydantic-settings.

Settings are loaded from (in priority order):
1. Environment variables prefixed with COLONEL_
2. .colonel/config.json in the current directory
3. ~/.colonel/config.json in the home directory
4. Built-in defaults
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


def _find_config_file() -> Path | None:
    """Look for .colonel/config.json in cwd, then home."""
    candidates = [
        Path.cwd() / ".colonel" / "config.json",
        Path.home() / ".colonel" / "config.json",
    ]
    for path in candidates:
        if path.is_file():
            return path
    return None


def _json_config_source(settings: BaseSettings) -> dict[str, Any]:
    """Load settings from the first config.json found."""
    path = _find_config_file()
    if path is None:
        return {}
    try:
        return json.loads(path.read_text())  # type: ignore[no-any-return]
    except (json.JSONDecodeError, OSError):
        return {}


class ColonelSettings(BaseSettings):
    """Global settings for the colonel CLI."""

    model_config = SettingsConfigDict(
        env_prefix="COLONEL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM provider ---
    llm_provider: str = Field(
        default="anthropic",
        description="LLM provider: 'anthropic', 'nvidia', 'huggingface', or 'openai'.",
    )
    anthropic_api_key: str = Field(
        default="",
        description="Anthropic API key for the analysis agent.",
    )
    anthropic_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Anthropic model to use for analysis.",
    )
    nvidia_api_key: str = Field(
        default="",
        description="NVIDIA NIM API key (build.nvidia.com) for Nemotron models.",
    )
    nvidia_model: str = Field(
        default="nvidia/llama-3.3-nemotron-super-49b-v1.5",
        description="NVIDIA NIM model to use for analysis.",
    )
    huggingface_api_key: str = Field(
        default="",
        description="HuggingFace API token for inference providers.",
    )
    huggingface_model: str = Field(
        default="nvidia/llama-3.1-nemotron-70b-instruct",
        description="HuggingFace model to use for analysis.",
    )
    openai_api_key: str = Field(
        default="",
        description="API key for custom OpenAI-compatible endpoints.",
    )
    openai_base_url: str = Field(
        default="",
        description="Base URL for custom OpenAI-compatible endpoints.",
    )
    openai_model: str = Field(
        default="",
        description="Model name for custom OpenAI-compatible endpoints.",
    )
    agent_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for agent responses.",
    )

    # --- Profiling defaults ---
    default_evaluator: str = Field(
        default="auto",
        description="Default evaluator: 'nsys', 'ncu', or 'auto'.",
    )
    default_target: str = Field(
        default="local",
        description="Default target: 'local' or 'ssh://user@host'.",
    )
    nsys_path: str = Field(
        default="nsys",
        description="Path to the nsys binary.",
    )
    ncu_path: str = Field(
        default="ncu",
        description="Path to the ncu binary.",
    )

    # --- Sessions ---
    sessions_dir: str = Field(
        default=".colonel/sessions",
        description="Directory for storing session checkpoints.",
    )

    # --- Output ---
    rich_output: bool = Field(
        default=True,
        description="Enable rich console output.",
    )

    # Settings sources are handled by the model_config and get_settings()
    # function which merges JSON config file values with env/init sources.


def get_settings(**overrides: Any) -> ColonelSettings:
    """Get settings, merging overrides with config file and env vars."""
    file_values = _json_config_source(ColonelSettings)
    file_values.update(overrides)
    return ColonelSettings(**file_values)


def get_config_dir() -> Path:
    """Return the .colonel directory in cwd, creating it if needed."""
    config_dir = Path.cwd() / ".colonel"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def save_config(key: str, value: str) -> Path:
    """Save a single config key to .colonel/config.json."""
    config_path = get_config_dir() / "config.json"
    data: dict[str, Any] = {}
    if config_path.is_file():
        try:
            data = json.loads(config_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    data[key] = value
    config_path.write_text(json.dumps(data, indent=2) + "\n")
    return config_path
