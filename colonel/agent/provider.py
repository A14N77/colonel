"""LLM provider interface for the analysis agent.

Abstracts the LLM API so the analyzer doesn't depend directly on
a specific provider. Anthropic (Claude) is the primary implementation.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any


@dataclass
class AgentMessage:
    """A message in the agent conversation.

    Attributes:
        role: "system", "user", or "assistant".
        content: Message text.
    """

    role: str
    content: str


@dataclass
class AgentResponse:
    """Response from the LLM provider.

    Attributes:
        content: The response text.
        model: Model that generated the response.
        input_tokens: Tokens in the input.
        output_tokens: Tokens in the output.
        metadata: Any provider-specific metadata.
    """

    content: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseLLMProvider(abc.ABC):
    """Abstract base for LLM providers."""

    @abc.abstractmethod
    def chat(
        self,
        messages: list[AgentMessage],
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Send a chat completion request.

        Args:
            messages: Conversation messages.
            system: System prompt.
            max_tokens: Maximum response tokens.

        Returns:
            AgentResponse with the model's reply.
        """
        ...


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Claude provider.

    Uses the anthropic Python SDK to call Claude models.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize the Anthropic provider.

        Args:
            api_key: Anthropic API key.
            model: Model name (e.g. "claude-sonnet-4-20250514").
        """
        if not api_key:
            raise ValueError(
                "Anthropic API key is required. Set COLONEL_ANTHROPIC_API_KEY "
                "or run: colonel config set anthropic_api_key <your-key>"
            )
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model

    def chat(
        self,
        messages: list[AgentMessage],
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Send a chat request to Claude.

        Args:
            messages: Conversation messages (role + content).
            system: System prompt.
            max_tokens: Maximum response tokens.

        Returns:
            AgentResponse with Claude's reply.
        """
        # Convert to Anthropic format
        api_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
            if msg.role in ("user", "assistant")
        ]

        kwargs: dict[str, Any] = {
            "model": self._model,
            "max_tokens": max_tokens,
            "messages": api_messages,
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        content = ""
        for block in response.content:
            if hasattr(block, "text"):
                content += block.text

        return AgentResponse(
            content=content,
            model=response.model,
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
        )
