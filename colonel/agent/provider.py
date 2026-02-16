"""LLM provider interface for the analysis agent.

Abstracts the LLM API so the analyzer doesn't depend directly on
a specific provider.  Ships with:

* **AnthropicProvider** – Claude via the ``anthropic`` SDK.
* **OpenAICompatibleProvider** – any OpenAI-compatible endpoint
  (NVIDIA NIM / Nemotron, HuggingFace Inference, OpenRouter, etc.)
  via the ``openai`` SDK.
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


# ---------------------------------------------------------------------------
# Well-known NVIDIA NIM / HuggingFace presets
# ---------------------------------------------------------------------------

NVIDIA_NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
HUGGINGFACE_BASE_URL = "https://router.huggingface.co/v1"

NVIDIA_DEFAULT_MODEL = "nvidia/llama-3.3-nemotron-super-49b-v1.5"
HUGGINGFACE_DEFAULT_MODEL = "nvidia/llama-3.1-nemotron-70b-instruct"


class OpenAICompatibleProvider(BaseLLMProvider):
    """Provider for any OpenAI-compatible chat completions endpoint.

    Works out-of-the-box with:
    * **NVIDIA NIM** (``https://integrate.api.nvidia.com/v1``)
    * **HuggingFace Inference** (``https://router.huggingface.co/v1``)
    * **OpenRouter**, **Together AI**, **DeepInfra**, etc.

    Uses the ``openai`` Python SDK under the hood.
    """

    def __init__(
        self,
        api_key: str,
        model: str = NVIDIA_DEFAULT_MODEL,
        base_url: str = NVIDIA_NIM_BASE_URL,
    ) -> None:
        if not api_key:
            raise ValueError(
                "API key is required for the OpenAI-compatible provider. "
                "Set the appropriate key via colonel config "
                "(e.g. colonel config set nvidia_api_key <key>)."
            )
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "The 'openai' package is required for NVIDIA / HuggingFace providers. "
                "Install it with: pip install openai"
            )

        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._base_url = base_url

    def chat(
        self,
        messages: list[AgentMessage],
        *,
        system: str = "",
        max_tokens: int = 4096,
    ) -> AgentResponse:
        """Send a chat request via OpenAI-compatible endpoint."""
        api_messages: list[dict[str, str]] = []
        if system:
            api_messages.append({"role": "system", "content": system})

        for msg in messages:
            if msg.role in ("user", "assistant"):
                api_messages.append({"role": msg.role, "content": msg.content})

        response = self._client.chat.completions.create(
            model=self._model,
            messages=api_messages,  # type: ignore[arg-type]
            max_tokens=max_tokens,
        )

        choice = response.choices[0] if response.choices else None
        content = choice.message.content or "" if choice else ""

        input_tokens = 0
        output_tokens = 0
        if response.usage:
            input_tokens = response.usage.prompt_tokens or 0
            output_tokens = response.usage.completion_tokens or 0

        return AgentResponse(
            content=content,
            model=response.model or self._model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            metadata={"base_url": self._base_url},
        )


def get_provider(
    provider_name: str,
    *,
    api_key: str,
    model: str = "",
    base_url: str = "",
) -> BaseLLMProvider:
    """Factory: build the right provider from a short name.

    Args:
        provider_name: One of ``"anthropic"``, ``"nvidia"``, ``"huggingface"``,
            or ``"openai"`` (custom endpoint).
        api_key: The API key for this provider.
        model: Model override (uses provider default when empty).
        base_url: Base URL override (only for ``"openai"`` provider).

    Returns:
        A configured :class:`BaseLLMProvider`.
    """
    name = provider_name.strip().lower()

    if name == "anthropic":
        return AnthropicProvider(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
        )

    if name == "nvidia":
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model or NVIDIA_DEFAULT_MODEL,
            base_url=NVIDIA_NIM_BASE_URL,
        )

    if name == "huggingface":
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model or HUGGINGFACE_DEFAULT_MODEL,
            base_url=HUGGINGFACE_BASE_URL,
        )

    if name == "openai":
        if not base_url:
            raise ValueError("base_url is required for the 'openai' provider type.")
        return OpenAICompatibleProvider(
            api_key=api_key,
            model=model or "gpt-4o",
            base_url=base_url,
        )

    raise ValueError(
        f"Unknown provider '{provider_name}'. "
        "Choose from: anthropic, nvidia, huggingface, openai."
    )
