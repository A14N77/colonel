"""AnalysisAgent: LLM-powered GPU profiling analysis.

Modeled after KForge's performance analysis agent -- takes profiling data
and returns structured recommendations. Uses Anthropic Claude by default.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from colonel.agent.prompts import (
    COMPARISON_PROMPT_TEMPLATE,
    DEEPER_ANALYSIS_PROMPT,
    SYSTEM_PROMPT,
    build_analysis_prompt,
    format_kernel_section,
)
from colonel.agent.provider import AgentMessage, AnthropicProvider, BaseLLMProvider
from colonel.config.settings import ColonelSettings, get_settings
from colonel.core.result import ProfileResult


@dataclass
class AnalysisResult:
    """Result of an LLM analysis.

    Attributes:
        summary: The full analysis text.
        model: Which model produced the analysis.
        input_tokens: Tokens consumed.
        output_tokens: Tokens generated.
        conversation: Message history for follow-up.
    """

    summary: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    conversation: list[AgentMessage] = field(default_factory=list)


class AnalysisAgent:
    """Agent that analyzes GPU profiling data using an LLM.

    Usage:
        agent = AnalysisAgent()
        result = agent.analyze(profile_result)
        deeper = agent.analyze_deeper(result, "Focus on memory bandwidth")
    """

    def __init__(
        self,
        provider: BaseLLMProvider | None = None,
        settings: ColonelSettings | None = None,
    ) -> None:
        """Initialize the analysis agent.

        Args:
            provider: LLM provider to use. If None, creates AnthropicProvider.
            settings: Optional settings override.
        """
        self._settings = settings or get_settings()
        if provider is not None:
            self._provider = provider
        else:
            self._provider = AnthropicProvider(
                api_key=self._settings.anthropic_api_key,
                model=self._settings.anthropic_model,
            )
        self._conversation: list[AgentMessage] = []

    def analyze(
        self,
        result: ProfileResult,
        *,
        source_code: str = "",
    ) -> AnalysisResult:
        """Analyze a profiling result and return recommendations.

        Args:
            result: The ProfileResult to analyze.
            source_code: Optional source code for context.

        Returns:
            AnalysisResult with the LLM's assessment.
        """
        prompt = build_analysis_prompt(result.to_dict(), source_code)

        user_msg = AgentMessage(role="user", content=prompt)
        self._conversation = [user_msg]

        response = self._provider.chat(
            self._conversation,
            system=SYSTEM_PROMPT,
            max_tokens=self._settings.agent_max_tokens,
        )

        assistant_msg = AgentMessage(role="assistant", content=response.content)
        self._conversation.append(assistant_msg)

        return AnalysisResult(
            summary=response.content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            conversation=list(self._conversation),
        )

    def analyze_deeper(
        self,
        previous: AnalysisResult,
        profile_result: ProfileResult,
        *,
        additional_context: str = "",
    ) -> AnalysisResult:
        """Perform a deeper follow-up analysis.

        Uses the prior conversation for context and asks the LLM
        to dig deeper into the identified bottlenecks.

        Args:
            previous: The prior analysis result.
            profile_result: The original profiling data.
            additional_context: Extra context or specific questions.

        Returns:
            AnalysisResult with deeper analysis.
        """
        prompt = DEEPER_ANALYSIS_PROMPT.format(
            previous_analysis=previous.summary,
            original_data=json.dumps(profile_result.to_dict(), indent=2, default=str)[:8000],
            additional_context=additional_context or "Provide more detailed analysis.",
        )

        # Continue the conversation
        self._conversation = list(previous.conversation)
        user_msg = AgentMessage(role="user", content=prompt)
        self._conversation.append(user_msg)

        response = self._provider.chat(
            self._conversation,
            system=SYSTEM_PROMPT,
            max_tokens=self._settings.agent_max_tokens,
        )

        assistant_msg = AgentMessage(role="assistant", content=response.content)
        self._conversation.append(assistant_msg)

        return AnalysisResult(
            summary=response.content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            conversation=list(self._conversation),
        )

    def compare(
        self,
        result_a: ProfileResult,
        result_b: ProfileResult,
        *,
        name_a: str = "Run A",
        name_b: str = "Run B",
    ) -> AnalysisResult:
        """Compare two profiling runs.

        Args:
            result_a: First profiling result.
            result_b: Second profiling result.
            name_a: Label for the first run.
            name_b: Label for the second run.

        Returns:
            AnalysisResult with comparison.
        """
        dict_a = result_a.to_dict()
        dict_b = result_b.to_dict()

        def _format_summary(d: dict) -> str:
            lines = [
                f"- Wall time: {d.get('wall_time_s', 0):.3f}s",
                f"- GPU time: {d.get('gpu_time_us', 0):.1f} us",
                f"- Kernel time: {d.get('kernel_time_us', 0):.1f} us",
                f"- Transfer time: {d.get('transfer_time_us', 0):.1f} us",
                f"- Kernels: {len(d.get('kernels', []))}",
                "",
                format_kernel_section(d.get("kernels", [])),
            ]
            return "\n".join(lines)

        prompt = COMPARISON_PROMPT_TEMPLATE.format(
            name_a=name_a,
            timestamp_a=dict_a.get("timestamp", ""),
            summary_a=_format_summary(dict_a),
            name_b=name_b,
            timestamp_b=dict_b.get("timestamp", ""),
            summary_b=_format_summary(dict_b),
        )

        user_msg = AgentMessage(role="user", content=prompt)
        self._conversation = [user_msg]

        response = self._provider.chat(
            self._conversation,
            system=SYSTEM_PROMPT,
            max_tokens=self._settings.agent_max_tokens,
        )

        assistant_msg = AgentMessage(role="assistant", content=response.content)
        self._conversation.append(assistant_msg)

        return AnalysisResult(
            summary=response.content,
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            conversation=list(self._conversation),
        )
