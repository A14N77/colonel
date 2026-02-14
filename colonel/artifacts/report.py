"""Report generator: produces markdown artifacts from profiling data.

Inspired by ParaCodex's artifact-driven approach -- produces structured
markdown reports (profile_summary.md, bottlenecks.md, recommendations.md)
that serve as both human-readable output and input for further analysis.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from colonel.core.result import ProfileResult


def _get_template_env() -> Environment:
    """Create a Jinja2 environment loading templates from the package."""
    return Environment(
        loader=PackageLoader("colonel.artifacts", "templates"),
        autoescape=select_autoescape([]),
        trim_blocks=True,
        lstrip_blocks=True,
    )


def render_profile_summary(
    result: ProfileResult,
    *,
    name: str = "",
) -> str:
    """Render a profile summary report.

    Args:
        result: The profiling result.
        name: Optional run name.

    Returns:
        Rendered markdown string.
    """
    env = _get_template_env()
    template = env.get_template("profile_summary.md.j2")

    return template.render(
        session_id=result.session_id,
        name=name,
        timestamp=result.timestamp,
        evaluator_name=result.evaluator_name,
        wall_time_s=result.wall_time_s,
        gpu_time_us=result.gpu_time_us,
        kernel_time_us=result.kernel_time_us,
        transfer_time_us=result.transfer_time_us,
        api_time_us=result.api_time_us,
        total_invocations=result.total_invocations,
        kernels=[k.to_dict() for k in result.top_kernels],
        transfers=[t.to_dict() for t in result.transfers],
        hardware=result.hardware.to_dict(),
    )


def render_bottlenecks(result: ProfileResult) -> str:
    """Render a bottleneck analysis report.

    Args:
        result: The profiling result (with bottlenecks populated).

    Returns:
        Rendered markdown string.
    """
    env = _get_template_env()
    template = env.get_template("bottlenecks.md.j2")

    return template.render(
        session_id=result.session_id,
        timestamp=result.timestamp,
        bottlenecks=[b.to_dict() for b in result.bottlenecks],
    )


def render_recommendations(
    result: ProfileResult,
    analysis: str,
    *,
    model: str = "",
) -> str:
    """Render a recommendations report.

    Args:
        result: The profiling result.
        analysis: Agent analysis text.
        model: Model name that generated the analysis.

    Returns:
        Rendered markdown string.
    """
    env = _get_template_env()
    template = env.get_template("recommendations.md.j2")

    return template.render(
        session_id=result.session_id,
        timestamp=result.timestamp,
        analysis=analysis,
        model=model,
    )


def save_reports(
    result: ProfileResult,
    output_dir: Path,
    *,
    name: str = "",
    analysis: str = "",
    model: str = "",
) -> list[Path]:
    """Generate and save all report artifacts to a directory.

    Args:
        result: The profiling result.
        output_dir: Directory to save reports in.
        name: Optional run name.
        analysis: Optional agent analysis text.
        model: Optional model name.

    Returns:
        List of paths to the saved report files.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []

    # Profile summary (always generated)
    summary_path = output_dir / "profile_summary.md"
    summary_path.write_text(render_profile_summary(result, name=name))
    saved.append(summary_path)

    # Bottlenecks (if any)
    if result.bottlenecks:
        bottlenecks_path = output_dir / "bottlenecks.md"
        bottlenecks_path.write_text(render_bottlenecks(result))
        saved.append(bottlenecks_path)

    # Recommendations (if analysis is available)
    if analysis:
        recs_path = output_dir / "recommendations.md"
        recs_path.write_text(
            render_recommendations(result, analysis, model=model)
        )
        saved.append(recs_path)

    return saved
