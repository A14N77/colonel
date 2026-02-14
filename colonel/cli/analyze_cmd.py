"""Analyze command: run AI-powered analysis on profiling results."""

from __future__ import annotations

from pathlib import Path

import typer

from colonel.utils.rich_output import (
    console,
    print_analysis,
    print_error,
    print_header,
    print_info,
    print_success,
)

analyze_app = typer.Typer(invoke_without_command=True)


@analyze_app.callback(invoke_without_command=True)
def analyze(
    session: str = typer.Option(
        "", "--session", "-s", help="Session ID to analyze (default: latest)."
    ),
    deeper: bool = typer.Option(False, "--deeper", "-d", help="Perform deeper follow-up analysis."),
    with_source: str = typer.Option("", "--with-source", help="Path to source code for context."),
    question: str = typer.Option("", "--question", "-q", help="Specific question for the agent."),
) -> None:
    """Analyze profiling results using AI.

    By default, analyzes the most recent session. Use --session to specify
    a particular session, and --deeper for follow-up analysis.

    Examples:
        colonel analyze
        colonel analyze --session abc123
        colonel analyze --deeper --question "Why is occupancy so low?"
        colonel analyze --with-source ./kernel.cu
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from colonel.agent.analyzer import AnalysisAgent
    from colonel.artifacts.report import save_reports
    from colonel.core.session import SessionManager

    session_mgr = SessionManager()

    # Resolve session
    if session:
        try:
            checkpoint = session_mgr.load(session)
        except (FileNotFoundError, ValueError) as exc:
            print_error(str(exc))
            raise typer.Exit(1)
    else:
        # Use the latest session
        sessions = session_mgr.list_sessions(limit=1)
        if not sessions:
            print_error("No sessions found. Run `colonel profile run` first.")
            raise typer.Exit(1)
        checkpoint = sessions[0]

    print_header(f"Colonel Analyze: {checkpoint.display_name}")

    # Load the profile result
    from colonel.core.result import ProfileResult
    result = ProfileResult.from_dict(checkpoint.result)

    # Load source code if provided
    source_code = ""
    if with_source:
        source_path = Path(with_source)
        if source_path.is_file():
            source_code = source_path.read_text()
            print_info(f"Loaded source: {source_path}")
        else:
            print_error(f"Source file not found: {with_source}")
            raise typer.Exit(1)

    try:
        agent = AnalysisAgent()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            if deeper and checkpoint.analysis:
                task = progress.add_task("Running deeper analysis...", total=None)
                from colonel.agent.analyzer import AnalysisResult
                from colonel.agent.provider import AgentMessage

                previous = AnalysisResult(
                    summary=checkpoint.analysis,
                    conversation=[
                        AgentMessage(role="assistant", content=checkpoint.analysis),
                    ],
                )
                additional = question or ""
                if source_code:
                    additional += f"\n\nSource code:\n```\n{source_code}\n```"

                analysis = agent.analyze_deeper(
                    previous, result,
                    additional_context=additional,
                )
            else:
                task = progress.add_task("Running AI analysis...", total=None)
                analysis = agent.analyze(result, source_code=source_code)

            progress.update(task, description="Analysis complete.")

        # Display
        print_analysis(analysis.summary)

        # Update the session with the new analysis
        session_mgr.update_analysis(checkpoint.session_id, analysis.summary)

        # Save updated reports
        report_dir = Path(session_mgr.sessions_dir) / checkpoint.session_id
        save_reports(
            result, report_dir,
            name=checkpoint.name,
            analysis=analysis.summary,
            model=analysis.model,
        )

        console.print()
        print_success(f"Analysis saved to session {checkpoint.short_id}")
        print_info(f"Tokens: {analysis.input_tokens} in / {analysis.output_tokens} out")

    except Exception as exc:
        print_error(f"Analysis failed: {exc}")
        raise typer.Exit(1)
