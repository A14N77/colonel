"""Session command: manage profiling session checkpoints."""

from __future__ import annotations

import typer

from colonel.utils.rich_output import (
    console,
    print_comparison,
    print_error,
    print_header,
    print_info,
    print_kernel_table,
    print_profile_summary,
    print_session_list,
    print_success,
    print_transfer_table,
)

session_app = typer.Typer(no_args_is_help=True)


@session_app.command("list")
def list_sessions(
    limit: int = typer.Option(20, "--limit", "-l", help="Maximum sessions to show."),
) -> None:
    """List saved profiling sessions.

    Shows recent sessions with their IDs, names, timestamps, and key metrics.

    Examples:
        colonel session list
        colonel session list --limit 50
    """
    from colonel.core.session import SessionManager

    session_mgr = SessionManager()
    sessions = session_mgr.list_sessions(limit=limit)

    print_header("Saved Sessions")
    print_session_list([s.to_dict() for s in sessions])


@session_app.command("show")
def show_session(
    session_id: str = typer.Argument(help="Session ID (full or prefix)."),
) -> None:
    """Show details of a saved profiling session.

    Displays the full profile summary, kernel table, and any saved analysis.

    Examples:
        colonel session show abc123
    """
    from colonel.core.session import SessionManager

    session_mgr = SessionManager()

    try:
        checkpoint = session_mgr.load(session_id)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    print_header(f"Session: {checkpoint.display_name}")

    result_dict = checkpoint.result
    print_profile_summary(result_dict)
    print_kernel_table(result_dict.get("kernels", []))
    print_transfer_table(result_dict.get("transfers", []))

    if checkpoint.analysis:
        from colonel.utils.rich_output import print_analysis
        print_analysis(checkpoint.analysis)


@session_app.command("compare")
def compare_sessions(
    session_a: str = typer.Argument(help="First session ID."),
    session_b: str = typer.Argument(help="Second session ID."),
    ai_compare: bool = typer.Option(False, "--ai", help="Also run AI comparison analysis."),
) -> None:
    """Compare two profiling sessions side by side.

    Shows metric deltas between two runs. Optionally runs AI analysis
    on the comparison.

    Examples:
        colonel session compare abc123 def456
        colonel session compare abc123 def456 --ai
    """
    from colonel.core.session import SessionManager

    session_mgr = SessionManager()

    try:
        cp_a = session_mgr.load(session_a)
        cp_b = session_mgr.load(session_b)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    print_header(f"Compare: {cp_a.display_name} vs {cp_b.display_name}")
    print_comparison(
        cp_a.display_name,
        cp_b.display_name,
        cp_a.result,
        cp_b.result,
    )

    if ai_compare:
        try:
            from rich.progress import Progress, SpinnerColumn, TextColumn

            from colonel.agent.analyzer import AnalysisAgent
            from colonel.core.result import ProfileResult

            result_a = ProfileResult.from_dict(cp_a.result)
            result_b = ProfileResult.from_dict(cp_b.result)

            agent = AnalysisAgent()

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running AI comparison...", total=None)
                analysis = agent.compare(
                    result_a, result_b,
                    name_a=cp_a.display_name,
                    name_b=cp_b.display_name,
                )
                progress.update(task, description="Comparison complete.")

            from colonel.utils.rich_output import print_analysis
            print_analysis(analysis.summary)

        except Exception as exc:
            from colonel.utils.rich_output import print_warning
            print_warning(f"AI comparison skipped: {exc}")


@session_app.command("delete")
def delete_session(
    session_id: str = typer.Argument(help="Session ID to delete."),
    force: bool = typer.Option(False, "--force", "-f", help="Skip confirmation."),
) -> None:
    """Delete a saved profiling session.

    Examples:
        colonel session delete abc123
        colonel session delete abc123 --force
    """
    from colonel.core.session import SessionManager

    session_mgr = SessionManager()

    try:
        checkpoint = session_mgr.load(session_id)
    except (FileNotFoundError, ValueError) as exc:
        print_error(str(exc))
        raise typer.Exit(1)

    if not force:
        confirm = typer.confirm(
            f"Delete session {checkpoint.display_name} ({checkpoint.session_id})?"
        )
        if not confirm:
            print_info("Cancelled.")
            raise typer.Exit(0)

    if session_mgr.delete(checkpoint.session_id):
        print_success(f"Deleted session {checkpoint.short_id}")
    else:
        print_error("Failed to delete session.")
