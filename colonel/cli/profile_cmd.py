"""Profile command: run GPU profilers and collect data."""

from __future__ import annotations

from pathlib import Path

import typer

from colonel.utils.rich_output import (
    console,
    print_error,
    print_header,
    print_info,
    print_kernel_table,
    print_profile_summary,
    print_success,
    print_transfer_table,
)

profile_app = typer.Typer(no_args_is_help=True)


@profile_app.command("run")
def run(
    command: str = typer.Argument(help="Command to profile (e.g. './my_kernel')."),
    args: list[str] = typer.Argument(default=None, help="Arguments for the command."),
    target: str = typer.Option("local", "--target", "-t", help="Target: 'local' or 'ssh://user@host'."),
    evaluator: str = typer.Option(
        "auto", "--evaluator", "-e", help="Evaluator: 'nsys', 'ncu', or 'auto'."
    ),
    name: str = typer.Option("", "--name", "-n", help="Human-readable label for this run."),
    no_analyze: bool = typer.Option(
        False, "--no-analyze", help="Skip AI analysis after profiling."
    ),
    working_dir: str = typer.Option(".", "--cwd", "-C", help="Working directory for execution."),
    ssh_key: str = typer.Option("", "--ssh-key", help="Path to SSH private key file."),
) -> None:
    """Profile a GPU application.

    Runs the specified command under a GPU profiler (nsys or ncu),
    collects metrics, saves a session checkpoint, and optionally
    runs AI-powered analysis.

    Examples:
        colonel profile run ./my_kernel
        colonel profile run "python train.py" --name baseline --evaluator ncu
        colonel profile run ./app --target ssh://user@gpu-server
        colonel profile run ./app --target ssh://user@host --ssh-key ~/.ssh/id_rsa
    """
    _run_profile(
        command=command,
        args=args or [],
        target=target,
        evaluator=evaluator,
        name=name,
        no_analyze=no_analyze,
        working_dir=working_dir,
        ssh_key=ssh_key or None,
    )


@profile_app.command("detect")
def detect(
    target: str = typer.Option("local", "--target", "-t", help="Target to check."),
) -> None:
    """Detect available profiling tools on the target."""
    from colonel.core.context import ProfileContext
    from colonel.core.executor import Executor

    ctx = ProfileContext(command="", target=target)
    executor = Executor()
    tools = executor.detect_tools(ctx)

    if tools:
        print_success(f"Available profiling tools: {', '.join(tools)}")
    else:
        print_error(
            "No GPU profiling tools found. "
            "Install NVIDIA Nsight Systems (nsys) or Nsight Compute (ncu)."
        )


def _run_profile(
    command: str,
    args: list[str],
    target: str,
    evaluator: str,
    name: str,
    no_analyze: bool,
    working_dir: str = ".",
    ssh_key: str | None = None,
) -> None:
    """Internal implementation for the profile command.

    Args:
        command: Command to profile.
        args: Command arguments.
        target: Execution target.
        evaluator: Profiler to use.
        name: Run label.
        no_analyze: Whether to skip analysis.
        working_dir: Working directory.
        ssh_key: Optional path to SSH private key file.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from colonel.artifacts.report import save_reports
    from colonel.core.context import ProfileContext
    from colonel.core.executor import Executor
    from colonel.core.session import SessionManager

    print_header(f"Colonel Profile: {command}")

    ctx = ProfileContext(
        command=command,
        args=args,
        target=target,
        evaluator=evaluator,
        name=name,
        working_dir=working_dir,
        ssh_key=ssh_key,
    )

    executor = Executor()
    session_mgr = SessionManager()

    # Run the profiler with progress indicator
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Profiling...", total=None)
        result = executor.run(ctx)
        progress.update(task, description="Profiling complete.")

    if not result.success:
        print_error(f"Profiling failed: {'; '.join(result.errors)}")
        # Still save the result for debugging
        session_mgr.save(ctx, result, name=name)
        raise typer.Exit(1)

    # Display results
    result_dict = result.to_dict()
    console.print()
    print_profile_summary(result_dict)
    print_kernel_table(result_dict.get("kernels", []))
    print_transfer_table(result_dict.get("transfers", []))

    # Run AI analysis if not skipped
    analysis_text = ""
    model_name = ""
    if not no_analyze:
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Running AI analysis...", total=None)
                from colonel.agent.analyzer import AnalysisAgent

                agent = AnalysisAgent()
                analysis = agent.analyze(result)
                analysis_text = analysis.summary
                model_name = analysis.model
                progress.update(task, description="Analysis complete.")

            from colonel.utils.rich_output import print_analysis
            print_analysis(analysis_text)

        except Exception as exc:
            from colonel.utils.rich_output import print_warning
            print_warning(
                f"AI analysis skipped: {exc}\n"
                "Set your API key with: colonel config set anthropic_api_key <key>"
            )

    # Save session
    checkpoint = session_mgr.save(
        ctx, result,
        name=name,
        analysis=analysis_text,
    )

    # Save artifact reports
    report_dir = Path(session_mgr.sessions_dir) / checkpoint.session_id
    saved_reports = save_reports(
        result, report_dir,
        name=name,
        analysis=analysis_text,
        model=model_name,
    )

    console.print()
    print_success(f"Session saved: {checkpoint.short_id}")
    print_info(f"Reports: {report_dir}")
    for rp in saved_reports:
        print_info(f"  - {rp.name}")
