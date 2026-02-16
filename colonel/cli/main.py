"""Colonel CLI entry point.

The main Typer application that registers all subcommands.
"""

from __future__ import annotations

import typer

from colonel.__version__ import __version__

app = typer.Typer(
    name="colonel",
    help="Agentic CLI tool for GPU profiling and analysis.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        typer.echo(f"colonel {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
) -> None:
    """Colonel: Agentic CLI for GPU profiling and analysis."""
    pass


# Register subcommands
from colonel.cli.analyze_cmd import analyze_app  # noqa: E402
from colonel.cli.config_cmd import config_app  # noqa: E402
from colonel.cli.profile_cmd import profile_app  # noqa: E402
from colonel.cli.session_cmd import session_app  # noqa: E402
from colonel.cli.setup_cmd import setup_app  # noqa: E402

app.add_typer(profile_app, name="profile", help="Profile GPU kernels and applications.")
app.add_typer(analyze_app, name="analyze", help="Analyze profiling results with AI.")
app.add_typer(session_app, name="session", help="Manage profiling sessions.")
app.add_typer(config_app, name="config", help="Manage colonel configuration.")
app.add_typer(setup_app, name="setup", help="Interactive environment setup wizard.")


# Also register `colonel profile` as a direct command (not just subgroup)
# so `colonel profile ./my_kernel` works without a subcommand
@app.command("run")
def run_profile(
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
    """Profile a GPU application and optionally analyze results.

    This is a shortcut for `colonel profile run`.
    """
    from colonel.cli.profile_cmd import _run_profile
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
