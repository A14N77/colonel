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


# Known flavor names. Evaluator knobs and env for each flavor live in
# one place so --flavor stays declarative.
_FLAVORS: dict[str, dict[str, object]] = {
    "generic": {
        "env": {},
        "metadata": {},
    },
    "vllm": {
        # vLLM V1 runs the GPU in an EngineCore subprocess. collective_rpc
        # needs insecure-serialization to ship profiler start/stop
        # closures; the flavor sets it so users don't have to.
        "env": {
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_USE_V1": "1",
            "COLONEL_VLLM_ENABLE_NVTX": "1",
        },
        "metadata": {
            "flavor": "vllm",
            # ncu defaults for vLLM: single `--section SpeedOfLight`
            # (1-2 kernel-replay passes), which is what the spike
            # proved works end-to-end. `--set full` fails on vLLM
            # because multiple passes in kernel or application replay
            # hit determinism issues on the CUDA-graph-captured decode.
            "ncu_section": "SpeedOfLight",
            # profile-from-start=off makes ncu wait for cudaProfilerStart,
            # which is what profile_region() calls inside the worker.
            "ncu_profile_from_start": "off",
            # nsys uses capture-range+capture-range-end for the same
            # semantics; nvtx trace is additive.
            "nsys_capture_range": "cudaProfilerApi",
            "nsys_enable_nvtx": True,
        },
    },
}


@profile_app.command("run")
def run(
    command: str = typer.Argument(help="Command to profile (e.g. './my_kernel')."),
    args: list[str] = typer.Argument(default=None, help="Arguments for the command."),
    target: str = typer.Option("local", "--target", "-t", help="Target: 'local' or 'ssh://user@host'."),
    evaluator: str = typer.Option(
        "auto", "--evaluator", "-e", help="Evaluator: 'nsys', 'ncu', or 'auto'."
    ),
    flavor: str = typer.Option(
        "generic", "--flavor", "-f",
        help="Workload flavor: 'generic' or 'vllm'. 'vllm' sets "
             "env + ncu/nsys knobs needed to profile real vLLM V1.",
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
        colonel profile run --flavor vllm -- python my_vllm_script.py
    """
    _run_profile(
        command=command,
        args=args or [],
        target=target,
        evaluator=evaluator,
        flavor=flavor,
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
    flavor: str = "generic",
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
        flavor: Workload flavor ('generic', 'vllm'). See _FLAVORS above.
    """
    from rich.progress import Progress, SpinnerColumn, TextColumn

    from colonel.artifacts.report import save_reports
    from colonel.core.context import ProfileContext
    from colonel.core.executor import Executor
    from colonel.core.session import SessionManager

    print_header(f"Colonel Profile: {command}")

    flavor_cfg = _FLAVORS.get(flavor)
    if flavor_cfg is None:
        print_error(
            f"Unknown flavor '{flavor}'. Known flavors: {', '.join(_FLAVORS)}."
        )
        raise typer.Exit(2)

    if flavor == "vllm":
        print_info(
            "Flavor 'vllm' — setting VLLM_ALLOW_INSECURE_SERIALIZATION=1, "
            "VLLM_USE_V1=1, and profiler capture-range=cudaProfilerApi. "
            "Your script must bracket the region to profile with "
            "`colonel.profiling.vllm.profile_region(llm)`."
        )

    ctx = ProfileContext(
        command=command,
        args=args,
        target=target,
        evaluator=evaluator,
        name=name,
        working_dir=working_dir,
        ssh_key=ssh_key,
        env=dict(flavor_cfg["env"]),  # type: ignore[arg-type]
        metadata=dict(flavor_cfg["metadata"]),  # type: ignore[arg-type]
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
