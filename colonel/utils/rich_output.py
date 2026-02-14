"""Rich console output utilities for the colonel CLI.

Provides styled tables, panels, and progress indicators
using the Rich library.
"""

from __future__ import annotations

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

# Colonel theme
COLONEL_THEME = Theme({
    "info": "cyan",
    "success": "green",
    "warning": "yellow",
    "error": "red bold",
    "metric": "bright_blue",
    "kernel": "magenta",
    "header": "bold bright_white",
})

console = Console(theme=COLONEL_THEME)


def print_header(text: str) -> None:
    """Print a styled header."""
    console.print()
    console.print(Panel(Text(text, style="header"), border_style="bright_blue"))


def print_success(text: str) -> None:
    """Print a success message."""
    console.print(f"[success]{text}[/success]")


def print_warning(text: str) -> None:
    """Print a warning message."""
    console.print(f"[warning]{text}[/warning]")


def print_error(text: str) -> None:
    """Print an error message."""
    console.print(f"[error]{text}[/error]")


def print_info(text: str) -> None:
    """Print an info message."""
    console.print(f"[info]{text}[/info]")


def print_profile_summary(result_dict: dict[str, Any]) -> None:
    """Print a rich profile summary table.

    Args:
        result_dict: Serialized ProfileResult.
    """
    # Overview table
    table = Table(title="Profile Summary", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="bright_white", justify="right")

    table.add_row("Wall Time", f"{result_dict.get('wall_time_s', 0):.3f}s")
    table.add_row("GPU Time", f"{result_dict.get('gpu_time_us', 0):.1f} us")
    table.add_row("Kernel Time", f"{result_dict.get('kernel_time_us', 0):.1f} us")
    table.add_row("Transfer Time", f"{result_dict.get('transfer_time_us', 0):.1f} us")
    table.add_row("API Overhead", f"{result_dict.get('api_time_us', 0):.1f} us")
    table.add_row("Evaluator", result_dict.get("evaluator_name", "unknown"))

    console.print(table)


def print_kernel_table(kernels: list[dict[str, Any]]) -> None:
    """Print a rich kernel summary table.

    Args:
        kernels: List of kernel dicts from ProfileResult.
    """
    if not kernels:
        print_info("No kernel data collected.")
        return

    table = Table(title="GPU Kernels", show_header=True)
    table.add_column("#", style="dim", width=3)
    table.add_column("Kernel", style="kernel", max_width=50)
    table.add_column("Duration (us)", style="metric", justify="right")
    table.add_column("Invocations", justify="right")
    table.add_column("Avg (us)", style="metric", justify="right")
    table.add_column("Occupancy", justify="right")

    # Sort by duration descending
    sorted_kernels = sorted(kernels, key=lambda k: k.get("duration_us", 0), reverse=True)

    for i, k in enumerate(sorted_kernels[:15], 1):
        name = k.get("name", "unknown")
        if len(name) > 50:
            name = name[:47] + "..."
        table.add_row(
            str(i),
            name,
            f"{k.get('duration_us', 0):.1f}",
            str(k.get("invocations", 0)),
            f"{k.get('avg_duration_us', 0):.1f}",
            f"{k.get('occupancy_pct', 0):.1f}%",
        )

    console.print(table)


def print_transfer_table(transfers: list[dict[str, Any]]) -> None:
    """Print a rich memory transfer table.

    Args:
        transfers: List of transfer dicts from ProfileResult.
    """
    if not transfers:
        return

    table = Table(title="Memory Transfers", show_header=True)
    table.add_column("Direction", style="cyan")
    table.add_column("Size (MB)", style="metric", justify="right")
    table.add_column("Duration (us)", style="metric", justify="right")
    table.add_column("Throughput (GB/s)", style="metric", justify="right")

    for t in transfers:
        size_mb = t.get("size_bytes", 0) / (1024 * 1024)
        table.add_row(
            t.get("direction", "?"),
            f"{size_mb:.2f}",
            f"{t.get('duration_us', 0):.1f}",
            f"{t.get('throughput_gbps', 0):.2f}",
        )

    console.print(table)


def print_analysis(analysis_text: str) -> None:
    """Print agent analysis as rendered markdown.

    Args:
        analysis_text: Markdown-formatted analysis from the agent.
    """
    console.print()
    console.print(Panel(
        Markdown(analysis_text),
        title="Colonel Analysis",
        border_style="bright_blue",
        padding=(1, 2),
    ))


def print_session_list(sessions: list[dict[str, Any]]) -> None:
    """Print a table of saved sessions.

    Args:
        sessions: List of checkpoint dicts.
    """
    if not sessions:
        print_info("No sessions found. Run `colonel profile` to create one.")
        return

    table = Table(title="Saved Sessions", show_header=True)
    table.add_column("ID", style="cyan", width=10)
    table.add_column("Name", style="bright_white")
    table.add_column("Date", style="dim")
    table.add_column("Evaluator", style="dim")
    table.add_column("GPU Time (us)", style="metric", justify="right")

    for s in sessions:
        result = s.get("result", {})
        table.add_row(
            s.get("session_id", "")[:8],
            s.get("name", "") or "-",
            s.get("timestamp", "")[:19],
            result.get("evaluator_name", "?"),
            f"{result.get('gpu_time_us', 0):.1f}",
        )

    console.print(table)


def print_comparison(
    name_a: str,
    name_b: str,
    result_a: dict[str, Any],
    result_b: dict[str, Any],
) -> None:
    """Print a side-by-side comparison of two profiling runs.

    Args:
        name_a: Label for run A.
        name_b: Label for run B.
        result_a: Serialized ProfileResult for run A.
        result_b: Serialized ProfileResult for run B.
    """
    table = Table(title=f"Comparison: {name_a} vs {name_b}", show_header=True)
    table.add_column("Metric", style="cyan")
    table.add_column(name_a, style="bright_white", justify="right")
    table.add_column(name_b, style="bright_white", justify="right")
    table.add_column("Delta", justify="right")

    metrics = [
        ("Wall Time (s)", "wall_time_s", ".3f"),
        ("GPU Time (us)", "gpu_time_us", ".1f"),
        ("Kernel Time (us)", "kernel_time_us", ".1f"),
        ("Transfer Time (us)", "transfer_time_us", ".1f"),
        ("API Overhead (us)", "api_time_us", ".1f"),
    ]

    for label, key, fmt in metrics:
        val_a = result_a.get(key, 0)
        val_b = result_b.get(key, 0)
        delta = val_b - val_a
        pct = ((val_b - val_a) / val_a * 100) if val_a > 0 else 0

        delta_style = "green" if delta < 0 else ("red" if delta > 0 else "dim")
        delta_str = f"[{delta_style}]{delta:{fmt}} ({pct:+.1f}%)[/{delta_style}]"

        table.add_row(
            label,
            f"{val_a:{fmt}}",
            f"{val_b:{fmt}}",
            delta_str,
        )

    console.print(table)
