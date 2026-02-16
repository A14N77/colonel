"""Parsers for profiler output formats (CSV, structured text)."""

from __future__ import annotations

import csv
import re
from typing import Any


def parse_csv_output(text: str) -> list[dict[str, str]]:
    """Parse CSV text into a list of row dicts.

    Handles nsys stats and ncu CSV output. Skips blank lines and
    lines that look like headers/separators.

    Args:
        text: Raw CSV text.

    Returns:
        List of dicts, one per row, with header keys.
    """
    # Strip leading/trailing whitespace and skip empty lines
    lines = [line for line in text.strip().splitlines() if line.strip()]
    if not lines:
        return []

    # Find the header line (first line that looks like a CSV header)
    reader = csv.DictReader(lines)
    rows = []
    for row in reader:
        # Skip rows where all values are empty
        if any(v.strip() for v in row.values() if v):
            rows.append(row)
    return rows


def parse_nsys_stats_csv(text: str) -> list[dict[str, str]]:
    """Parse nsys stats CSV output.

    nsys stats outputs multiple CSV tables separated by blank lines
    and table headers. This function extracts all rows from all tables.

    Args:
        text: Raw nsys stats output.

    Returns:
        List of row dicts.
    """
    return parse_csv_output(text)


def parse_ncu_csv(text: str) -> list[dict[str, str]]:
    """Parse ncu --csv output.

    ncu --csv --set full outputs a *long-format* CSV where each row is
    one metric for one kernel launch.  Columns include:
        ID, Kernel Name, Block Size, Grid Size, Section Name,
        Metric Name, Metric Unit, Metric Value, …

    We filter out profiler info lines (==PROF==) and application stdout
    by looking for the actual CSV header (starts with ``"ID"``).

    Args:
        text: Raw ncu CSV output (may include app stdout and ==PROF== lines).

    Returns:
        List of row dicts with the CSV column names as keys.
    """
    raw_lines = text.strip().splitlines()

    # Find the real CSV header – it starts with "ID" (quoted or not).
    header_idx = -1
    for idx, line in enumerate(raw_lines):
        stripped = line.strip().strip('"')
        if stripped.startswith("ID,") or stripped.startswith("ID\""):
            header_idx = idx
            break

    if header_idx < 0:
        return []

    # Take the header + all subsequent non-empty, non-profiler lines
    csv_lines = [raw_lines[header_idx]]
    for line in raw_lines[header_idx + 1 :]:
        s = line.strip()
        if not s or s.startswith("=="):
            continue
        csv_lines.append(line)

    reader = csv.DictReader(csv_lines)
    return [row for row in reader if any(v.strip() for v in row.values() if v)]


def pivot_ncu_rows(
    rows: list[dict[str, str]],
) -> dict[str, list[dict[str, str]]]:
    """Pivot long-format ncu CSV into per-launch metric dicts grouped by kernel.

    Each input row has columns like ``Kernel Name``, ``Metric Name``,
    ``Metric Value``, ``Block Size``, ``Grid Size``, etc.  We pivot so
    that each launch (identified by ID + Kernel Name) becomes a single
    dict mapping metric names to their values, plus the per-row columns
    that are the same across all metrics for a launch.

    Returns:
        ``{kernel_name: [launch_dict, …]}`` where each *launch_dict*
        maps metric names (e.g. ``"Duration"``, ``"Achieved Occupancy"``)
        to string values, and also contains ``"Block Size"``,
        ``"Grid Size"`` etc.
    """
    # Group rows by (ID, Kernel Name) to get one group per launch
    launches: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        kernel = row.get("Kernel Name", "")
        launch_id = row.get("ID", "")
        if not kernel:
            continue

        key = (launch_id, kernel)
        if key not in launches:
            # Seed with the non-metric columns that are constant per launch
            launches[key] = {
                "Kernel Name": kernel,
                "Block Size": row.get("Block Size", ""),
                "Grid Size": row.get("Grid Size", ""),
                "Device": row.get("Device", ""),
                "CC": row.get("CC", ""),
                "Context": row.get("Context", ""),
                "Stream": row.get("Stream", ""),
            }

        metric_name = row.get("Metric Name", "").strip()
        metric_value = row.get("Metric Value", "").strip()
        metric_unit = row.get("Metric Unit", "").strip()
        section = row.get("Section Name", "").strip()

        if metric_name and metric_value:
            # Store as "Metric Name" -> value (primary lookup)
            launches[key][metric_name] = metric_value
            # Also store with unit for Duration-type metrics
            if metric_unit and metric_unit not in ("%",):
                launches[key][f"{metric_name}__unit"] = metric_unit
            # Store section-qualified name for disambiguation
            if section:
                launches[key][f"{section}::{metric_name}"] = metric_value

    # Group by kernel name
    result: dict[str, list[dict[str, str]]] = {}
    for (_lid, kernel), data in launches.items():
        result.setdefault(kernel, []).append(data)
    return result


def safe_float(value: str | Any, default: float = 0.0) -> float:
    """Safely convert a string to float.

    Handles commas in numbers, percentage signs, and units.

    Args:
        value: String value to convert.
        default: Default if conversion fails.

    Returns:
        Float value.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        return default
    # Remove commas, percentage signs, and common units
    cleaned = value.strip().replace(",", "").rstrip("%")
    # Remove unit suffixes like "us", "ms", "s", "GB/s", etc.
    cleaned = re.sub(r"\s*(us|ms|s|GB/s|MB/s|KB|MB|GB|B)\s*$", "", cleaned)
    try:
        return float(cleaned)
    except (ValueError, TypeError):
        return default


def safe_int(value: str | Any, default: int = 0) -> int:
    """Safely convert a string to int.

    Args:
        value: String value to convert.
        default: Default if conversion fails.

    Returns:
        Int value.
    """
    try:
        return int(safe_float(value, float(default)))
    except (ValueError, TypeError):
        return default


def parse_duration_string(text: str) -> float:
    """Parse a duration string like '1.234 ms' or '5.678 us' to microseconds.

    Args:
        text: Duration string.

    Returns:
        Duration in microseconds.
    """
    text = text.strip().lower()
    match = re.match(r"([\d.,]+)\s*(us|ms|s|ns)", text)
    if not match:
        return safe_float(text)

    value = safe_float(match.group(1))
    unit = match.group(2)

    multipliers = {
        "ns": 0.001,
        "us": 1.0,
        "ms": 1000.0,
        "s": 1_000_000.0,
    }
    return value * multipliers.get(unit, 1.0)
