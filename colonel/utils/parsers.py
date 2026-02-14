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

    ncu CSV includes a header row and metric rows per kernel launch.

    Args:
        text: Raw ncu CSV output.

    Returns:
        List of row dicts.
    """
    # ncu sometimes prefixes lines with "==" for info lines; skip those
    lines = [
        line for line in text.strip().splitlines()
        if line.strip() and not line.startswith("==")
    ]
    if not lines:
        return []

    reader = csv.DictReader(lines)
    return [row for row in reader if any(v.strip() for v in row.values() if v)]


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
