#!/usr/bin/env python3
"""Show experiment leaderboard from experiments.jsonl.

Reads .claude/coral/experiments/experiments.jsonl and prints a ranked table
sorted by the chosen primary metric. Supports per-version aggregation
(mean ± std across folds) or raw per-fold display.

Usage:
    python scripts/show_experiments.py
    python scripts/show_experiments.py --sort-by unseen_mae --limit 10
    python scripts/show_experiments.py --per-fold
    python scripts/show_experiments.py --format json
    python scripts/show_experiments.py --path /tmp/demo.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import sys
from pathlib import Path

# Columns included in the output table, in order.
# Currently unused at runtime but kept in sync with _HEADERS / _SORT_CHOICES.
_MAE_FIELDS = ("primary_mae", "overall_mae", "seen_mae", "unseen_mae", "q4_mae")

_SORT_CHOICES = (
    "primary_mae",
    "overall_mae",
    "seen_mae",
    "unseen_mae",
    "q4_mae",
    "timestamp",
)

# Column header labels (must stay in sync with _row_cells / _agg_cells).
_HEADERS = [
    "rank",
    "version",
    "primary_mae",
    "overall_mae",
    "seen_mae",
    "unseen_mae",
    "q4_mae",
    "unseen_evt_pct",
    "features_n",
    "commit",
    "timestamp",
    "fold_idx",
]


def _load_rows(jsonl_path: Path) -> list[dict[str, object]]:
    """Load all non-empty JSON lines; silently skip malformed lines."""
    rows: list[dict[str, object]] = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    rows.append(json.loads(line))
    return rows


def _fmt_mae(value: object) -> str:
    """Format a MAE value as '$XX.XX' or '-'."""
    if value is None:
        return "-"
    try:
        return f"${float(str(value)):.2f}"
    except (TypeError, ValueError):
        return "-"


def _fmt_pct(value: object) -> str:
    """Format a fraction (0-1) as 'XX.X%' or '-'."""
    if value is None:
        return "-"
    try:
        return f"{float(str(value)) * 100:.1f}%"
    except (TypeError, ValueError):
        return "-"


def _fmt_str(value: object, width: int = 0) -> str:
    """Return str(value) or '-' if missing, optionally truncated."""
    if value is None:
        return "-"
    s = str(value)
    if width and len(s) > width:
        s = s[:width]
    return s


def _sort_key(row: dict[str, object], sort_by: str) -> tuple[int, float | str]:
    """Return a sort key that puts missing values last."""
    if sort_by == "primary_mae":
        val = row.get("primary_mae")
        if val is None:
            val = row.get("overall_mae")
    else:
        val = row.get(sort_by)
    if val is None:
        return (1, 0.0)
    if sort_by == "timestamp":
        # Sort descending (newest first) — negate not possible for str, use inverse flag below
        return (0, str(val))
    try:
        return (0, float(str(val)))
    except (TypeError, ValueError):
        return (1, 0.0)


def _cells_for_raw_row(row: dict[str, object]) -> list[str]:
    """Return table cells for a single un-aggregated row."""
    commit = _fmt_str(row.get("commit"), width=8)
    ts = _fmt_str(row.get("timestamp"), width=19)
    fold = _fmt_str(row.get("fold_idx"))
    return [
        _fmt_str(row.get("version")),
        _fmt_mae(row.get("primary_mae")),
        _fmt_mae(row.get("overall_mae")),
        _fmt_mae(row.get("seen_mae")),
        _fmt_mae(row.get("unseen_mae")),
        _fmt_mae(row.get("q4_mae")),
        _fmt_pct(row.get("unseen_event_pct_by_event")),
        _fmt_str(row.get("features_n")),
        commit,
        ts,
        fold,
    ]


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def _fmt_mae_agg(values: list[float | None]) -> str:
    """Format aggregated MAE as '$mean ± std' or '-'."""
    clean = [v for v in values if v is not None]
    if not clean:
        return "-"
    mean, std = _mean_std(clean)
    if std == 0.0:
        return f"${mean:.2f}"
    return f"${mean:.2f}±{std:.2f}"


def _aggregate_by_version(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Group rows by version and return aggregated dicts sorted by insertion order."""
    # Preserve first-seen order of versions.
    order: list[str] = []
    groups: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        ver = str(row.get("version") or "")
        if ver not in groups:
            order.append(ver)
            groups[ver] = []
        groups[ver].append(row)

    aggregated: list[dict[str, object]] = []
    for ver in order:
        grp = groups[ver]
        n = len(grp)

        def _agg_mae(field: str, _grp: list[dict[str, object]] = grp) -> float | None:
            vals = [float(str(r[field])) for r in _grp if r.get(field) is not None]
            if not vals:
                return None
            return sum(vals) / len(vals)

        # For sort key we store the mean as the primary metric value.
        agg: dict[str, object] = {
            "version": ver,
            "primary_mae": _agg_mae("primary_mae"),
            "overall_mae": _agg_mae("overall_mae"),
            "seen_mae": _agg_mae("seen_mae"),
            "unseen_mae": _agg_mae("unseen_mae"),
            "q4_mae": _agg_mae("q4_mae"),
            "unseen_event_pct_by_event": _agg_mae("unseen_event_pct_by_event"),
            "features_n": _agg_mae("features_n"),
            # Use latest timestamp for sorting.
            "timestamp": max(
                (str(r.get("timestamp") or "") for r in grp), default=None
            ),
            # Use most common commit (last seen).
            "commit": next(
                (str(r.get("commit")) for r in reversed(grp) if r.get("commit")),
                None,
            ),
            # Store raw groups for display formatting.
            "_groups": grp,
            "_n_folds": n,
        }
        aggregated.append(agg)
    return aggregated


def _cells_for_agg_row(agg: dict[str, object]) -> list[str]:
    """Return table cells for a per-version aggregated row."""
    grp: list[dict[str, object]] = agg["_groups"]  # type: ignore[assignment]
    n: int = agg["_n_folds"]  # type: ignore[assignment]

    def _agg_fmt(field: str) -> str:
        vals: list[float | None] = [
            float(str(r[field])) if r.get(field) is not None else None for r in grp
        ]
        return _fmt_mae_agg(vals)

    commit = _fmt_str(agg.get("commit"), width=8)
    ts = _fmt_str(agg.get("timestamp"), width=19)
    return [
        _fmt_str(agg.get("version")),
        _agg_fmt("primary_mae"),
        _agg_fmt("overall_mae"),
        _agg_fmt("seen_mae"),
        _agg_fmt("unseen_mae"),
        _agg_fmt("q4_mae"),
        _fmt_pct(agg.get("unseen_event_pct_by_event")),
        _fmt_str(agg.get("features_n")),
        commit,
        ts,
        f"mean({n})",
    ]


def _print_table(
    rows: list[dict[str, object]],
    sort_by: str,
    limit: int,
    per_fold: bool,
) -> None:
    """Print a plain-ASCII fixed-width table to stdout."""
    display_rows = _aggregate_by_version(rows) if not per_fold else list(rows)

    # Sort: ascending for MAE fields (lower is better), descending for timestamp.
    reverse = sort_by == "timestamp"
    display_rows.sort(key=lambda r: _sort_key(r, sort_by), reverse=reverse)

    display_rows = display_rows[:limit]

    # Build cell matrix: list of (cells) per row.
    matrix: list[list[str]] = []
    for rank, row in enumerate(display_rows, start=1):
        cells = _cells_for_raw_row(row) if per_fold else _cells_for_agg_row(row)
        matrix.append([str(rank)] + cells)

    # Compute column widths from headers + data.
    col_widths = [len(h) for h in _HEADERS]
    for cells in matrix:
        for i, cell in enumerate(cells):
            col_widths[i] = max(col_widths[i], len(cell))

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(c.ljust(col_widths[i]) for i, c in enumerate(cells))

    separator = "  ".join("-" * w for w in col_widths)

    print(_fmt_row(_HEADERS))
    print(separator)
    for cells in matrix:
        print(_fmt_row(cells))

    print()
    suffix = "(per-fold)" if per_fold else "(per-version, aggregated)"
    print(
        f"Showing {len(display_rows)} row(s) {suffix}, "
        f"sorted by {sort_by} {'desc' if reverse else 'asc'}."
    )


def _print_json(
    rows: list[dict[str, object]],
    sort_by: str,
    limit: int,
    per_fold: bool,
) -> None:
    """Print JSON output."""
    if not per_fold:
        display_rows = _aggregate_by_version(rows)
        # Strip internal keys before serialising.
        for r in display_rows:
            r.pop("_groups", None)
            r.pop("_n_folds", None)
    else:
        display_rows = list(rows)

    reverse = sort_by == "timestamp"
    display_rows.sort(key=lambda r: _sort_key(r, sort_by), reverse=reverse)
    display_rows = display_rows[:limit]

    print(json.dumps(display_rows, indent=2, default=str))


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Show experiment leaderboard from experiments.jsonl"
    )
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(".claude/coral/experiments/experiments.jsonl"),
        help="Path to experiments.jsonl (default: .claude/coral/experiments/experiments.jsonl)",
    )
    parser.add_argument(
        "--sort-by",
        default="primary_mae",
        choices=list(_SORT_CHOICES),
        help=(
            "Column to sort by (default: primary_mae, with fallback to "
            "overall_mae when primary_mae is missing)"
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of rows to show (default: 20)",
    )
    parser.add_argument(
        "--format",
        default="table",
        choices=["table", "json"],
        dest="fmt",
        help="Output format (default: table)",
    )
    parser.add_argument(
        "--per-fold",
        action="store_true",
        help="Disable per-version aggregation and show every fold row individually",
    )
    args = parser.parse_args()

    if not args.path.exists():
        print(
            "No experiments logged yet. "
            "Run train_model.py --cv --cv-with-breakdown.",
            file=sys.stderr,
        )
        sys.exit(0)

    rows = _load_rows(args.path)

    if not rows:
        print(
            "No experiments logged yet. "
            "Run train_model.py --cv --cv-with-breakdown.",
            file=sys.stderr,
        )
        sys.exit(0)

    if args.fmt == "json":
        _print_json(rows, args.sort_by, args.limit, args.per_fold)
    else:
        _print_table(rows, args.sort_by, args.limit, args.per_fold)


if __name__ == "__main__":
    main()
