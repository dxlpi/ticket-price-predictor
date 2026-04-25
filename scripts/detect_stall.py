#!/usr/bin/env python3
"""Detect plateau in experiment results.

Reads .claude/coral/experiments/experiments.jsonl and prints a plain-text stall
memo when the last 3 rows each show < $1 improvement over the previous row's
overall_mae. Not a kill switch — observability only.

Usage:
    python scripts/detect_stall.py
    python scripts/detect_stall.py --path path/to/experiments.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import json
from pathlib import Path


def _load_rows(jsonl_path: Path) -> list[dict]:  # type: ignore[type-arg]
    """Load all JSON lines from jsonl_path. Returns [] if file does not exist."""
    if not jsonl_path.exists():
        return []
    rows = []
    with jsonl_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                with contextlib.suppress(json.JSONDecodeError):
                    rows.append(json.loads(line))
    return rows


def _get_mae(row: dict) -> float | None:  # type: ignore[type-arg]
    val = row.get("primary_mae")
    if val is None:
        val = row.get("overall_mae")
    return val


def _detect_stall(rows: list[dict]) -> bool:  # type: ignore[type-arg]
    """Return True if the last 3 rows each improve < $1 over their predecessor."""
    if len(rows) < 4:
        return False
    # Need 4 rows to check 3 consecutive improvements (row[i] vs row[i-1])
    tail = rows[-4:]
    maes = [_get_mae(r) for r in tail]
    if any(m is None for m in maes):
        return False
    floats = [float(m) for m in maes if m is not None]
    improvements = [floats[i] - floats[i + 1] for i in range(3)]
    return all(imp < 1.0 for imp in improvements)


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Detect experiment plateau")
    parser.add_argument(
        "--path",
        type=Path,
        default=Path(".claude/coral/experiments/experiments.jsonl"),
        help="Path to experiments.jsonl",
    )
    args = parser.parse_args()

    rows = _load_rows(args.path)

    if not rows:
        print(f"No experiments found at {args.path}")
        return

    print(f"Loaded {len(rows)} experiment row(s) from {args.path}")

    if _detect_stall(rows):
        tail = rows[-4:]
        tried = [
            (
                f"  fold_idx={r.get('fold_idx')} version={r.get('version')} primary_mae=${_get_mae(r):.2f}"
                if r.get("primary_mae") is not None
                else f"  fold_idx={r.get('fold_idx')} version={r.get('version')} overall_mae=${_get_mae(r):.2f} (legacy)"
            )
            if isinstance(_get_mae(r), (int, float))
            else (
                f"  fold_idx={r.get('fold_idx')} version={r.get('version')} primary_mae=n/a"
                if r.get("primary_mae") is not None
                else f"  fold_idx={r.get('fold_idx')} version={r.get('version')} overall_mae=n/a"
            )
            for r in tail[1:]
        ]
        last_row = rows[-1]
        last_mae = _get_mae(last_row)
        last_label = "primary_mae" if last_row.get("primary_mae") is not None else "overall_mae"
        print()
        print("STALL DETECTED: last 3 experiments each improved < $1 primary_mae")
        print(
            f"Current plateau: {last_label}=${last_mae:.2f}"
            if isinstance(last_mae, (int, float))
            else f"Current plateau: {last_label}=n/a"
        )
        print("Recent experiments:")
        for line in tried:
            print(line)
    else:
        last_row = rows[-1]
        last_mae = _get_mae(last_row)
        last_label = "primary_mae" if last_row.get("primary_mae") is not None else "overall_mae"
        print(
            f"No stall detected. Latest {last_label}: ${last_mae:.2f}"
            if isinstance(last_mae, (int, float))
            else f"No stall detected. Latest {last_label}: n/a"
        )


if __name__ == "__main__":
    main()
