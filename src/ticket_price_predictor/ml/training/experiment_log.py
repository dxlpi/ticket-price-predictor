"""Experiment logging — appends per-fold metrics to a JSONL file."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _get_commit_sha() -> str | None:
    """Return short git commit SHA, or None if not in a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def log_experiment(
    jsonl_path: Path,
    metrics: dict[str, Any],
    config: dict[str, Any],
    commit_sha: str | None = None,
    seed: int | None = None,
    fold_idx: int | None = None,
) -> None:
    """Append one JSON line to jsonl_path.

    Creates parent directories if needed. Fields written:
    - timestamp: ISO-8601 UTC
    - version: from config["version"] if present
    - primary_mae, overall_mae, seen_mae, unseen_mae, q4_mae, unseen_event_pct_by_event:
      from metrics (None if absent)
    - features_n: from metrics["features_n"] if present
    - commit: commit_sha (auto-resolved via _get_commit_sha() if None)
    - seed: seed
    - config_summary: the full config dict
    - fold_idx: fold index (optional)

    Args:
        jsonl_path: Path to the JSONL file (created if absent).
        metrics: Dict containing breakdown metrics from evaluate_with_breakdown.
        config: Dict of training configuration (model, version, n_folds, …).
        commit_sha: Git short SHA. Auto-resolved when None.
        seed: Random seed used for this run.
        fold_idx: CV fold index, or None for a single-split run.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    resolved_sha = commit_sha if commit_sha is not None else _get_commit_sha()

    record: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "version": config.get("version"),
        "primary_mae": metrics.get("primary_mae"),
        "overall_mae": metrics.get("overall_mae"),
        "seen_mae": metrics.get("seen_mae"),
        "unseen_mae": metrics.get("unseen_mae"),
        "q4_mae": metrics.get("q4_mae"),
        "unseen_event_pct_by_event": metrics.get("unseen_event_pct_by_event"),
        "features_n": metrics.get("features_n"),
        "commit": resolved_sha,
        "seed": seed,
        "config_summary": config,
        "fold_idx": fold_idx,
    }

    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
