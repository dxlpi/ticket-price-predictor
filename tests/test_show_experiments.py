"""Tests for scripts/show_experiments.py."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Add scripts/ to the path so we can import directly.
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import show_experiments as se  # noqa: E402, I001


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:  # type: ignore[type-arg]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


_ROW_V1_F0 = {
    "version": "v37-baseline",
    "fold_idx": 0,
    "primary_mae": 52.0,
    "overall_mae": 85.0,
    "seen_mae": 52.0,
    "unseen_mae": 130.0,
    "q4_mae": 300.0,
    "unseen_event_pct_by_event": 0.43,
    "features_n": 81,
    "commit": "abc1234",
    "timestamp": "2026-04-16T10:00:00",
}

_ROW_V1_F1 = {
    **_ROW_V1_F0,
    "fold_idx": 1,
    "primary_mae": 54.0,
    "overall_mae": 87.0,
    "seen_mae": 54.0,
    "unseen_mae": 132.0,
    "q4_mae": 302.0,
    "timestamp": "2026-04-16T11:00:00",
}

_ROW_V2_F0 = {
    "version": "v38",
    "fold_idx": 0,
    "primary_mae": 50.0,
    "overall_mae": 80.0,
    "seen_mae": 50.0,
    "unseen_mae": 120.0,
    "q4_mae": 290.0,
    "unseen_event_pct_by_event": 0.40,
    "features_n": 85,
    "commit": "def5678",
    "timestamp": "2026-04-16T12:00:00",
}

# Legacy row with no primary_mae — exercises the fallback to overall_mae.
_ROW_V_LEGACY = {
    "version": "v36-legacy",
    "fold_idx": 0,
    "overall_mae": 95.0,
    "seen_mae": 60.0,
    "unseen_mae": 140.0,
    "q4_mae": 310.0,
    "unseen_event_pct_by_event": 0.45,
    "features_n": 79,
    "commit": "fed9876",
    "timestamp": "2026-04-15T10:00:00",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_missing_file_exits_zero(tmp_path: Path) -> None:
    """Missing JSONL file should exit 0, not 1."""
    missing = tmp_path / "nope.jsonl"
    with pytest.raises(SystemExit) as exc_info:
        sys.argv = ["show_experiments.py", "--path", str(missing)]
        se.main()
    assert exc_info.value.code == 0


def test_empty_file_exits_zero(tmp_path: Path) -> None:
    """Empty JSONL file should exit 0."""
    empty = tmp_path / "empty.jsonl"
    empty.write_text("")
    with pytest.raises(SystemExit) as exc_info:
        sys.argv = ["show_experiments.py", "--path", str(empty)]
        se.main()
    assert exc_info.value.code == 0


def test_single_row_prints_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Single row should produce a ranked table with rank=1."""
    jsonl = tmp_path / "exp.jsonl"
    _write_jsonl(jsonl, [_ROW_V1_F0])

    sys.argv = ["show_experiments.py", "--path", str(jsonl)]
    se.main()

    out = capsys.readouterr().out
    assert "rank" in out
    assert "v37-baseline" in out
    assert "$85.00" in out
    # Rank 1 appears
    assert "1" in out


def test_multi_fold_aggregation(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Two folds of the same version should be aggregated into a single row."""
    jsonl = tmp_path / "exp.jsonl"
    _write_jsonl(jsonl, [_ROW_V1_F0, _ROW_V1_F1])

    sys.argv = ["show_experiments.py", "--path", str(jsonl)]
    se.main()

    out = capsys.readouterr().out
    # Mean of 85 and 87 is 86
    assert "$86.00" in out
    # Fold summary column
    assert "mean(2)" in out
    # Only one data row (aggregated)
    lines = [ln for ln in out.splitlines() if "v37-baseline" in ln]
    assert len(lines) == 1


def test_per_fold_shows_all_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--per-fold should show each fold row individually."""
    jsonl = tmp_path / "exp.jsonl"
    _write_jsonl(jsonl, [_ROW_V1_F0, _ROW_V1_F1])

    sys.argv = ["show_experiments.py", "--path", str(jsonl), "--per-fold"]
    se.main()

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "v37-baseline" in ln]
    assert len(lines) == 2


def test_limit(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--limit 1 should show only 1 row even when 2 versions are present."""
    jsonl = tmp_path / "exp.jsonl"
    _write_jsonl(jsonl, [_ROW_V1_F0, _ROW_V2_F0])

    sys.argv = ["show_experiments.py", "--path", str(jsonl), "--limit", "1"]
    se.main()

    out = capsys.readouterr().out
    # Only 1 data row after header + separator
    # v38 has lower primary_mae (50.0 vs 52.0) so it should appear first
    # (default sort is asc by primary_mae)
    assert "v38" in out
    assert "v37-baseline" not in out
    assert "Showing 1 row" in out
    assert "sorted by primary_mae" in out


def test_primary_mae_fallback_to_overall_mae_when_missing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A row with only overall_mae should sort/render via the legacy fallback."""
    jsonl = tmp_path / "exp.jsonl"
    # _ROW_V_LEGACY has overall_mae=95 (no primary_mae) → fallback value 95.
    # _ROW_V1_F0 has primary_mae=52.
    # Default sort is asc by primary_mae; v37-baseline (52) ranks ahead of v36-legacy (95).
    _write_jsonl(jsonl, [_ROW_V_LEGACY, _ROW_V1_F0])

    sys.argv = ["show_experiments.py", "--path", str(jsonl)]
    se.main()

    out = capsys.readouterr().out
    assert "v36-legacy" in out
    assert "v37-baseline" in out
    # Legacy row's primary_mae cell renders as '-' (missing), overall_mae is $95.00.
    assert "$95.00" in out

    data_lines = [ln for ln in out.splitlines() if "v37-baseline" in ln or "v36-legacy" in ln]
    assert len(data_lines) == 2
    # v37-baseline (primary_mae=52) sorts ahead of v36-legacy (fallback=95).
    assert "v37-baseline" in data_lines[0]
    assert "v36-legacy" in data_lines[1]


def test_sort_by_timestamp_desc(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--sort-by timestamp should order newest first."""
    jsonl = tmp_path / "exp.jsonl"
    # v37-baseline has earlier timestamp; v38 has later.
    _write_jsonl(jsonl, [_ROW_V1_F0, _ROW_V2_F0])

    sys.argv = ["show_experiments.py", "--path", str(jsonl), "--sort-by", "timestamp"]
    se.main()

    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if "v3" in ln and "rank" not in ln and "---" not in ln]
    # First data line should be v38 (newer timestamp)
    assert "v38" in lines[0]
    assert "v37-baseline" in lines[1]


def test_missing_keys_show_dash(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Rows with missing optional keys should print '-' without raising."""
    sparse = {"version": "v-sparse", "overall_mae": 90.0, "timestamp": "2026-04-16T09:00:00"}
    jsonl = tmp_path / "exp.jsonl"
    _write_jsonl(jsonl, [sparse])

    sys.argv = ["show_experiments.py", "--path", str(jsonl)]
    se.main()

    out = capsys.readouterr().out
    assert "v-sparse" in out
    # Missing fields rendered as '-'
    assert "-" in out


def test_json_format(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """--format json should emit valid JSON."""
    jsonl = tmp_path / "exp.jsonl"
    _write_jsonl(jsonl, [_ROW_V1_F0, _ROW_V2_F0])

    sys.argv = ["show_experiments.py", "--path", str(jsonl), "--format", "json"]
    se.main()

    out = capsys.readouterr().out
    parsed = json.loads(out)
    assert isinstance(parsed, list)
    assert len(parsed) == 2
