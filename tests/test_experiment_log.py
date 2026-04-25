"""Tests for experiment_log module."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pytest

from ticket_price_predictor.ml.training.experiment_log import _get_commit_sha, log_experiment


def test_log_experiment_round_trip(tmp_path: Path) -> None:
    """log_experiment writes a valid JSON line that can be read back."""
    jsonl_path = tmp_path / "subdir" / "experiments.jsonl"
    metrics = {
        "primary_mae": 52.0,
        "overall_mae": 85.5,
        "seen_mae": 52.0,
        "unseen_mae": 128.0,
        "q4_mae": 210.0,
        "unseen_event_pct_by_event": 0.43,
        "features_n": 81,
    }
    config = {"model": "lightgbm", "version": "v37", "n_folds": 5, "loss": None}

    log_experiment(
        jsonl_path, metrics=metrics, config=config, commit_sha="abc1234", seed=42, fold_idx=0
    )

    assert jsonl_path.exists()
    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    record = json.loads(lines[0])
    assert record["primary_mae"] == 52.0
    assert record["overall_mae"] == 85.5
    assert record["seen_mae"] == 52.0
    assert record["unseen_mae"] == 128.0
    assert record["q4_mae"] == 210.0
    assert record["unseen_event_pct_by_event"] == pytest.approx(0.43)
    assert record["features_n"] == 81
    assert record["version"] == "v37"
    assert record["commit"] == "abc1234"
    assert record["seed"] == 42
    assert record["fold_idx"] == 0
    assert record["config_summary"] == config
    assert "timestamp" in record


def test_log_experiment_appends(tmp_path: Path) -> None:
    """Multiple calls append multiple lines."""
    jsonl_path = tmp_path / "experiments.jsonl"
    metrics = {"overall_mae": 90.0}
    config = {"version": "v1"}

    log_experiment(jsonl_path, metrics=metrics, config=config, commit_sha="aaa")
    log_experiment(jsonl_path, metrics={"overall_mae": 85.0}, config=config, commit_sha="bbb")

    lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["overall_mae"] == 90.0
    assert json.loads(lines[1])["overall_mae"] == 85.0


def test_log_experiment_missing_metrics_keys(tmp_path: Path) -> None:
    """Missing metric keys produce None values, not KeyError."""
    jsonl_path = tmp_path / "experiments.jsonl"
    log_experiment(jsonl_path, metrics={}, config={}, commit_sha="x")
    record = json.loads(jsonl_path.read_text(encoding="utf-8").strip())
    assert record["overall_mae"] is None
    assert record["seen_mae"] is None


def test_get_commit_sha_in_empty_tempdir() -> None:
    """_get_commit_sha returns None when run outside a git repo."""
    original_dir = os.getcwd()
    with tempfile.TemporaryDirectory() as tmpdir:
        try:
            os.chdir(tmpdir)
            result = _get_commit_sha()
            assert result is None
        finally:
            os.chdir(original_dir)
