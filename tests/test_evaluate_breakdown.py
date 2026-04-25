"""Tests for evaluate_with_breakdown in evaluator.py."""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from ticket_price_predictor.ml.training.evaluator import evaluate_with_breakdown


class _IdentityModel:
    """Stub model that returns predictions passed at construction time."""

    def __init__(self, predictions: npt.NDArray[Any]) -> None:
        self._predictions = predictions

    def predict(self, X: npt.NDArray[Any]) -> npt.NDArray[Any]:  # noqa: ARG002
        return self._predictions

    def get_feature_importance(self) -> dict[str, float]:
        return {}

    @property
    def name(self) -> str:
        return "identity"


def _make_breakdown(
    y_true: list[float],
    y_pred: list[float],
    event_ids: list[str],
    train_events: set[str],
) -> dict[str, Any]:
    arr_true = np.array(y_true, dtype=float)
    arr_pred = np.array(y_pred, dtype=float)
    X_dummy = np.zeros((len(y_true), 1))
    model = _IdentityModel(arr_pred)
    return evaluate_with_breakdown(
        X_test=X_dummy,
        y_test=arr_true,
        test_events=np.array(event_ids),
        train_events=train_events,
        model=model,
    )


def test_seen_unseen_mae_hand_computed() -> None:
    """3 seen events, 2 unseen; verify seen_mae and unseen_mae match hand calc."""
    # Seen events: e1, e2, e3 — each contributes 2 rows
    # Unseen events: e4, e5 — each contributes 2 rows
    #
    # y_true for seen:    [100, 200, 150, 250, 120, 180]  mean abs error vs pred
    # y_pred for seen:    [90,  220, 140, 260, 130, 170]
    # abs errors seen:    [10,  20,  10,  10,  10,  10]  → mean = 70/6 ≈ 11.667
    #
    # y_true for unseen:  [300, 400, 500, 600]
    # y_pred for unseen:  [250, 430, 450, 650]
    # abs errors unseen:  [50,  30,  50,  50]  → mean = 180/4 = 45.0

    y_true = [100.0, 200.0, 150.0, 250.0, 120.0, 180.0, 300.0, 400.0, 500.0, 600.0]
    y_pred = [90.0, 220.0, 140.0, 260.0, 130.0, 170.0, 250.0, 430.0, 450.0, 650.0]
    event_ids = ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4", "e5", "e5"]
    train_events = {"e1", "e2", "e3"}

    result = _make_breakdown(y_true, y_pred, event_ids, train_events)

    assert result["seen_mae"] == pytest.approx(70.0 / 6, rel=1e-6)
    assert result["unseen_mae"] == pytest.approx(45.0, rel=1e-6)
    assert result["n_seen"] == 6
    assert result["n_unseen"] == 4
    assert result["primary_mae"] == result["seen_mae"]


def test_unseen_event_pct_by_event() -> None:
    """unseen_event_pct_by_event is computed by distinct event count, not rows."""
    # 3 seen events, 2 unseen → 2/5 = 0.4 by event
    event_ids = ["e1", "e1", "e2", "e2", "e3", "e3", "e4", "e4", "e5", "e5"]
    train_events = {"e1", "e2", "e3"}
    y_vals = [100.0] * 10

    result = _make_breakdown(y_vals, y_vals, event_ids, train_events)

    assert result["unseen_event_pct_by_event"] == pytest.approx(0.4, rel=1e-6)


def test_all_seen() -> None:
    """When all test events are seen, unseen_mae is nan and pct is 0."""
    event_ids = ["e1", "e2", "e3"]
    train_events = {"e1", "e2", "e3"}
    y_vals = [100.0, 200.0, 300.0]

    result = _make_breakdown(y_vals, y_vals, event_ids, train_events)

    assert result["n_unseen"] == 0
    assert np.isnan(result["unseen_mae"])
    assert result["unseen_event_pct_by_event"] == pytest.approx(0.0)


def test_all_unseen() -> None:
    """When no test events are seen, seen_mae is nan and pct is 1."""
    event_ids = ["e4", "e5"]
    train_events = {"e1", "e2", "e3"}
    y_vals = [100.0, 200.0]

    result = _make_breakdown(y_vals, y_vals, event_ids, train_events)

    assert result["n_seen"] == 0
    assert np.isnan(result["seen_mae"])
    assert result["unseen_event_pct_by_event"] == pytest.approx(1.0)


def test_primary_mae_when_all_unseen() -> None:
    """When no test events are seen, primary_mae is nan (mirrors seen_mae)."""
    event_ids = ["e4", "e5"]
    train_events = {"e1", "e2", "e3"}
    y_vals = [100.0, 200.0]

    result = _make_breakdown(y_vals, y_vals, event_ids, train_events)

    assert result["n_seen"] == 0
    assert np.isnan(result["primary_mae"])


def test_overall_mae_consistent() -> None:
    """overall_mae matches sklearn mean_absolute_error over all rows."""
    from sklearn.metrics import mean_absolute_error

    y_true = [100.0, 200.0, 300.0, 400.0]
    y_pred = [110.0, 180.0, 320.0, 380.0]
    event_ids = ["e1", "e2", "e3", "e4"]
    train_events = {"e1", "e2"}

    result = _make_breakdown(y_true, y_pred, event_ids, train_events)

    expected = mean_absolute_error(y_true, y_pred)
    assert result["overall_mae"] == pytest.approx(expected, rel=1e-6)
