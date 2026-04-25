"""Tests for asymmetric Huber loss objective."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import numpy.typing as npt
import pandas as pd

from ticket_price_predictor.ml.models.lightgbm_model import (
    _ASYM_HUBER_DELTA,
    _ASYM_HUBER_FACTOR,
    _ASYM_HUBER_THRESHOLD,
    LightGBMModel,
    _asymmetric_huber_fobj,
)


def _make_dataset(_y_pred: npt.NDArray[Any], y_true: npt.NDArray[Any]) -> MagicMock:
    ds = MagicMock()
    ds.get_label.return_value = y_true
    return ds


def test_over_predict_positive_grad() -> None:
    # Over-predicting (y_pred > y_true) → positive gradient
    y_true = np.array([3.0, 3.0, 3.0])
    y_pred = np.array([4.0, 4.0, 4.0])
    grad, _ = _asymmetric_huber_fobj(y_pred, _make_dataset(y_pred, y_true))
    assert np.all(grad > 0)


def test_under_predict_negative_grad() -> None:
    # Under-predicting (y_pred < y_true, cheap ticket) → negative gradient
    y_true = np.array([3.0, 3.0, 3.0])
    y_pred = np.array([2.0, 2.0, 2.0])
    grad, _ = _asymmetric_huber_fobj(y_pred, _make_dataset(y_pred, y_true))
    assert np.all(grad < 0)


def test_asymmetric_factor_applied_on_expensive_under_prediction() -> None:
    # Expensive ticket (y_true > threshold), under-predicting by exactly delta
    # so we're in the L1 region: grad_symmetric = -delta, grad_asymmetric = -delta * factor
    y_true = np.array([_ASYM_HUBER_THRESHOLD + 0.5])
    y_pred = y_true - (_ASYM_HUBER_DELTA + 0.5)  # residual = -(delta+0.5), L1 region
    grad, hess = _asymmetric_huber_fobj(y_pred, _make_dataset(y_pred, y_true))
    expected_grad = -_ASYM_HUBER_DELTA * _ASYM_HUBER_FACTOR
    assert np.isclose(grad[0], expected_grad), f"expected {expected_grad}, got {grad[0]}"
    expected_hess = _ASYM_HUBER_DELTA * _ASYM_HUBER_FACTOR
    assert np.isclose(hess[0], expected_hess), f"expected {expected_hess}, got {hess[0]}"


def test_asymmetric_factor_not_applied_below_threshold() -> None:
    # Cheap ticket (y_true < threshold), under-predicting → symmetric Huber, no factor
    y_true = np.array([_ASYM_HUBER_THRESHOLD - 0.5])
    y_pred = y_true - (_ASYM_HUBER_DELTA + 0.5)  # L1 region
    grad, hess = _asymmetric_huber_fobj(y_pred, _make_dataset(y_pred, y_true))
    assert np.isclose(grad[0], -_ASYM_HUBER_DELTA)
    assert np.isclose(hess[0], _ASYM_HUBER_DELTA)


def test_asymmetric_factor_not_applied_when_over_predicting_expensive() -> None:
    # Expensive ticket, but over-predicting → factor NOT applied
    y_true = np.array([_ASYM_HUBER_THRESHOLD + 0.5])
    y_pred = y_true + (_ASYM_HUBER_DELTA + 0.5)  # L1 region, residual positive
    grad, hess = _asymmetric_huber_fobj(y_pred, _make_dataset(y_pred, y_true))
    assert np.isclose(grad[0], _ASYM_HUBER_DELTA)
    assert np.isclose(hess[0], _ASYM_HUBER_DELTA)


def test_hessian_always_positive() -> None:
    rng = np.random.default_rng(42)
    y_true = rng.uniform(0.0, 10.0, size=100)
    y_pred = rng.uniform(0.0, 10.0, size=100)
    _, hess = _asymmetric_huber_fobj(y_pred, _make_dataset(y_pred, y_true))
    assert np.all(hess > 0), "Hessian must be strictly positive"


def test_lightgbm_model_fits_with_asymmetric_huber() -> None:
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.random(5), "b": rng.random(5)})
    y = pd.Series(rng.random(5) * 3.0 + 4.0)  # log-space values around 4-7

    model = LightGBMModel(
        params={
            "objective": "asymmetric_huber",
            "n_estimators": 10,
            "early_stopping_rounds": 5,
            "verbose": -1,
            "min_child_samples": 1,
            "num_leaves": 2,
        }
    )
    model.fit(X, y)
    assert model.is_fitted
    preds = model.predict(X)
    assert preds.shape == (5,)
