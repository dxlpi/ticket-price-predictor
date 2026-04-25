"""Tests for StackingEnsembleV2 OOF dollar-space metrics (AC8 wiring)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2
from ticket_price_predictor.ml.training.target_transforms import LogTransform


def _make_tiny_df(n: int = 120) -> tuple[pd.DataFrame, pd.Series]:
    """Return a minimal feature DataFrame and log-space target Series."""
    rng = np.random.default_rng(42)
    prices = rng.uniform(20.0, 300.0, size=n)
    X = pd.DataFrame(
        {
            "event_section_median_price": np.log1p(prices),
            "event_zone_median_price": np.log1p(prices * 0.9),
            "artist_regional_median_price": np.log1p(prices * 1.1),
        }
    )
    y = pd.Series(np.log1p(prices), name="listing_price")
    return X, y


def _tiny_base_configs() -> list[dict]:  # type: ignore[type-arg]
    """Minimal base configs: two Ridge-like LightGBM stubs with low n_estimators."""
    from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel

    return [
        {
            "name": "lgb_a",
            "cls": LightGBMModel,
            "params": {**LightGBMModel.DEFAULT_PARAMS, "n_estimators": 5, "verbose": -1},
        },
        {
            "name": "lgb_b",
            "cls": LightGBMModel,
            "params": {**LightGBMModel.DEFAULT_PARAMS, "n_estimators": 5, "verbose": -1},
        },
    ]


class TestOofMetricsNoTransform:
    """With target_transform=None, _oof_metrics must be empty dict and no error."""

    def test_fit_does_not_error(self) -> None:
        X, y = _make_tiny_df()
        model = StackingEnsembleV2(
            base_configs=_tiny_base_configs(),
            n_folds=2,
            target_transform=None,
        )
        model.fit(X, y)
        assert model._fitted

    def test_oof_metrics_empty_when_no_transform(self) -> None:
        X, y = _make_tiny_df()
        model = StackingEnsembleV2(
            base_configs=_tiny_base_configs(),
            n_folds=2,
            target_transform=None,
        )
        model.fit(X, y)
        # _oof_metrics exists but is empty (no transform supplied)
        assert isinstance(model._oof_metrics, dict)
        assert len(model._oof_metrics) == 0


class TestOofMetricsWithLogTransform:
    """With target_transform=LogTransform(), _oof_metrics must be populated."""

    @pytest.fixture(scope="class")
    def fitted_model(self) -> StackingEnsembleV2:
        X, y = _make_tiny_df()
        model = StackingEnsembleV2(
            base_configs=_tiny_base_configs(),
            n_folds=2,
            target_transform=LogTransform(),
        )
        model.fit(X, y)
        return model

    def test_oof_metrics_exists(self, fitted_model: StackingEnsembleV2) -> None:
        assert hasattr(fitted_model, "_oof_metrics")
        assert isinstance(fitted_model._oof_metrics, dict)

    def test_oof_metrics_has_base_learner_keys(self, fitted_model: StackingEnsembleV2) -> None:
        assert "lgb_a" in fitted_model._oof_metrics
        assert "lgb_b" in fitted_model._oof_metrics

    def test_oof_metrics_has_ridge_key(self, fitted_model: StackingEnsembleV2) -> None:
        assert "ridge" in fitted_model._oof_metrics

    def test_oof_metrics_values_are_positive_floats(self, fitted_model: StackingEnsembleV2) -> None:
        for key, val in fitted_model._oof_metrics.items():
            assert isinstance(val, float), f"{key} value is not float: {val!r}"
            assert val > 0.0, f"{key} MAE must be positive, got {val}"
