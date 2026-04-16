"""Tests for StackingEnsembleV2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def make_test_data(
    n: int = 100,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Create minimal test data with coarse features required by ResidualModel."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "event_section_median_price": rng.normal(5.0, 0.3, n),
            "event_zone_median_price": rng.normal(4.8, 0.3, n),
            "event_median_price": rng.normal(4.9, 0.3, n),
            "event_zone_price_ratio": rng.normal(1.0, 0.1, n),
            "event_section_price_ratio": rng.normal(1.0, 0.1, n),
            "feature_a": rng.normal(0, 1, n),
            "feature_b": rng.normal(0, 1, n),
        }
    )
    y = pd.Series(rng.normal(5.0, 0.3, n))
    split = n // 5
    return X.iloc[:-split], y.iloc[:-split], X.iloc[-split:], y.iloc[-split:]


def test_stacking_v2_fits_and_predicts() -> None:
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    X_train, y_train, X_val, y_val = make_test_data()
    # Use 2 folds and no neural to keep test fast
    model = StackingEnsembleV2(n_folds=2, include_neural=False)
    model.fit(X_train, y_train, X_val, y_val)
    assert model.is_fitted
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)


def test_stacking_v2_name() -> None:
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    assert StackingEnsembleV2().name == "stacking_v2"


def test_stacking_v2_registered_in_trainer() -> None:
    from ticket_price_predictor.ml.training.trainer import ModelTrainer

    trainer = ModelTrainer(model_type="stacking_v2", model_version="test")
    model = trainer._create_model()
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    assert isinstance(model, StackingEnsembleV2)


def test_stacking_v2_default_base_configs_include_residual() -> None:
    from ticket_price_predictor.ml.models.stacking_v2 import _default_v2_base_configs

    configs = _default_v2_base_configs()
    cls_names = [c["cls"].__name__ for c in configs]
    assert "ResidualModel" in cls_names
    # Also verify LightGBMModel variants
    assert cls_names.count("LightGBMModel") == 2


def test_stacking_v2_predict_before_fit_raises() -> None:
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    model = StackingEnsembleV2()
    with pytest.raises(RuntimeError):
        model.predict(pd.DataFrame({"x": [1.0]}))


def test_stacking_v2_feature_importance() -> None:
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    X_train, y_train, X_val, y_val = make_test_data()
    model = StackingEnsembleV2(n_folds=2, include_neural=False)
    model.fit(X_train, y_train, X_val, y_val)
    importance = model.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    # Sum is 0 when no splits built (tiny test data) or ~1 when normalized
    total = sum(importance.values())
    assert total == 0.0 or abs(total - 1.0) < 0.01


def test_stacking_v2_get_params() -> None:
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    model = StackingEnsembleV2(n_folds=3, meta_alpha=2.0, include_neural=False)
    params = model.get_params()
    assert params["n_folds"] == 3
    assert params["meta_alpha"] == 2.0
    assert params["n_base_models"] == 3
    assert "lgb_huber" in params["base_model_names"]
    assert "residual" in params["base_model_names"]


def test_fit_sorts_by_timestamps_when_provided() -> None:
    """Verify that passing timestamps=scrambled_ts makes the fit equivalent
    to passing rows in already-sorted order without timestamps.

    This proves that the internal sort correctly reorders X_train so positional
    k-fold indices correspond to true temporal order (not artist-grouped order).
    """
    from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    rng = np.random.default_rng(42)
    n = 120
    # Features correlated with time
    ts_sorted = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    X_sorted = pd.DataFrame(
        {
            "f1": np.arange(n, dtype=float) + rng.normal(0, 0.1, n),
            "f2": rng.normal(0, 1, n),
        }
    )
    y_sorted = pd.Series(np.arange(n, dtype=float) * 0.5 + rng.normal(0, 0.2, n))

    # Scramble rows to simulate artist-grouped concat order
    perm = rng.permutation(n)
    X_scrambled = X_sorted.iloc[perm].reset_index(drop=True)
    y_scrambled = y_sorted.iloc[perm].reset_index(drop=True)
    ts_scrambled = pd.Series(ts_sorted).iloc[perm].reset_index(drop=True)

    # Fit baseline on pre-sorted rows without timestamps kwarg
    base_configs = [
        {
            "name": "lgb_fast",
            "cls": LightGBMModel,
            "params": {**LightGBMModel.DEFAULT_PARAMS, "n_estimators": 20, "verbose": -1},
        },
    ]

    model_sorted = StackingEnsembleV2(base_configs=base_configs, n_folds=2)
    model_sorted.fit(X_sorted, y_sorted)

    # Fit with scrambled rows + timestamps — internal sort should recover order
    model_scrambled = StackingEnsembleV2(base_configs=base_configs, n_folds=2)
    model_scrambled.fit(X_scrambled, y_scrambled, timestamps=ts_scrambled)

    # Meta-learner weights should match (up to numeric noise) because after
    # internal sort the two fits see identical training data in identical order.
    np.testing.assert_allclose(
        model_sorted._meta_model.coef_,  # type: ignore[union-attr]
        model_scrambled._meta_model.coef_,  # type: ignore[union-attr]
        rtol=1e-6,
        atol=1e-8,
    )


def test_fit_rejects_mismatched_timestamps_length() -> None:
    """fit() must raise ValueError when timestamps length differs from X_train."""
    from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
    from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2

    X = pd.DataFrame({"f1": np.arange(50, dtype=float)})
    y = pd.Series(np.arange(50, dtype=float))
    wrong_ts = pd.Series(pd.date_range("2026-01-01", periods=10, freq="h", tz="UTC"))

    base_configs = [
        {
            "name": "lgb_fast",
            "cls": LightGBMModel,
            "params": {**LightGBMModel.DEFAULT_PARAMS, "n_estimators": 10, "verbose": -1},
        },
    ]
    model = StackingEnsembleV2(base_configs=base_configs, n_folds=2)
    with pytest.raises(ValueError, match="timestamps length"):
        model.fit(X, y, timestamps=wrong_ts)
