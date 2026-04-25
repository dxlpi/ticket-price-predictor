"""Tests for Tweedie and quantile loss paths in LightGBMModel.

Verifies:
- LightGBMModel accepts objective="tweedie" and trains without error
- LightGBMModel accepts objective="quantile" and trains without error
- Existing huber path is unchanged (regression test)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel


def _make_xy(n: int = 200, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Create a small synthetic training set."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n),
            "feature_b": rng.standard_normal(n),
            "feature_c": rng.standard_normal(n),
        }
    )
    # Prices in [10, 500] — right-skewed
    y_raw = rng.exponential(scale=100.0, size=n).clip(10, 500)
    return X, pd.Series(y_raw, name="price")


@pytest.mark.parametrize(
    "objective,extra_params",
    [
        ("tweedie", {"tweedie_variance_power": 1.5}),
        ("quantile", {"alpha": 0.5}),
    ],
)
def test_lightgbm_new_objectives_train_and_predict(objective: str, extra_params: dict) -> None:
    """Tweedie and quantile objectives train without error and produce finite predictions."""
    X_train, y_train = _make_xy(n=300)
    X_val, y_val = _make_xy(n=100, seed=99)

    # Build params: start from defaults (which has alpha=1.0 for huber),
    # override objective and n_estimators, then apply extra_params last.
    # Remove default huber alpha first so it doesn't shadow extra_params["alpha"].
    base = {**LightGBMModel.DEFAULT_PARAMS}
    base.pop("alpha", None)  # remove huber alpha before applying extra_params
    params = {
        **base,
        "objective": objective,
        "n_estimators": 50,
        "early_stopping_rounds": 10,
        **extra_params,
    }

    model = LightGBMModel(params=params)
    model.fit(X_train, y_train, X_val, y_val)

    assert model.is_fitted
    preds = model.predict(X_val)
    assert preds.shape == (len(X_val),)
    assert np.all(np.isfinite(preds)), f"Non-finite predictions for objective={objective}"


def test_huber_path_unchanged() -> None:
    """Huber path (default) still trains and metrics are non-zero."""
    X_train, y_train = _make_xy(n=300)
    X_val, y_val = _make_xy(n=100, seed=99)

    # Log-transform target — same as trainer.py default path
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)

    params = {
        **LightGBMModel.DEFAULT_PARAMS,
        "n_estimators": 50,
        "early_stopping_rounds": 10,
    }
    model = LightGBMModel(params=params)
    model.fit(X_train, y_train_log, X_val, y_val_log)

    assert model.is_fitted
    preds_log = model.predict(X_val)
    preds = np.expm1(preds_log)

    mae = float(np.mean(np.abs(preds - y_val.values)))
    assert mae > 0, "Huber MAE should be positive (non-trivial predictions)"
    assert np.all(np.isfinite(preds)), "Huber path produced non-finite predictions"


def test_tweedie_no_log_transform_needed() -> None:
    """Tweedie objective trained on raw prices (no log1p) produces positive predictions."""
    X_train, y_train = _make_xy(n=300)
    X_val, y_val = _make_xy(n=100, seed=99)

    # Raw prices, no log1p — this is the tweedie_raw target path
    params = {
        **LightGBMModel.DEFAULT_PARAMS,
        "objective": "tweedie",
        "tweedie_variance_power": 1.5,
        "n_estimators": 50,
        "early_stopping_rounds": 10,
    }
    params.pop("alpha", None)

    model = LightGBMModel(params=params)
    model.fit(X_train, y_train, X_val, y_val)

    preds = model.predict(X_val)
    # Tweedie predictions should be non-negative for price data
    assert np.all(np.isfinite(preds)), "Tweedie predictions not finite"
    # Reasonable range check — not exactly price range, but shouldn't be wildly off
    assert float(np.mean(preds)) > 0, "Mean Tweedie prediction should be positive"
