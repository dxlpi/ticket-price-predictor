"""Tests for SaleProbabilityModel."""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.models.sale_probability import SaleProbabilityModel


def _make_binary_data(n_samples: int = 200):
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "feature_a": rng.standard_normal(n_samples),
            "feature_b": rng.standard_normal(n_samples),
            "feature_c": rng.uniform(0, 1, n_samples),
        }
    )
    # Simple separable signal for fast training
    y = pd.Series((X["feature_a"] + X["feature_b"] > 0).astype(int))
    return X, y


# Use fast params to keep tests quick — override the 8000-tree default
_FAST_PARAMS = {
    "n_estimators": 50,
    "early_stopping_rounds": 10,
    "num_leaves": 16,
    "learning_rate": 0.1,
    "verbose": -1,
}


class TestSaleProbabilityModel:
    def test_predict_returns_probabilities(self):
        """predict() returns values in [0, 1]."""
        X, y = _make_binary_data()
        split = len(X) // 2
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X[:split], y[:split])

        probs = model.predict(X[split:])
        assert probs.min() >= 0.0
        assert probs.max() <= 1.0

    def test_predict_class_returns_binary(self):
        """predict_class() returns only 0s and 1s."""
        X, y = _make_binary_data()
        split = len(X) // 2
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X[:split], y[:split])

        classes = model.predict_class(X[split:])
        assert set(classes).issubset({0, 1})

    def test_predict_class_custom_threshold(self):
        """predict_class() respects custom threshold."""
        X, y = _make_binary_data()
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X, y)
        probs = model.predict(X)

        # At threshold=0.0 everything should be class 1
        all_ones = model.predict_class(X, threshold=0.0)
        assert all(all_ones == 1)

        # At threshold=1.0 everything should be class 0
        all_zeros = model.predict_class(X, threshold=1.0)
        assert all(all_zeros == 0)

        # Threshold=0.5 should match manual comparison
        manual = (probs >= 0.5).astype(int)
        np.testing.assert_array_equal(model.predict_class(X, threshold=0.5), manual)

    def test_predict_with_uncertainty_raises(self):
        """predict_with_uncertainty() raises NotImplementedError."""
        X, y = _make_binary_data()
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X, y)

        with pytest.raises(NotImplementedError):
            model.predict_with_uncertainty(X)

    def test_name(self):
        """Model name is 'sale_probability'."""
        assert SaleProbabilityModel().name == "sale_probability"

    def test_is_fitted_before_and_after(self):
        """is_fitted is False before fit, True after."""
        X, y = _make_binary_data()
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        assert not model.is_fitted
        model.fit(X, y)
        assert model.is_fitted

    def test_predict_before_fit_raises(self):
        """predict() before fit raises RuntimeError."""
        model = SaleProbabilityModel()
        X, _ = _make_binary_data(10)
        with pytest.raises(RuntimeError):
            model.predict(X)

    def test_predict_class_before_fit_raises(self):
        """predict_class() before fit raises RuntimeError (delegates to predict)."""
        model = SaleProbabilityModel()
        X, _ = _make_binary_data(10)
        with pytest.raises(RuntimeError):
            model.predict_class(X)

    def test_get_feature_importance_after_fit(self):
        """get_feature_importance() returns non-empty dict after fit."""
        X, y = _make_binary_data()
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) > 0
        assert all(isinstance(v, float) for v in importance.values())

    def test_get_feature_importance_before_fit(self):
        """get_feature_importance() returns empty dict before fit."""
        model = SaleProbabilityModel()
        importance = model.get_feature_importance()
        assert importance == {}

    def test_get_feature_importance_normalized(self):
        """Feature importances sum to approximately 1.0."""
        X, y = _make_binary_data()
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X, y)

        importance = model.get_feature_importance()
        total = sum(importance.values())
        assert total == pytest.approx(1.0, abs=1e-6)

    def test_get_feature_importance_top_k(self):
        """get_feature_importance(top_k=2) returns at most 2 features."""
        X, y = _make_binary_data()
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X, y)

        importance = model.get_feature_importance(top_k=2)
        assert len(importance) <= 2

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load preserves model predictions."""
        X, y = _make_binary_data()
        split = len(X) // 2
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X[:split], y[:split])
        original_probs = model.predict(X[split:])

        path = tmp_path / "sale_model.joblib"
        model.save(path)

        loaded = SaleProbabilityModel.load(path)
        assert loaded.is_fitted
        loaded_probs = loaded.predict(X[split:])

        np.testing.assert_array_almost_equal(original_probs, loaded_probs)

    def test_save_load_preserves_params(self, tmp_path):
        """Loaded model has the same params as the original."""
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        X, y = _make_binary_data()
        model.fit(X, y)

        path = tmp_path / "sale_model_params.joblib"
        model.save(path)

        loaded = SaleProbabilityModel.load(path)
        # Fast params should be merged in; check a key we set
        assert loaded._params["num_leaves"] == 16

    def test_fit_with_validation(self):
        """fit() accepts X_val/y_val for early stopping."""
        X, y = _make_binary_data(300)
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        # Should not raise
        model.fit(X[:200], y[:200], X_val=X[200:], y_val=y[200:])
        assert model.is_fitted

    def test_fit_with_sample_weight(self):
        """fit() accepts optional sample_weight without error."""
        X, y = _make_binary_data()
        weights = np.ones(len(X))
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        model.fit(X, y, sample_weight=weights)
        assert model.is_fitted

    def test_default_params_are_binary_objective(self):
        """DEFAULT_PARAMS uses binary objective with AUC metric."""
        assert SaleProbabilityModel.DEFAULT_PARAMS["objective"] == "binary"
        assert "auc" in SaleProbabilityModel.DEFAULT_PARAMS["metric"]

    def test_get_params(self):
        """get_params() returns dict with 'params' and 'categorical_features' keys."""
        model = SaleProbabilityModel(params=_FAST_PARAMS)
        p = model.get_params()
        assert "params" in p
        assert "categorical_features" in p
