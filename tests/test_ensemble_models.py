"""Tests for ensemble and new model architectures.

Tests XGBoost, CatBoost, stacking ensemble, residual model,
and custom asymmetric loss functions.
"""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.catboost_model import CatBoostModel
from ticket_price_predictor.ml.models.custom_objectives import (
    asymmetric_huber_metric,
    asymmetric_huber_objective,
    make_asymmetric_huber,
)
from ticket_price_predictor.ml.models.residual import ResidualModel
from ticket_price_predictor.ml.models.stacking import StackingEnsemble
from ticket_price_predictor.ml.models.xgboost_model import XGBoostModel


@pytest.fixture
def sample_data():
    """Create sample training data with enough samples for all models."""
    np.random.seed(42)
    n = 300
    X = pd.DataFrame(
        {
            "feature1": np.random.randn(n),
            "feature2": np.random.randn(n),
            "feature3": np.random.randn(n),
            # Simulate event pricing features for residual model
            "event_section_median_price": np.random.rand(n) * 5 + 3,
            "event_zone_median_price": np.random.rand(n) * 5 + 3,
            "event_median_price": np.random.rand(n) * 5 + 3,
        }
    )
    # Target correlated with features (in log-space range)
    y = pd.Series(
        4.5
        + 0.5 * X["feature1"]
        + 0.3 * X["feature2"]
        + 0.8 * X["event_section_median_price"]
        + np.random.randn(n) * 0.3
    )
    return X, y


# ============================================================================
# XGBoostModel Tests
# ============================================================================


class TestXGBoostModel:
    """Tests for XGBoostModel."""

    def test_is_price_model(self):
        """Test XGBoostModel inherits from PriceModel."""
        model = XGBoostModel()
        assert isinstance(model, PriceModel)

    def test_name(self):
        """Test model name."""
        assert XGBoostModel().name == "xgboost"

    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X, y = sample_data
        params = {
            "n_estimators": 50,
            "max_depth": 3,
            "learning_rate": 0.1,
            "verbosity": 0,
        }
        model = XGBoostModel(params=params)

        model.fit(X[:200], y[:200], X[200:250], y[200:250])
        assert model.is_fitted

        preds = model.predict(X[250:])
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)

    def test_predict_unfitted_raises(self):
        """Test prediction before fitting raises error."""
        model = XGBoostModel()
        X = pd.DataFrame({"feature1": [1, 2]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_feature_importance(self, sample_data):
        """Test feature importance."""
        X, y = sample_data
        model = XGBoostModel(params={"n_estimators": 50, "verbosity": 0})
        model.fit(X[:200], y[:200])

        importance = model.get_feature_importance()
        assert len(importance) > 0
        # Values should sum to ~1.0
        assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        model = XGBoostModel(params={"n_estimators": 50, "verbosity": 0})
        model.fit(X[:200], y[:200])

        path = tmp_path / "xgb.joblib"
        model.save(path)

        loaded = XGBoostModel.load(path)
        assert loaded.is_fitted

        orig_preds = model.predict(X[200:210])
        loaded_preds = loaded.predict(X[200:210])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)


# ============================================================================
# CatBoostModel Tests
# ============================================================================


class TestCatBoostModel:
    """Tests for CatBoostModel."""

    def test_is_price_model(self):
        """Test CatBoostModel inherits from PriceModel."""
        model = CatBoostModel()
        assert isinstance(model, PriceModel)

    def test_name(self):
        """Test model name."""
        assert CatBoostModel().name == "catboost"

    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X, y = sample_data
        params = {
            "iterations": 50,
            "depth": 3,
            "learning_rate": 0.1,
            "verbose": 0,
            "allow_writing_files": False,
        }
        model = CatBoostModel(params=params)

        model.fit(X[:200], y[:200], X[200:250], y[200:250])
        assert model.is_fitted

        preds = model.predict(X[250:])
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)

    def test_predict_unfitted_raises(self):
        """Test prediction before fitting raises error."""
        model = CatBoostModel()
        X = pd.DataFrame({"feature1": [1, 2]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_feature_importance(self, sample_data):
        """Test feature importance."""
        X, y = sample_data
        params = {"iterations": 50, "verbose": 0, "allow_writing_files": False}
        model = CatBoostModel(params=params)
        model.fit(X[:200], y[:200])

        importance = model.get_feature_importance()
        assert len(importance) > 0

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        params = {"iterations": 50, "verbose": 0, "allow_writing_files": False}
        model = CatBoostModel(params=params)
        model.fit(X[:200], y[:200])

        path = tmp_path / "catboost.joblib"
        model.save(path)

        loaded = CatBoostModel.load(path)
        assert loaded.is_fitted

        orig_preds = model.predict(X[200:210])
        loaded_preds = loaded.predict(X[200:210])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)


# ============================================================================
# Custom Objectives Tests
# ============================================================================


class TestCustomObjectives:
    """Tests for custom loss functions."""

    def test_asymmetric_huber_gradient_shape(self):
        """Test that gradient and hessian have correct shapes."""

        class FakeDataset:
            def get_label(self):
                return np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        y_pred = np.array([1.5, 1.8, 3.5, 3.0, 5.5])
        dtrain = FakeDataset()

        grad, hess = asymmetric_huber_objective(y_pred, dtrain)

        assert grad.shape == (5,)
        assert hess.shape == (5,)
        assert all(np.isfinite(grad))
        assert all(np.isfinite(hess))
        # Hessians must be positive for convergence
        assert all(h > 0 for h in hess)

    def test_asymmetric_penalty_direction(self):
        """Test that under-prediction is penalized more heavily."""

        class FakeDataset:
            def get_label(self):
                return np.array([3.0, 3.0])

        dtrain = FakeDataset()

        # Under-prediction (true=3, pred=2, residual=+1)
        # Over-prediction (true=3, pred=4, residual=-1)
        y_pred = np.array([2.0, 4.0])

        grad, _ = asymmetric_huber_objective(y_pred, dtrain, delta=2.0, under_penalty=2.0)

        # Under-prediction gradient should be larger in magnitude
        assert abs(grad[0]) > abs(grad[1])

    def test_symmetric_when_penalty_is_one(self):
        """Test symmetry when under_penalty=1.0."""

        class FakeDataset:
            def get_label(self):
                return np.array([3.0, 3.0])

        dtrain = FakeDataset()
        y_pred = np.array([2.0, 4.0])  # symmetric errors

        grad, hess = asymmetric_huber_objective(y_pred, dtrain, delta=2.0, under_penalty=1.0)

        # Gradients should be equal in magnitude but opposite sign
        np.testing.assert_almost_equal(abs(grad[0]), abs(grad[1]))

    def test_metric_returns_correct_format(self):
        """Test that metric returns (name, value, is_higher_better)."""

        class FakeDataset:
            def get_label(self):
                return np.array([1.0, 2.0, 3.0])

        y_pred = np.array([1.1, 2.2, 2.8])
        dtrain = FakeDataset()

        name, value, is_higher_better = asymmetric_huber_metric(y_pred, dtrain)

        assert name == "asymmetric_mae"
        assert isinstance(value, float)
        assert value > 0
        assert is_higher_better is False

    def test_make_asymmetric_huber_creates_closures(self):
        """Test factory function creates working objective/metric pair."""
        obj_fn, metric_fn = make_asymmetric_huber(delta=1.0, under_penalty=1.5)

        class FakeDataset:
            def get_label(self):
                return np.array([1.0, 2.0])

        dtrain = FakeDataset()
        y_pred = np.array([1.5, 1.5])

        grad, hess = obj_fn(y_pred, dtrain)
        assert grad.shape == (2,)

        name, val, higher = metric_fn(y_pred, dtrain)
        assert name == "asymmetric_mae"


# ============================================================================
# ResidualModel Tests
# ============================================================================


class TestResidualModel:
    """Tests for two-stage residual model."""

    def test_is_price_model(self):
        """Test ResidualModel inherits from PriceModel."""
        model = ResidualModel()
        assert isinstance(model, PriceModel)

    def test_name(self):
        """Test model name."""
        assert ResidualModel().name == "residual"

    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X, y = sample_data

        coarse_params = {"n_estimators": 50, "num_leaves": 7, "verbose": -1}
        refiner_params = {"n_estimators": 50, "num_leaves": 15, "verbose": -1}

        model = ResidualModel(
            coarse_params=coarse_params,
            refiner_params=refiner_params,
        )

        model.fit(X[:200], y[:200], X[200:250], y[200:250])
        assert model.is_fitted

        preds = model.predict(X[250:])
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)

    def test_predict_unfitted_raises(self):
        """Test prediction before fitting raises error."""
        model = ResidualModel()
        X = pd.DataFrame({"feature1": [1, 2]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_coarse_features_partitioned(self, sample_data):
        """Test that features are correctly partitioned between stages."""
        X, y = sample_data

        model = ResidualModel(
            coarse_params={"n_estimators": 20, "num_leaves": 7, "verbose": -1},
            refiner_params={"n_estimators": 20, "num_leaves": 7, "verbose": -1},
        )
        model.fit(X[:200], y[:200])

        # Coarse features should be event pricing features
        assert "event_section_median_price" in model._actual_coarse_features
        assert "event_zone_median_price" in model._actual_coarse_features

        # Refiner features should NOT contain coarse features
        for f in model._actual_coarse_features:
            assert f not in model._refiner_features

    def test_feature_importance(self, sample_data):
        """Test feature importance combines both stages."""
        X, y = sample_data
        model = ResidualModel(
            coarse_params={"n_estimators": 20, "num_leaves": 7, "verbose": -1},
            refiner_params={"n_estimators": 20, "num_leaves": 7, "verbose": -1},
        )
        model.fit(X[:200], y[:200])

        importance = model.get_feature_importance()
        assert len(importance) > 0
        # Should have features from both stages
        has_coarse = any(f in importance for f in model._actual_coarse_features)
        has_refiner = any(f in importance for f in model._refiner_features)
        assert has_coarse
        assert has_refiner

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        model = ResidualModel(
            coarse_params={"n_estimators": 20, "num_leaves": 7, "verbose": -1},
            refiner_params={"n_estimators": 20, "num_leaves": 7, "verbose": -1},
        )
        model.fit(X[:200], y[:200])

        path = tmp_path / "residual.joblib"
        model.save(path)

        loaded = ResidualModel.load(path)
        assert loaded.is_fitted

        orig_preds = model.predict(X[200:210])
        loaded_preds = loaded.predict(X[200:210])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_raises_without_coarse_features(self):
        """Test error when no coarse features exist in data."""
        X = pd.DataFrame({"feat_a": [1, 2, 3], "feat_b": [4, 5, 6]})
        y = pd.Series([1.0, 2.0, 3.0])

        model = ResidualModel()
        with pytest.raises(ValueError, match="No coarse features"):
            model.fit(X, y)


# ============================================================================
# StackingEnsemble Tests
# ============================================================================


class TestStackingEnsemble:
    """Tests for stacking ensemble model."""

    def _make_simple_configs(self):
        """Create minimal base learner configs for fast testing."""
        from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel
        from ticket_price_predictor.ml.models.xgboost_model import XGBoostModel

        return [
            {
                "name": "lgb",
                "cls": LightGBMModel,
                "params": {
                    "objective": "regression",
                    "n_estimators": 30,
                    "num_leaves": 7,
                    "learning_rate": 0.1,
                    "verbose": -1,
                    "boosting_type": "gbdt",
                },
            },
            {
                "name": "xgb",
                "cls": XGBoostModel,
                "params": {
                    "n_estimators": 30,
                    "max_depth": 3,
                    "learning_rate": 0.1,
                    "verbosity": 0,
                },
            },
        ]

    def test_is_price_model(self):
        """Test StackingEnsemble inherits from PriceModel."""
        model = StackingEnsemble(base_configs=self._make_simple_configs())
        assert isinstance(model, PriceModel)

    def test_name(self):
        """Test model name."""
        assert StackingEnsemble().name == "stacking"

    def test_fit_and_predict(self, sample_data):
        """Test fitting and prediction."""
        X, y = sample_data

        model = StackingEnsemble(
            base_configs=self._make_simple_configs(),
            n_folds=3,
        )

        model.fit(X[:200], y[:200], X[200:250], y[200:250])
        assert model.is_fitted

        preds = model.predict(X[250:])
        assert len(preds) == 50
        assert all(np.isfinite(p) for p in preds)

    def test_predict_unfitted_raises(self):
        """Test prediction before fitting raises error."""
        model = StackingEnsemble(base_configs=self._make_simple_configs())
        X = pd.DataFrame({"feature1": [1, 2]})
        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_feature_importance(self, sample_data):
        """Test aggregated feature importance."""
        X, y = sample_data
        model = StackingEnsemble(
            base_configs=self._make_simple_configs(),
            n_folds=3,
        )
        model.fit(X[:200], y[:200])

        importance = model.get_feature_importance()
        assert len(importance) > 0

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        model = StackingEnsemble(
            base_configs=self._make_simple_configs(),
            n_folds=3,
        )
        model.fit(X[:200], y[:200])

        path = tmp_path / "stacking.joblib"
        model.save(path)

        loaded = StackingEnsemble.load(path)
        assert loaded.is_fitted

        orig_preds = model.predict(X[200:210])
        loaded_preds = loaded.predict(X[200:210])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_with_anchor_feature(self, sample_data):
        """Test stacking with an anchor feature in meta-learner."""
        X, y = sample_data
        model = StackingEnsemble(
            base_configs=self._make_simple_configs(),
            n_folds=3,
            anchor_feature="event_section_median_price",
        )

        model.fit(X[:200], y[:200])
        preds = model.predict(X[200:210])
        assert len(preds) == 10

    def test_temporal_kfold_indices(self):
        """Test temporal K-fold index generation."""
        model = StackingEnsemble(n_folds=3)
        folds = model._temporal_kfold_indices(100)

        assert len(folds) == 3

        for train_idx, val_idx in folds:
            # No overlap
            assert len(set(train_idx) & set(val_idx)) == 0
            # Train always before val (temporal ordering)
            assert train_idx.max() < val_idx.min()


# ============================================================================
# Trainer Integration Tests
# ============================================================================


class TestTrainerNewModels:
    """Test ModelTrainer supports new model types."""

    def test_create_xgboost_model(self):
        """Test trainer creates XGBoost model."""
        from ticket_price_predictor.ml.training.trainer import ModelTrainer

        trainer = ModelTrainer(model_type="xgboost")
        model = trainer._create_model()
        assert isinstance(model, XGBoostModel)

    def test_create_catboost_model(self):
        """Test trainer creates CatBoost model."""
        from ticket_price_predictor.ml.training.trainer import ModelTrainer

        trainer = ModelTrainer(model_type="catboost")
        model = trainer._create_model()
        assert isinstance(model, CatBoostModel)

    def test_create_stacking_model(self):
        """Test trainer creates stacking model."""
        from ticket_price_predictor.ml.training.trainer import ModelTrainer

        trainer = ModelTrainer(model_type="stacking")
        model = trainer._create_model()
        assert isinstance(model, StackingEnsemble)

    def test_create_residual_model(self):
        """Test trainer creates residual model."""
        from ticket_price_predictor.ml.training.trainer import ModelTrainer

        trainer = ModelTrainer(model_type="residual")
        model = trainer._create_model()
        assert isinstance(model, ResidualModel)
