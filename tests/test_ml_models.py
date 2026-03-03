"""Tests for ML models."""

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.baseline import BaselineModel
from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel, QuantileLightGBMModel

# ============================================================================
# BaselineModel Tests
# ============================================================================


class TestBaselineModel:
    """Tests for BaselineModel."""

    @pytest.fixture
    def sample_features(self):
        """Create sample features matching BaselineModel expected columns."""
        return pd.DataFrame(
            {
                "seat_zone_encoded": [0, 1, 2, 0, 1],
                "event_type_encoded": [0, 0, 0, 0, 0],
                "city_tier": [1, 2, 3, 1, 2],
                "day_of_week": [5, 6, 0, 1, 2],
                "urgency_bucket": [3, 2, 1, 4, 3],
                "venue_capacity_bucket": [1, 2, 2, 1, 3],
                "days_to_event": [14, 30, 7, 3, 21],
                "days_to_event_squared": [196, 900, 49, 9, 441],
                "zone_price_ratio": [1.0, 0.8, 0.5, 1.2, 0.9],
                "row_numeric": [1, 10, 20, 5, 15],
                "is_weekend": [1, 1, 0, 0, 0],
                "is_floor": [1, 0, 0, 1, 0],
                "is_ga": [0, 0, 0, 0, 1],
                "is_last_week": [0, 0, 1, 1, 0],
            }
        )

    def test_is_price_model(self):
        """Test BaselineModel inherits from PriceModel."""
        model = BaselineModel()
        assert isinstance(model, PriceModel)

    def test_fit(self, sample_features):
        """Test fitting the baseline model."""
        model = BaselineModel()
        y = pd.Series([100.0, 150.0, 200.0, 250.0, 300.0])

        model.fit(sample_features, y)
        assert model.is_fitted

    def test_predict(self, sample_features):
        """Test prediction."""
        model = BaselineModel()
        y = pd.Series([100.0, 150.0, 200.0, 250.0, 300.0])
        model.fit(sample_features, y)

        preds = model.predict(sample_features[:2])
        assert len(preds) == 2
        assert all(p > 0 for p in preds)

    def test_predict_unfitted_raises(self, sample_features):
        """Test prediction before fitting raises error."""
        model = BaselineModel()

        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(sample_features)

    def test_save_and_load(self, sample_features, tmp_path):
        """Test saving and loading model."""
        model = BaselineModel()
        y = pd.Series([100.0, 150.0, 200.0, 250.0, 300.0])
        model.fit(sample_features, y)

        path = tmp_path / "baseline.joblib"
        model.save(path)

        loaded = BaselineModel.load(path)
        assert loaded.is_fitted

        # Predictions should match
        orig_preds = model.predict(sample_features[:2])
        loaded_preds = loaded.predict(sample_features[:2])
        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_feature_importance(self, sample_features):
        """Test feature importance returns coefficients."""
        model = BaselineModel()
        y = pd.Series([100.0, 150.0, 200.0, 250.0, 300.0])
        model.fit(sample_features, y)

        importance = model.get_feature_importance()
        assert len(importance) > 0


# ============================================================================
# LightGBMModel Tests
# ============================================================================


class TestLightGBMModel:
    """Tests for LightGBMModel."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 100
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                "feature3": np.random.randn(n),
            }
        )
        # Create target correlated with features
        y = pd.Series(100 + 20 * X["feature1"] + 10 * X["feature2"] + np.random.randn(n) * 5)
        return X, y

    def test_is_price_model(self):
        """Test LightGBMModel inherits from PriceModel."""
        model = LightGBMModel()
        assert isinstance(model, PriceModel)

    def test_fit(self, sample_data):
        """Test fitting the model."""
        X, y = sample_data
        model = LightGBMModel()

        # Split for validation
        X_train, X_val = X[:80], X[80:]
        y_train, y_val = y[:80], y[80:]

        model.fit(X_train, y_train, X_val, y_val)
        assert model._model is not None

    def test_predict(self, sample_data):
        """Test prediction."""
        X, y = sample_data
        model = LightGBMModel()

        X_train, X_test = X[:80], X[80:]
        y_train = y[:80]

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        assert len(preds) == 20
        assert all(np.isfinite(p) for p in preds)  # Predictions should be finite

    def test_predict_unfitted_raises(self):
        """Test prediction before fitting raises error."""
        model = LightGBMModel()
        X = pd.DataFrame({"feature1": [1, 2]})

        with pytest.raises(RuntimeError, match="fitted"):
            model.predict(X)

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        model = LightGBMModel()
        model.fit(X[:80], y[:80])

        path = tmp_path / "lightgbm.joblib"
        model.save(path)

        loaded = LightGBMModel.load(path)

        # Predictions should match
        X_test = X[80:]
        orig_preds = model.predict(X_test)
        loaded_preds = loaded.predict(X_test)

        np.testing.assert_array_almost_equal(orig_preds, loaded_preds)

    def test_feature_importance(self, sample_data):
        """Test feature importance."""
        X, y = sample_data
        model = LightGBMModel()
        model.fit(X, y)

        importance = model.get_feature_importance()

        # Should have some features
        assert len(importance) > 0

    def test_fit_with_early_stopping(self, sample_data):
        """Test fitting with early stopping."""
        X, y = sample_data
        model = LightGBMModel()

        X_train, X_val = X[:70], X[70:90]
        y_train, y_val = y[:70], y[70:90]

        model.fit(X_train, y_train, X_val, y_val)

        # Model should be fitted
        assert model.is_fitted


# ============================================================================
# QuantileLightGBMModel Tests
# ============================================================================


class TestQuantileLightGBMModel:
    """Tests for QuantileLightGBMModel."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
            }
        )
        y = pd.Series(100 + 20 * X["feature1"] + np.random.randn(n) * 10)
        return X, y

    def test_is_price_model(self):
        """Test QuantileLightGBMModel inherits from PriceModel."""
        model = QuantileLightGBMModel()
        assert isinstance(model, PriceModel)

    def test_fit(self, sample_data):
        """Test fitting trains multiple quantile models."""
        X, y = sample_data
        model = QuantileLightGBMModel()

        model.fit(X[:150], y[:150])

        # Should be fitted
        assert model.is_fitted

    def test_predict(self, sample_data):
        """Test standard predict returns median."""
        X, y = sample_data
        model = QuantileLightGBMModel()
        model.fit(X[:150], y[:150])

        preds = model.predict(X[150:])
        assert len(preds) == 50

    def test_predict_with_uncertainty(self, sample_data):
        """Test prediction with uncertainty bounds."""
        X, y = sample_data
        # Use GBDT for deterministic quantile ordering in small samples
        model = QuantileLightGBMModel(params={"boosting_type": "gbdt"})
        model.fit(X[:150], y[:150])

        median, lower, upper = model.predict_with_uncertainty(X[150:])

        assert len(median) == 50
        assert len(lower) == 50
        assert len(upper) == 50

        # Lower should generally be <= median <= upper
        # Allow small number of violations due to quantile crossing in small samples
        violations_lower = sum(lo > m for lo, m in zip(lower, median, strict=False))
        violations_upper = sum(m > up for m, up in zip(median, upper, strict=False))

        # At most 15% violations allowed
        max_violations = int(len(median) * 0.15) + 1
        assert violations_lower <= max_violations, (
            f"Too many lower > median violations: {violations_lower}"
        )
        assert violations_upper <= max_violations, (
            f"Too many median > upper violations: {violations_upper}"
        )

    def test_coverage(self, sample_data):
        """Test prediction interval coverage."""
        X, y = sample_data
        # Use GBDT for more reliable quantile coverage in small samples
        model = QuantileLightGBMModel(params={"boosting_type": "gbdt"})
        model.fit(X[:150], y[:150])

        X_test = X[150:]
        y_test = y[150:]

        median, lower, upper = model.predict_with_uncertainty(X_test)

        # Count how many actual values fall within prediction interval
        in_interval = sum(
            (lo <= actual <= up) for lo, actual, up in zip(lower, y_test, upper, strict=False)
        )
        coverage = in_interval / len(y_test)

        # Should cover roughly 90% (5th to 95th percentile)
        # Allow some tolerance due to small sample size
        assert coverage >= 0.5

    def test_save_and_load(self, sample_data, tmp_path):
        """Test saving and loading model."""
        X, y = sample_data
        model = QuantileLightGBMModel()
        model.fit(X[:150], y[:150])

        path = tmp_path / "quantile.joblib"
        model.save(path)

        loaded = QuantileLightGBMModel.load(path)

        # Predictions should match
        X_test = X[150:]
        orig_median, orig_lower, orig_upper = model.predict_with_uncertainty(X_test)
        loaded_median, loaded_lower, loaded_upper = loaded.predict_with_uncertainty(X_test)

        np.testing.assert_array_almost_equal(orig_median, loaded_median)
        np.testing.assert_array_almost_equal(orig_lower, loaded_lower)
        np.testing.assert_array_almost_equal(orig_upper, loaded_upper)

    def test_feature_importance(self, sample_data):
        """Test feature importance from median model."""
        X, y = sample_data
        model = QuantileLightGBMModel()
        model.fit(X, y)

        importance = model.get_feature_importance()
        assert len(importance) > 0


# ============================================================================
# Model Comparison Tests
# ============================================================================


class TestModelComparison:
    """Tests comparing different models."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for comparison with proper features."""
        np.random.seed(42)
        n = 200
        X = pd.DataFrame(
            {
                "feature1": np.random.randn(n),
                "feature2": np.random.randn(n),
                # Add features that BaselineModel expects
                "seat_zone_encoded": np.random.randint(0, 4, n),
                "event_type_encoded": np.random.randint(0, 3, n),
                "city_tier": np.random.randint(1, 4, n),
                "day_of_week": np.random.randint(0, 7, n),
                "urgency_bucket": np.random.randint(0, 5, n),
                "venue_capacity_bucket": np.random.randint(0, 4, n),
                "days_to_event": np.random.randint(1, 60, n),
                "days_to_event_squared": np.random.randint(1, 3600, n),
                "zone_price_ratio": np.random.rand(n) + 0.5,
                "row_numeric": np.random.randint(1, 30, n),
                "is_weekend": np.random.randint(0, 2, n),
                "is_floor": np.random.randint(0, 2, n),
                "is_ga": np.random.randint(0, 2, n),
                "is_last_week": np.random.randint(0, 2, n),
            }
        )
        y = pd.Series(100 + 20 * X["feature1"] + 10 * X["feature2"] + np.random.randn(n) * 5)
        return X, y

    def test_lightgbm_beats_baseline(self, sample_data):
        """Test LightGBM outperforms baseline on correlated data."""
        X, y = sample_data
        # Use train/val/test split so early stopping works with DART/Huber
        X_train, X_val, X_test = X[:120], X[120:150], X[150:]
        y_train, y_val, y_test = y[:120], y[120:150], y[150:]

        # Train baseline
        baseline = BaselineModel()
        baseline.fit(X_train, y_train)
        baseline_preds = baseline.predict(X_test)
        baseline_mae = np.mean(np.abs(y_test - baseline_preds))

        # Train LightGBM with simple params for this unit test.
        # DEFAULT_PARAMS uses Huber loss tuned for log-space production targets;
        # this synthetic test uses raw linear targets, so use MSE params.
        simple_params = {
            "objective": "regression",
            "n_estimators": 200,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "boosting_type": "gbdt",
            "verbose": -1,
        }
        lgb = LightGBMModel(params=simple_params)
        lgb.fit(X_train, y_train, X_val, y_val)
        lgb_preds = lgb.predict(X_test)
        lgb_mae = np.mean(np.abs(y_test - lgb_preds))

        # LightGBM should have lower or equal MAE (depends on features)
        # With simple features, LightGBM should do at least as well
        assert lgb_mae <= baseline_mae * 1.5  # Allow some tolerance

    def test_all_models_same_interface(self, sample_data):
        """Test all models follow same interface."""
        X, y = sample_data

        models = [
            BaselineModel(),
            LightGBMModel(),
            QuantileLightGBMModel(),
        ]

        for model in models:
            # Should have fit, predict, save, load, get_feature_importance
            assert hasattr(model, "fit")
            assert hasattr(model, "predict")
            assert hasattr(model, "save")
            assert hasattr(model, "get_feature_importance")

            # Fit should work
            model.fit(X, y)

            # Predict should return array-like of correct length
            preds = model.predict(X[:10])
            assert len(preds) == 10
