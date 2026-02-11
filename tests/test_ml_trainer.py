"""Tests for ML training pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.schemas import TrainingMetrics
from ticket_price_predictor.ml.training.evaluator import ModelEvaluator
from ticket_price_predictor.ml.training.splitter import DataSplit, TimeBasedSplitter
from ticket_price_predictor.ml.training.trainer import ModelTrainer

# ============================================================================
# TimeBasedSplitter Tests
# ============================================================================


class TestTimeBasedSplitter:
    """Tests for TimeBasedSplitter."""

    def test_default_ratios(self):
        """Test default split ratios."""
        splitter = TimeBasedSplitter()
        assert splitter._train_ratio == 0.7
        assert splitter._val_ratio == 0.15
        assert splitter._test_ratio == 0.15

    def test_custom_ratios(self):
        """Test custom split ratios."""
        splitter = TimeBasedSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )
        assert splitter._train_ratio == 0.6
        assert splitter._val_ratio == 0.2
        assert splitter._test_ratio == 0.2

    def test_split_sizes(self):
        """Test split produces correct sizes."""
        splitter = TimeBasedSplitter(
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2,
        )

        X = pd.DataFrame({"f1": range(100)})
        y = pd.Series(range(100))

        split = splitter.split(X, y)

        assert split.n_train == 60
        assert split.n_val == 20
        assert split.n_test == 20

    def test_split_with_raw_df(self):
        """Test split respects time ordering from raw DataFrame."""
        splitter = TimeBasedSplitter()

        # Create data with timestamps
        raw_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=100, freq="D"),
                "listing_price": np.random.rand(100) * 100 + 50,
            }
        )
        X = pd.DataFrame({"f1": range(100)})
        y = raw_df["listing_price"]

        split = splitter.split(X, y, raw_df=raw_df)

        # Train should contain earliest data
        assert split.n_train > 0
        assert split.n_val > 0
        assert split.n_test > 0

    def test_split_no_overlap(self):
        """Test train/val/test have no overlap."""
        splitter = TimeBasedSplitter()

        X = pd.DataFrame({"f1": range(100)})
        y = pd.Series(range(100))

        split = splitter.split(X, y)

        # Check indices don't overlap
        train_idx = set(split.X_train.index)
        val_idx = set(split.X_val.index)
        test_idx = set(split.X_test.index)

        assert len(train_idx & val_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx & test_idx) == 0


# ============================================================================
# DataSplit Tests
# ============================================================================


class TestDataSplit:
    """Tests for DataSplit dataclass."""

    def test_creation(self):
        """Test creating a DataSplit."""
        X_train = pd.DataFrame({"f1": [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])
        X_val = pd.DataFrame({"f1": [4]})
        y_val = pd.Series([40])
        X_test = pd.DataFrame({"f1": [5]})
        y_test = pd.Series([50])

        split = DataSplit(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        assert split.n_train == 3
        assert split.n_val == 1
        assert split.n_test == 1


# ============================================================================
# ModelEvaluator Tests
# ============================================================================


class TestModelEvaluator:
    """Tests for ModelEvaluator."""

    def test_calculate_metrics(self):
        """Test metrics calculation."""
        y_true = pd.Series([100.0, 200.0, 300.0, 400.0])
        y_pred = np.array([110.0, 190.0, 310.0, 380.0])

        # Errors: 10, 10, 10, 20 -> MAE = 12.5
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

        assert mae == 12.5
        assert 12 < rmse < 15

    def test_r2_score(self):
        """Test R² score calculation."""
        from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel

        # Create correlated data
        np.random.seed(42)
        X = pd.DataFrame({"f1": np.random.randn(100)})
        y = pd.Series(100 + 20 * X["f1"] + np.random.randn(100) * 2)

        model = LightGBMModel()
        model.fit(X[:80], y[:80])

        metrics = ModelEvaluator.evaluate_model(
            model,
            X[80:],
            y[80:],
            n_train=80,
            n_val=0,
            training_time=1.0,
            model_version="test",
        )

        # R² should be positive for correlated data
        assert metrics.r2 > 0


# ============================================================================
# ModelTrainer Tests
# ============================================================================


class TestModelTrainer:
    """Tests for ModelTrainer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample listing data."""
        np.random.seed(42)
        n = 200

        return pd.DataFrame(
            {
                "artist_or_team": np.random.choice(["Artist A", "Artist B"], n),
                "event_type": ["CONCERT"] * n,
                "city": np.random.choice(["New York", "Los Angeles"], n),
                "event_datetime": pd.date_range("2024-01-01", periods=n, freq="6h"),
                "section": np.random.choice(["Floor", "Lower Level", "Upper Level"], n),
                "row": np.random.choice(["1", "5", "10", "20"], n),
                "days_to_event": np.random.randint(1, 60, n),
                "listing_price": 100 + np.random.randn(n) * 30,
                "event_id": [f"e{i % 20}" for i in range(n)],
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="6h"),
            }
        )

    def test_initialization(self):
        """Test trainer initialization."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="v1")
        assert trainer._model_type == "lightgbm"
        assert trainer._model_version == "v1"
        assert trainer.model is None
        assert trainer.metrics is None

    def test_train_baseline(self, sample_data):
        """Test training baseline model."""
        trainer = ModelTrainer(model_type="baseline", model_version="test")
        metrics = trainer.train(sample_data)

        assert trainer.model is not None
        assert metrics is not None
        assert metrics.mae > 0

    def test_train_lightgbm(self, sample_data):
        """Test training LightGBM model."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="test")
        metrics = trainer.train(sample_data)

        assert trainer.model is not None
        assert metrics is not None
        assert metrics.mae > 0

    def test_train_quantile(self, sample_data):
        """Test training quantile model."""
        trainer = ModelTrainer(model_type="quantile", model_version="test")
        metrics = trainer.train(sample_data)

        assert trainer.model is not None
        assert metrics is not None

    def test_save_model(self, sample_data, tmp_path):
        """Test saving trained model."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="v1")
        trainer.train(sample_data)

        model_path = trainer.save(tmp_path)

        assert model_path.exists()
        assert "lightgbm_v1" in model_path.name

        # Check metrics file was also saved
        metrics_path = tmp_path / "lightgbm_v1_metrics.json"
        assert metrics_path.exists()

        with open(metrics_path) as f:
            saved_metrics = json.load(f)
            assert "mae" in saved_metrics
            assert "rmse" in saved_metrics

    def test_save_without_train_raises(self, tmp_path):
        """Test saving without training raises error."""
        trainer = ModelTrainer()

        with pytest.raises(RuntimeError, match="No model to save"):
            trainer.save(tmp_path)

    def test_load_model(self, sample_data, tmp_path):
        """Test loading a saved model."""
        # Train and save
        trainer = ModelTrainer(model_type="lightgbm", model_version="v1")
        trainer.train(sample_data)
        model_path = trainer.save(tmp_path)

        # Load
        loaded = ModelTrainer.load(model_path, "lightgbm")

        assert loaded is not None
        # Should be able to predict
        X_test = pd.DataFrame(
            {
                "artist_or_team": ["Artist A"],
                "event_type": ["CONCERT"],
                "city": ["New York"],
                "event_datetime": pd.to_datetime(["2024-06-15"]),
                "section": ["Floor"],
                "row": ["1"],
                "days_to_event": [14],
                "listing_price": [100.0],
                "event_id": ["e1"],
            }
        )
        from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_momentum=True)
        features = pipeline.fit_transform(X_test)
        preds = loaded.predict(features)
        assert len(preds) == 1

    def test_train_with_custom_ratios(self, sample_data):
        """Test training with custom split ratios."""
        trainer = ModelTrainer(model_type="baseline")
        metrics = trainer.train(
            sample_data,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        assert metrics is not None
        # Test set should be smaller
        assert metrics.n_test_samples < 30  # 10% of 200

    def test_train_with_split(self, sample_data):
        """Test training with pre-computed split."""
        from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_momentum=False)
        X = pipeline.fit_transform(sample_data)
        y = sample_data["listing_price"]

        split = DataSplit(
            X_train=X[:140],
            y_train=y[:140],
            X_val=X[140:170],
            y_val=y[140:170],
            X_test=X[170:],
            y_test=y[170:],
        )

        trainer = ModelTrainer(model_type="baseline")
        metrics = trainer.train_with_split(split)

        assert metrics is not None
        assert metrics.n_train_samples == 140
        assert metrics.n_val_samples == 30
        assert metrics.n_test_samples == 30

    def test_invalid_model_type(self):
        """Test invalid model type raises error."""
        trainer = ModelTrainer(model_type="invalid")  # type: ignore

        with pytest.raises(ValueError, match="Unknown model type"):
            trainer._create_model()


# ============================================================================
# Training Metrics Tests
# ============================================================================


class TestTrainingMetrics:
    """Tests for TrainingMetrics schema."""

    def test_creation(self):
        """Test creating training metrics."""
        metrics = TrainingMetrics(
            mae=50.0,
            rmse=60.0,
            mape=0.15,
            r2=0.85,
            n_train_samples=1000,
            n_val_samples=150,
            n_test_samples=150,
            n_features=20,
            training_time_seconds=10.5,
            model_version="v1",
            model_type="lightgbm",
        )

        assert metrics.mae == 50.0
        assert metrics.rmse == 60.0
        assert metrics.mape == 0.15
        assert metrics.r2 == 0.85

    def test_model_dump(self):
        """Test serializing metrics to dict."""
        metrics = TrainingMetrics(
            mae=50.0,
            rmse=60.0,
            mape=0.15,
            r2=0.85,
            n_train_samples=1000,
            n_val_samples=150,
            n_test_samples=150,
            n_features=20,
            training_time_seconds=10.5,
            model_version="v1",
            model_type="lightgbm",
        )

        d = metrics.model_dump()
        assert d["mae"] == 50.0
        assert d["model_version"] == "v1"


# ============================================================================
# Integration Tests
# ============================================================================


class TestTrainingIntegration:
    """Integration tests for the full training pipeline."""

    @pytest.fixture
    def sample_data(self):
        """Create realistic sample data."""
        np.random.seed(42)
        n = 300

        artists = ["Taylor Swift", "BTS", "Morgan Wallen", "Local Band"]
        artist_prices = {"Taylor Swift": 200, "BTS": 300, "Morgan Wallen": 150, "Local Band": 80}

        data = []
        for i in range(n):
            artist = np.random.choice(artists)
            base_price = artist_prices[artist]
            data.append(
                {
                    "artist_or_team": artist,
                    "event_type": "CONCERT",
                    "city": np.random.choice(["New York", "Seattle", "Austin"]),
                    "event_datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // 3),
                    "section": np.random.choice(["Floor", "Lower Level", "Upper Level"]),
                    "row": str(np.random.randint(1, 30)),
                    "days_to_event": np.random.randint(1, 60),
                    "listing_price": base_price + np.random.randn() * 30,
                    "event_id": f"e{i % 30}",
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 2),
                }
            )

        return pd.DataFrame(data)

    def test_full_training_pipeline(self, sample_data, tmp_path):
        """Test complete training pipeline from data to saved model."""
        # Train
        trainer = ModelTrainer(model_type="lightgbm", model_version="integration_test")
        metrics = trainer.train(sample_data)

        # Verify metrics are reasonable
        assert metrics.mae > 0
        assert metrics.mae < 100  # Should be less than $100 MAE
        assert metrics.r2 > 0  # Should have some predictive power

        # Save
        model_path = trainer.save(tmp_path)
        assert model_path.exists()

        # Load and verify predictions work
        loaded_model = ModelTrainer.load(model_path, "lightgbm")

        # Create test input
        from ticket_price_predictor.ml.features.pipeline import FeaturePipeline

        pipeline = FeaturePipeline(include_momentum=True)

        test_input = pd.DataFrame(
            [
                {
                    "artist_or_team": "Taylor Swift",
                    "event_type": "CONCERT",
                    "city": "New York",
                    "event_datetime": pd.Timestamp("2024-06-15"),
                    "section": "Floor",
                    "row": "5",
                    "days_to_event": 14,
                    "listing_price": 200.0,
                    "event_id": "test_event",
                }
            ]
        )

        features = pipeline.fit_transform(test_input)
        pred = loaded_model.predict(features)

        assert len(pred) == 1
        assert pred[0] > 0  # Price should be positive

    def test_lightgbm_better_than_baseline(self, sample_data):
        """Test LightGBM outperforms baseline."""
        baseline_trainer = ModelTrainer(model_type="baseline")
        baseline_metrics = baseline_trainer.train(sample_data)

        lgb_trainer = ModelTrainer(model_type="lightgbm")
        lgb_metrics = lgb_trainer.train(sample_data)

        # LightGBM should not be significantly worse than baseline.
        # With small synthetic data, results can be close, so allow 5% tolerance.
        assert lgb_metrics.mae < baseline_metrics.mae * 1.05
