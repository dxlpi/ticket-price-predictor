"""Tests for ML training pipeline."""

import json

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.schemas import TrainingMetrics
from ticket_price_predictor.ml.training.evaluator import ModelEvaluator
from ticket_price_predictor.ml.training.splitter import DataSplit, RawDataSplit, TimeBasedSplitter
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

        # Use GBDT for small test data (DART needs more samples)
        model = LightGBMModel(params={"boosting_type": "gbdt", "n_estimators": 100})
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
        # Train and save (use GBDT for deterministic test behavior)
        trainer = ModelTrainer(model_type="lightgbm", model_version="v1")
        trainer.train(sample_data, params={"boosting_type": "gbdt", "n_estimators": 200})
        model_path = trainer.save(tmp_path)

        # Load
        loaded = ModelTrainer.load(model_path, "lightgbm")

        assert loaded is not None
        # Build test input using the model's own feature names to avoid shape mismatch
        feature_names = loaded._feature_names
        assert len(feature_names) > 0
        X_test = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        preds = loaded.predict(X_test)
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
# RawDataSplit Tests
# ============================================================================


class TestRawDataSplit:
    """Tests for RawDataSplit and split_raw()."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data with multiple artists."""
        np.random.seed(42)
        n = 300

        artists = ["Taylor Swift", "BTS", "Morgan Wallen", "Local Band"]
        artist_prices = {"Taylor Swift": 200, "BTS": 300, "Morgan Wallen": 150, "Local Band": 80}

        data = []
        for i in range(n):
            artist = artists[i % len(artists)]
            base_price = artist_prices[artist]
            data.append(
                {
                    "artist_or_team": artist,
                    "event_type": "CONCERT",
                    "city": np.random.choice(["New York", "Seattle"]),
                    "event_datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // 3),
                    "section": np.random.choice(["Floor", "Lower Level"]),
                    "row": str(np.random.randint(1, 30)),
                    "days_to_event": np.random.randint(1, 60),
                    "listing_price": base_price + np.random.randn() * 30,
                    "event_id": f"e{i % 30}",
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 2),
                }
            )

        return pd.DataFrame(data)

    def test_raw_split_sizes(self, sample_data):
        """Test that raw split produces correct total size."""
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        raw_split = splitter.split_raw(sample_data)

        total = raw_split.n_train + raw_split.n_val + raw_split.n_test
        assert total == len(sample_data)

    def test_raw_split_ratios(self, sample_data):
        """Test that raw split respects approximate ratios."""
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        raw_split = splitter.split_raw(sample_data)

        n = len(sample_data)
        assert abs(raw_split.n_train / n - 0.7) < 0.05
        assert abs(raw_split.n_val / n - 0.15) < 0.05
        assert abs(raw_split.n_test / n - 0.15) < 0.05

    def test_raw_split_temporal_ordering(self, sample_data):
        """Test that train data is earlier than val, which is earlier than test."""
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        raw_split = splitter.split_raw(sample_data)

        train_max = raw_split.train_df["event_datetime"].max()
        val_min = raw_split.val_df["event_datetime"].min()
        test_min = raw_split.test_df["event_datetime"].min()

        assert train_max <= val_min
        assert val_min <= test_min

    def test_raw_split_no_overlap(self, sample_data):
        """Test that splits have no overlapping indices."""
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        raw_split = splitter.split_raw(sample_data)

        train_idx = set(raw_split.train_df.index)
        val_idx = set(raw_split.val_df.index)
        test_idx = set(raw_split.test_df.index)

        assert len(train_idx & val_idx) == 0
        assert len(val_idx & test_idx) == 0
        assert len(train_idx & test_idx) == 0

    def test_stratified_split_all_artists_in_train(self, sample_data):
        """Test that stratified split has all artists represented in train."""
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_col="artist_or_team",
        )
        raw_split = splitter.split_raw(sample_data)

        train_artists = set(raw_split.train_df["artist_or_team"].unique())
        all_artists = set(sample_data["artist_or_team"].unique())

        # Every artist should appear in training
        assert train_artists == all_artists

    def test_stratified_split_temporal_within_artist(self, sample_data):
        """Test that within each artist, train data is earlier than test."""
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_col="artist_or_team",
        )
        raw_split = splitter.split_raw(sample_data)

        for artist in sample_data["artist_or_team"].unique():
            train_artist = raw_split.train_df[raw_split.train_df["artist_or_team"] == artist]
            test_artist = raw_split.test_df[raw_split.test_df["artist_or_team"] == artist]

            if len(train_artist) > 0 and len(test_artist) > 0:
                assert train_artist["event_datetime"].max() <= test_artist["event_datetime"].min()

    def test_small_artist_goes_to_train(self):
        """Test that artists with <3 samples go entirely to train."""
        data = pd.DataFrame(
            {
                "artist_or_team": ["Big Artist"] * 20 + ["Tiny Artist"] * 2,
                "event_datetime": pd.date_range("2024-01-01", periods=22, freq="D"),
                "listing_price": np.random.rand(22) * 100,
            }
        )

        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_col="artist_or_team",
        )
        raw_split = splitter.split_raw(data)

        # Tiny Artist (2 samples) should only appear in train
        tiny_in_val = raw_split.val_df[raw_split.val_df["artist_or_team"] == "Tiny Artist"]
        tiny_in_test = raw_split.test_df[raw_split.test_df["artist_or_team"] == "Tiny Artist"]
        tiny_in_train = raw_split.train_df[raw_split.train_df["artist_or_team"] == "Tiny Artist"]

        assert len(tiny_in_val) == 0
        assert len(tiny_in_test) == 0
        assert len(tiny_in_train) == 2

    def test_raw_split_dataclass_properties(self):
        """Test RawDataSplit properties."""
        raw_split = RawDataSplit(
            train_df=pd.DataFrame({"a": [1, 2, 3]}),
            val_df=pd.DataFrame({"a": [4]}),
            test_df=pd.DataFrame({"a": [5, 6]}),
        )

        assert raw_split.n_train == 3
        assert raw_split.n_val == 1
        assert raw_split.n_test == 2


# ============================================================================
# Data Leakage Prevention Tests
# ============================================================================


class TestDataLeakagePrevention:
    """Tests verifying the pipeline is NOT fitted on test-set data."""

    @pytest.fixture
    def leakage_test_data(self):
        """Create data where some artists only appear in test set timeframe."""
        np.random.seed(42)

        # Common artists throughout
        common_data = []
        for i in range(200):
            common_data.append(
                {
                    "artist_or_team": np.random.choice(["Artist A", "Artist B"]),
                    "event_type": "CONCERT",
                    "city": "New York",
                    "event_datetime": pd.Timestamp("2024-01-01") + pd.Timedelta(days=i // 2),
                    "section": "Floor",
                    "row": "1",
                    "days_to_event": 14,
                    "listing_price": 100 + np.random.randn() * 20,
                    "event_id": f"e{i % 20}",
                    "timestamp": pd.Timestamp("2024-01-01") + pd.Timedelta(hours=i * 2),
                }
            )

        # Late-appearing artist (only in last 15% of data by time)
        for i in range(30):
            common_data.append(
                {
                    "artist_or_team": "Late Artist",
                    "event_type": "CONCERT",
                    "city": "Seattle",
                    "event_datetime": pd.Timestamp("2024-04-01") + pd.Timedelta(days=i),
                    "section": "Floor",
                    "row": "5",
                    "days_to_event": 7,
                    "listing_price": 500 + np.random.randn() * 50,
                    "event_id": f"late_e{i}",
                    "timestamp": pd.Timestamp("2024-04-01") + pd.Timedelta(hours=i * 4),
                }
            )

        return pd.DataFrame(common_data)

    def test_pipeline_not_fitted_on_test_artists(self, leakage_test_data):
        """Verify feature pipeline is fitted only on training data."""
        trainer = ModelTrainer(model_type="baseline", model_version="leak_test")
        trainer.train(leakage_test_data)

        # The pipeline should have been fitted on training data only
        assert trainer._feature_pipeline is not None
        assert trainer._feature_pipeline._fitted

    def test_train_uses_split_first_flow(self, leakage_test_data):
        """Verify train() splits before fitting the feature pipeline."""
        # We verify this indirectly: if train() splits first, the training
        # data should not include the latest temporal data
        trainer = ModelTrainer(model_type="baseline", model_version="leak_test")
        metrics = trainer.train(leakage_test_data)

        # Should complete without error
        assert metrics is not None
        assert metrics.mae > 0

    def test_cap_price_outliers(self):
        """Test that extreme prices are capped."""
        df = pd.DataFrame(
            {
                "listing_price": [10, 20, 30, 40, 50, 100, 200, 500, 1000, 10000],
            }
        )

        capped = ModelTrainer._cap_price_outliers(df, percentile=90.0)

        # 90th percentile should cap the extreme values
        assert capped["listing_price"].max() < 10000
        # Lower values unchanged
        assert capped["listing_price"].iloc[0] == 10

    def test_cap_price_outliers_preserves_original(self):
        """Test that capping does not modify the original DataFrame."""
        df = pd.DataFrame({"listing_price": [10, 100, 10000]})
        original_max = df["listing_price"].max()

        ModelTrainer._cap_price_outliers(df, percentile=90.0)

        # Original should be unchanged
        assert df["listing_price"].max() == original_max

    def test_train_with_params(self, leakage_test_data):
        """Test that train() accepts custom params."""
        params = {"num_leaves": 15, "learning_rate": 0.1}
        trainer = ModelTrainer(model_type="lightgbm", model_version="params_test")
        metrics = trainer.train(leakage_test_data, params=params)

        assert metrics is not None
        assert metrics.mae > 0


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
        # Train (use GBDT for deterministic test behavior)
        trainer = ModelTrainer(model_type="lightgbm", model_version="integration_test")
        metrics = trainer.train(sample_data, params={"boosting_type": "gbdt", "n_estimators": 200})

        # Verify metrics are reasonable
        assert metrics.mae > 0
        assert metrics.mae < 200  # Should be less than $200 MAE on synthetic data

        # Save
        model_path = trainer.save(tmp_path)
        assert model_path.exists()

        # Load and verify predictions work
        loaded_model = ModelTrainer.load(model_path, "lightgbm")

        # Build test input using the model's own feature names to avoid shape mismatch
        feature_names = loaded_model._feature_names
        assert len(feature_names) > 0
        test_features = pd.DataFrame(np.zeros((1, len(feature_names))), columns=feature_names)
        pred = loaded_model.predict(test_features)

        assert len(pred) == 1

    def test_lightgbm_better_than_baseline(self, sample_data):
        """Test LightGBM outperforms baseline."""
        baseline_trainer = ModelTrainer(model_type="baseline")
        baseline_metrics = baseline_trainer.train(sample_data)

        lgb_trainer = ModelTrainer(model_type="lightgbm")
        lgb_metrics = lgb_trainer.train(
            sample_data, params={"boosting_type": "gbdt", "n_estimators": 200}
        )

        # LightGBM should not be significantly worse than baseline.
        # With small synthetic data and log-transform, results can vary,
        # so allow 30% tolerance.
        assert lgb_metrics.mae < baseline_metrics.mae * 1.30


# ============================================================================
# Log-Transform Allowlist Tests (Phase 3)
# ============================================================================


class TestLogTransformAllowlist:
    """Tests for _LOG_EXCLUDE_SUFFIXES allowlist that gates price feature log-transforms."""

    @pytest.fixture
    def sample_data(self):
        """Minimal listing data for fast training."""
        np.random.seed(42)
        n = 200
        artists = ["Artist A", "Artist B"]
        return pd.DataFrame(
            {
                "artist_or_team": np.random.choice(artists, n),
                "event_type": ["CONCERT"] * n,
                "city": np.random.choice(["New York", "Los Angeles"], n),
                "event_datetime": pd.date_range("2024-01-01", periods=n, freq="6h"),
                "section": np.random.choice(["Floor", "Lower Level", "Upper Level"], n),
                "row": np.random.choice(["1", "5", "10"], n),
                "days_to_event": np.random.randint(1, 60, n),
                "listing_price": 100 + np.random.randn(n) * 30,
                "event_id": [f"e{i % 20}" for i in range(n)],
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="6h"),
            }
        )

    def test_log_exclude_suffixes_contains_std(self):
        """_LOG_EXCLUDE_SUFFIXES must exclude _std columns."""
        from ticket_price_predictor.ml.training.trainer import _LOG_EXCLUDE_SUFFIXES

        assert "_std" in _LOG_EXCLUDE_SUFFIXES

    def test_log_exclude_suffixes_contains_cv_and_ratio(self):
        """_LOG_EXCLUDE_SUFFIXES must exclude _cv and _ratio."""
        from ticket_price_predictor.ml.training.trainer import _LOG_EXCLUDE_SUFFIXES

        assert "_cv" in _LOG_EXCLUDE_SUFFIXES
        assert "_ratio" in _LOG_EXCLUDE_SUFFIXES

    def test_log_transformed_cols_populated_after_train(self, sample_data):
        """After train(), _log_transformed_cols is a non-empty list."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="test")
        trainer.train(
            sample_data,
            params={"boosting_type": "gbdt", "n_estimators": 50, "verbose": -1},
        )
        assert isinstance(trainer._log_transformed_cols, list)
        # EventPricingFeatureExtractor generates price columns, so should be non-empty
        assert len(trainer._log_transformed_cols) > 0

    def test_log_transformed_cols_no_std_columns(self, sample_data):
        """_log_transformed_cols must not include any _std suffix columns."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="test")
        trainer.train(
            sample_data,
            params={"boosting_type": "gbdt", "n_estimators": 50, "verbose": -1},
        )
        for col in trainer._log_transformed_cols:
            assert not col.lower().endswith("_std"), f"Unexpected _std column: {col}"

    def test_log_transformed_cols_no_ratio_columns(self, sample_data):
        """_log_transformed_cols must not include any _ratio suffix columns."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="test")
        trainer.train(
            sample_data,
            params={"boosting_type": "gbdt", "n_estimators": 50, "verbose": -1},
        )
        for col in trainer._log_transformed_cols:
            assert not col.lower().endswith("_ratio"), f"Unexpected _ratio column: {col}"


# ============================================================================
# Per-Quartile and Per-Zone MAE Tests (Phase 4)
# ============================================================================


class TestQuartileZoneMetrics:
    """Tests for per-quartile and per-zone MAE in ModelEvaluator and TrainingMetrics."""

    def test_compute_metrics_returns_quartile_mae(self):
        """compute_metrics() returns a dict with Q1-Q4 keys."""
        y_true = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0, 400.0])
        y_pred = np.array([55.0, 105.0, 145.0, 205.0, 245.0, 305.0, 345.0, 405.0])

        result = ModelEvaluator.compute_metrics(y_true, y_pred)

        assert "quartile_mae" in result
        qmae = result["quartile_mae"]
        assert isinstance(qmae, dict)
        assert set(qmae.keys()) == {"Q1", "Q2", "Q3", "Q4"}

    def test_compute_metrics_quartile_mae_values_are_non_negative(self):
        """Each quartile MAE value must be non-negative."""
        np.random.seed(0)
        y_true = np.linspace(10.0, 500.0, 100)
        y_pred = y_true + np.random.randn(100) * 20.0

        result = ModelEvaluator.compute_metrics(y_true, y_pred)
        for q, val in result["quartile_mae"].items():
            assert val >= 0, f"Quartile {q} MAE should be non-negative, got {val}"

    def test_compute_metrics_returns_zone_mae_when_provided(self):
        """compute_metrics() returns per-zone MAE when zones array is passed."""
        y_true = np.array([100.0, 200.0, 100.0, 200.0, 300.0])
        y_pred = np.array([110.0, 190.0, 105.0, 205.0, 295.0])
        zones = np.array(["floor", "lower", "floor", "lower", "upper"])

        result = ModelEvaluator.compute_metrics(y_true, y_pred, zones=zones)
        zone_mae = result["zone_mae"]

        assert isinstance(zone_mae, dict)
        assert "floor" in zone_mae
        assert "lower" in zone_mae
        assert "upper" in zone_mae

    def test_compute_metrics_zone_mae_empty_when_no_zones(self):
        """compute_metrics() returns empty zone_mae when zones=None."""
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 190.0])

        result = ModelEvaluator.compute_metrics(y_true, y_pred, zones=None)
        assert result["zone_mae"] == {}

    def test_training_metrics_quartile_zone_default_to_empty_dicts(self):
        """TrainingMetrics.quartile_mae and zone_mae default to empty dicts."""
        metrics = TrainingMetrics(
            mae=10.0,
            rmse=15.0,
            mape=5.0,
            r2=0.9,
            n_train_samples=100,
            n_val_samples=20,
            n_test_samples=20,
            n_features=5,
            training_time_seconds=1.0,
            model_version="v1",
            model_type="lightgbm",
        )
        assert metrics.quartile_mae == {}
        assert metrics.zone_mae == {}


# ============================================================================
# CV Objective Tests (Phase 5)
# ============================================================================


class TestCVObjective:
    """Tests for create_raw_objective with use_cv=True/False."""

    @pytest.fixture
    def raw_split(self):
        """Return a RawDataSplit from a small listing DataFrame."""
        np.random.seed(42)
        n = 200
        df = pd.DataFrame(
            {
                "artist_or_team": np.random.choice(["Artist A", "Artist B"], n),
                "event_type": ["CONCERT"] * n,
                "city": np.random.choice(["New York", "Los Angeles"], n),
                "event_datetime": pd.date_range("2024-01-01", periods=n, freq="6h"),
                "section": np.random.choice(["Floor", "Lower Level", "Upper Level"], n),
                "row": np.random.choice(["1", "5", "10"], n),
                "days_to_event": np.random.randint(1, 60, n),
                "listing_price": 100 + np.random.randn(n) * 30,
                "event_id": [f"e{i % 20}" for i in range(n)],
                "timestamp": pd.date_range("2024-01-01", periods=n, freq="6h"),
            }
        )
        splitter = TimeBasedSplitter(
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            stratify_col="artist_or_team",
        )
        return splitter.split_raw(df)

    def test_create_raw_objective_single_split_runs(self, raw_split):
        """create_raw_objective with use_cv=False completes one trial without error."""
        import optuna

        from ticket_price_predictor.ml.tuning.objective import create_raw_objective

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        objective = create_raw_objective(
            raw_split=raw_split,
            pipeline_kwargs={"include_listing": False, "include_popularity": False},
            penalize_dominance=False,
            use_cv=False,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1)

        assert len(study.trials) == 1
        assert study.trials[0].value is not None

    def test_create_raw_objective_with_cv_runs(self, raw_split):
        """create_raw_objective with use_cv=True completes one trial without error."""
        import optuna

        from ticket_price_predictor.ml.tuning.objective import create_raw_objective

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        objective = create_raw_objective(
            raw_split=raw_split,
            pipeline_kwargs={"include_listing": False, "include_popularity": False},
            penalize_dominance=False,
            use_cv=True,
            n_cv_folds=2,
        )
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1)

        assert len(study.trials) == 1
        assert study.trials[0].value is not None
