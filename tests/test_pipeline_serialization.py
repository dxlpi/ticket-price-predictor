"""Tests for FeaturePipeline serialization and trainer companion file save/load."""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.features.popularity import PopularityFeatureExtractor
from ticket_price_predictor.ml.training.trainer import ModelTrainer


@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Minimal listing DataFrame sufficient for FeaturePipeline.fit/transform."""
    np.random.seed(0)
    n = 60
    return pd.DataFrame(
        {
            "artist_or_team": np.random.choice(["Artist A", "Artist B"], n),
            "event_type": ["CONCERT"] * n,
            "city": np.random.choice(["New York", "Los Angeles"], n),
            "event_datetime": pd.date_range("2024-01-01", periods=n, freq="6h"),
            "section": np.random.choice(["Floor", "Lower Level", "Upper Level"], n),
            "row": np.random.choice(["1", "5", "10"], n),
            "days_to_event": np.random.randint(1, 60, n),
            "listing_price": 100 + np.random.randn(n) * 30,
            "event_id": [f"e{i % 10}" for i in range(n)],
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="6h"),
        }
    )


# ============================================================================
# Pipeline Round-Trip Tests
# ============================================================================


class TestPipelineRoundTrip:
    """Tests for FeaturePipeline save/load round-trip."""

    def test_pipeline_save_load_preserves_columns(self, sample_df, tmp_path):
        """Loaded pipeline produces identical columns after save/load."""
        pipeline = FeaturePipeline(include_momentum=False, include_popularity=False)
        pipeline.fit(sample_df)
        X_before = pipeline.transform(sample_df)

        path = tmp_path / "pipeline.joblib"
        pipeline.save(path)

        loaded = FeaturePipeline.load(path)
        X_after = loaded.transform(sample_df)

        assert list(X_before.columns) == list(X_after.columns)
        assert X_before.shape == X_after.shape

    def test_pipeline_fitted_flag_preserved(self, sample_df, tmp_path):
        """_fitted flag is True after loading a fitted pipeline."""
        pipeline = FeaturePipeline(include_momentum=False, include_popularity=False)
        pipeline.fit(sample_df)

        path = tmp_path / "pipeline.joblib"
        pipeline.save(path)

        loaded = FeaturePipeline.load(path)
        assert loaded._fitted is True

    def test_pipeline_load_type_check(self, tmp_path):
        """Loading a non-FeaturePipeline file raises TypeError."""
        import joblib

        path = tmp_path / "wrong.joblib"
        joblib.dump({"not": "a pipeline"}, path)

        with pytest.raises(TypeError, match="Expected FeaturePipeline"):
            FeaturePipeline.load(path)


# ============================================================================
# PopularityFeatureExtractor Pickle Tests
# ============================================================================


class TestPopularityExtractorPickle:
    """Tests for PopularityFeatureExtractor __getstate__/__setstate__."""

    def test_getstate_nullifies_service(self):
        """__getstate__ sets _service to None to avoid pickling YTMusic."""
        extractor = PopularityFeatureExtractor(popularity_service=None)
        state = extractor.__getstate__()
        assert state["_service"] is None

    def test_pickle_round_trip_preserves_artist_cache(self, sample_df):
        """Pickle round-trip preserves _artist_cache contents."""
        extractor = PopularityFeatureExtractor(popularity_service=None)
        extractor.fit(sample_df)

        # Manually populate cache to verify it survives round-trip
        extractor._artist_cache["test_artist"] = {"score": 0.5}  # type: ignore[assignment]

        data = pickle.dumps(extractor)
        loaded: PopularityFeatureExtractor = pickle.loads(data)

        assert "test_artist" in loaded._artist_cache
        assert loaded._artist_cache["test_artist"] == {"score": 0.5}  # type: ignore[comparison-overlap]

    def test_setstate_restores_dict(self):
        """__setstate__ restores state from dict."""
        extractor = PopularityFeatureExtractor(popularity_service=None)
        extractor.__setstate__({"_service": None, "_artist_cache": {"a": 1}, "_warned": False})
        assert extractor._artist_cache == {"a": 1}  # type: ignore[comparison-overlap]
        assert extractor._service is None


# ============================================================================
# Trainer Companion File Tests
# ============================================================================


class TestTrainerSaveCompanionFiles:
    """Tests that ModelTrainer.save() produces pipeline + meta companion files."""

    @pytest.fixture
    def trained_trainer(self, sample_df: pd.DataFrame) -> ModelTrainer:
        """Return a trainer trained on sample_df with GBDT (fast)."""
        trainer = ModelTrainer(model_type="lightgbm", model_version="test")
        trainer.train(
            sample_df,
            params={"boosting_type": "gbdt", "n_estimators": 50, "verbose": -1},
        )
        return trainer

    def test_save_creates_pipeline_file(self, trained_trainer: ModelTrainer, tmp_path: Path):
        """trainer.save() creates a _pipeline.joblib companion file."""
        trained_trainer.save(tmp_path)
        pipeline_path = tmp_path / "lightgbm_test_pipeline.joblib"
        assert pipeline_path.exists(), f"Expected {pipeline_path} to exist"

    def test_save_creates_meta_file(self, trained_trainer: ModelTrainer, tmp_path: Path):
        """trainer.save() creates a _meta.json with log_transformed_cols."""
        trained_trainer.save(tmp_path)
        meta_path = tmp_path / "lightgbm_test_meta.json"
        assert meta_path.exists(), f"Expected {meta_path} to exist"

        meta = json.loads(meta_path.read_text())
        assert "log_transformed_cols" in meta
        assert isinstance(meta["log_transformed_cols"], list)

    def test_meta_contains_only_price_like_columns(
        self, trained_trainer: ModelTrainer, tmp_path: Path
    ):
        """meta.json log_transformed_cols contains only price/avg/median columns."""
        trained_trainer.save(tmp_path)
        meta_path = tmp_path / "lightgbm_test_meta.json"
        meta = json.loads(meta_path.read_text())
        cols = meta["log_transformed_cols"]
        for col in cols:
            assert any(kw in col.lower() for kw in ("price", "avg", "median")), (
                f"Unexpected column in log_transformed_cols: {col}"
            )

    def test_loaded_pipeline_is_feature_pipeline(
        self, trained_trainer: ModelTrainer, tmp_path: Path
    ):
        """Pipeline file contains a FeaturePipeline instance."""
        trained_trainer.save(tmp_path)
        pipeline_path = tmp_path / "lightgbm_test_pipeline.joblib"
        loaded = FeaturePipeline.load(pipeline_path)
        assert isinstance(loaded, FeaturePipeline)
        assert loaded._fitted is True
