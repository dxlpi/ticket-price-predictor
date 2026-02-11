"""Model training utilities."""

import json
import time
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.baseline import BaselineModel
from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel, QuantileLightGBMModel
from ticket_price_predictor.ml.schemas import TrainingMetrics
from ticket_price_predictor.ml.training.evaluator import ModelEvaluator
from ticket_price_predictor.ml.training.splitter import DataSplit, TimeBasedSplitter

ModelType = Literal["baseline", "lightgbm", "quantile"]


class ModelTrainer:
    """Train and evaluate models."""

    def __init__(
        self,
        model_type: ModelType = "lightgbm",
        target_col: str = "listing_price",
        model_version: str = "v1",
    ) -> None:
        """Initialize trainer.

        Args:
            model_type: Type of model to train
            target_col: Target column name
            model_version: Version string for the model
        """
        self._model_type = model_type
        self._target_col = target_col
        self._model_version = model_version
        self._model: PriceModel | None = None
        self._feature_pipeline: FeaturePipeline | None = None
        self._metrics: TrainingMetrics | None = None

    @property
    def model(self) -> PriceModel | None:
        """Return trained model."""
        return self._model

    @property
    def metrics(self) -> TrainingMetrics | None:
        """Return training metrics."""
        return self._metrics

    def _create_model(self, params: dict[str, Any] | None = None) -> PriceModel:
        """Create model instance based on type.

        Args:
            params: Optional hyperparameters for the model
        """
        if self._model_type == "baseline":
            return BaselineModel()
        elif self._model_type == "lightgbm":
            return LightGBMModel(params=params) if params else LightGBMModel()
        elif self._model_type == "quantile":
            return QuantileLightGBMModel(params=params) if params else QuantileLightGBMModel()
        else:
            raise ValueError(f"Unknown model type: {self._model_type}")

    def train(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        preprocess: bool = False,
    ) -> TrainingMetrics:
        """Train model on data.

        Args:
            df: DataFrame with raw listing data
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            preprocess: Enable preprocessing pipeline before training (default: False)

        Returns:
            Training metrics
        """
        print(f"Training {self._model_type} model...")
        print(f"Data shape: {df.shape}")

        # Optional preprocessing
        if preprocess:
            print("Running preprocessing pipeline...")
            df = self._preprocess_data(df)

        # Extract features (momentum features enabled for better predictions)
        print("Extracting features...")
        self._feature_pipeline = FeaturePipeline(include_momentum=True)
        X = self._feature_pipeline.fit_transform(df)
        y = df[self._target_col]

        print(f"Feature shape: {X.shape}")
        print(f"Features: {self._feature_pipeline.feature_names}")

        # Split data
        print("Splitting data...")
        splitter = TimeBasedSplitter(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )
        split = splitter.split(X, y, raw_df=df)

        print(f"Train: {split.n_train}, Val: {split.n_val}, Test: {split.n_test}")

        # Train model
        print("Training model...")
        start_time = time.time()

        self._model = self._create_model()
        self._model.fit(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
        )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")

        # Evaluate
        print("Evaluating model...")
        self._metrics = ModelEvaluator.evaluate_model(
            self._model,
            split.X_test,
            split.y_test,
            n_train=split.n_train,
            n_val=split.n_val,
            training_time=training_time,
            model_version=self._model_version,
        )

        ModelEvaluator.print_metrics(self._metrics)

        return self._metrics

    def train_with_split(
        self,
        split: DataSplit,
    ) -> TrainingMetrics:
        """Train model with pre-computed split.

        Args:
            split: DataSplit object

        Returns:
            Training metrics
        """
        print(f"Training {self._model_type} model...")
        print(f"Train: {split.n_train}, Val: {split.n_val}, Test: {split.n_test}")

        start_time = time.time()

        self._model = self._create_model()
        self._model.fit(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
        )

        training_time = time.time() - start_time

        self._metrics = ModelEvaluator.evaluate_model(
            self._model,
            split.X_test,
            split.y_test,
            n_train=split.n_train,
            n_val=split.n_val,
            training_time=training_time,
            model_version=self._model_version,
        )

        ModelEvaluator.print_metrics(self._metrics)

        return self._metrics

    def train_with_params(
        self,
        split: DataSplit,
        params: dict[str, Any],
    ) -> TrainingMetrics:
        """Train model with custom hyperparameters.

        Args:
            split: DataSplit object
            params: Hyperparameter dictionary

        Returns:
            Training metrics
        """
        print(f"Training {self._model_type} model with custom params...")
        print(f"Train: {split.n_train}, Val: {split.n_val}, Test: {split.n_test}")

        start_time = time.time()

        self._model = self._create_model(params=params)
        self._model.fit(
            split.X_train,
            split.y_train,
            split.X_val,
            split.y_val,
        )

        training_time = time.time() - start_time

        self._metrics = ModelEvaluator.evaluate_model(
            self._model,
            split.X_test,
            split.y_test,
            n_train=split.n_train,
            n_val=split.n_val,
            training_time=training_time,
            model_version=self._model_version,
        )

        ModelEvaluator.print_metrics(self._metrics)

        return self._metrics

    def save(self, output_dir: Path) -> Path:
        """Save trained model and metrics.

        Args:
            output_dir: Directory to save model

        Returns:
            Path to saved model
        """
        if self._model is None:
            raise RuntimeError("No model to save. Train first.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        model_path = output_dir / f"{self._model_type}_{self._model_version}.joblib"
        self._model.save(model_path)
        print(f"Model saved to: {model_path}")

        # Save metrics
        if self._metrics:
            metrics_path = output_dir / f"{self._model_type}_{self._model_version}_metrics.json"
            with open(metrics_path, "w") as f:
                json.dump(self._metrics.model_dump(), f, indent=2, default=str)
            print(f"Metrics saved to: {metrics_path}")

        return model_path

    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply preprocessing pipeline to training data.

        Args:
            df: Raw DataFrame

        Returns:
            Preprocessed DataFrame
        """
        from ticket_price_predictor.preprocessing import (
            PipelineBuilder,
            PreprocessingConfig,
            QualityReporter,
        )

        config = PreprocessingConfig()
        pipeline = PipelineBuilder.build_listings_pipeline(config=config)

        result = pipeline.process(df)

        # Report quality metrics
        reporter = QualityReporter(config)
        metrics = reporter.extract_metrics(result)
        print("\nPreprocessing Quality Report:")
        print(f"  Input rows:  {metrics.input_rows:,}")
        print(f"  Output rows: {metrics.output_rows:,}")
        print(f"  Drop rate:   {metrics.drop_rate:.2f}%")
        print(f"  Alert level: {reporter.check_thresholds(metrics).value.upper()}")
        print()

        if result.issues:
            print(f"Preprocessing issues found: {len(result.issues)}")
            for issue in result.issues[:5]:  # Show first 5
                print(f"  - {issue}")
            if len(result.issues) > 5:
                print(f"  ... and {len(result.issues) - 5} more")
            print()

        return result.data

    @classmethod
    def load(cls, model_path: Path, model_type: ModelType) -> PriceModel:
        """Load a trained model.

        Args:
            model_path: Path to saved model
            model_type: Type of model

        Returns:
            Loaded model
        """
        if model_type == "baseline":
            return BaselineModel.load(model_path)
        elif model_type == "lightgbm":
            return LightGBMModel.load(model_path)
        elif model_type == "quantile":
            return QuantileLightGBMModel.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
