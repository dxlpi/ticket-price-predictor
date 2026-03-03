"""Model training utilities."""

import json
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
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
        params: dict[str, Any] | None = None,
        popularity_service: Any | None = None,
        pipeline_kwargs: dict[str, Any] | None = None,
        sample_weight_strategy: Literal["none", "inverse_artist_freq"] = "none",
    ) -> TrainingMetrics:
        """Train model on data.

        Splits raw data BEFORE feature extraction to prevent data leakage.
        The feature pipeline is fit only on training data, then used to
        transform validation and test sets independently.

        Args:
            df: DataFrame with raw listing data
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            preprocess: Enable preprocessing pipeline before training (default: False)
            params: Optional hyperparameters for the model
            popularity_service: PopularityService instance for API-based features
            pipeline_kwargs: Extra keyword arguments forwarded to FeaturePipeline
            sample_weight_strategy: How to weight training samples.
                "none" (default): uniform weights.
                "inverse_artist_freq": weight = 1/count_per_artist, normalized
                so that weights sum to n_train.

        Returns:
            Training metrics
        """
        print(f"Training {self._model_type} model...")
        print(f"Data shape: {df.shape}")

        # Optional preprocessing
        if preprocess:
            print("Running preprocessing pipeline...")
            df = self._preprocess_data(df)

        # Filter obviously invalid prices and cap outliers
        df = self._filter_invalid_prices(df)
        df = self._cap_price_outliers(df)

        # Normalize city names for consistent grouping
        if "city" in df.columns:
            from ticket_price_predictor.ml.features.geo_mapping import _normalize_city

            df = df.copy()
            df["city"] = df["city"].apply(_normalize_city)

        # Normalize artist name aliases (e.g. "BTS - Bangtan Boys" → "BTS")
        if "artist_or_team" in df.columns:
            df = self._normalize_artist_names(df)

        # Split raw data FIRST (before feature extraction)
        print("Splitting raw data...")
        splitter = TimeBasedSplitter(
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
            stratify_col="artist_or_team",
        )
        raw_split = splitter.split_raw(df)

        print(
            f"Raw split — Train: {raw_split.n_train}, Val: {raw_split.n_val}, Test: {raw_split.n_test}"
        )

        # Diagnostic: report artist coverage across splits
        if "artist_or_team" in df.columns:
            train_artists = set(raw_split.train_df["artist_or_team"].unique())
            val_artists = set(raw_split.val_df["artist_or_team"].unique())
            test_artists = set(raw_split.test_df["artist_or_team"].unique())
            val_unknown = val_artists - train_artists
            test_unknown = test_artists - train_artists
            print(
                f"Artists — Train: {len(train_artists)}, Val: {len(val_artists)}, Test: {len(test_artists)}"
            )
            if val_unknown:
                print(f"  WARNING: {len(val_unknown)} val artists not in train: {val_unknown}")
            if test_unknown:
                print(f"  WARNING: {len(test_unknown)} test artists not in train: {test_unknown}")

        # Fit feature pipeline on training data ONLY
        print("Extracting features (fit on train only)...")
        _pipeline_kwargs: dict[str, Any] = {
            "include_momentum": True,
            "popularity_service": popularity_service,
        }
        if pipeline_kwargs:
            _pipeline_kwargs.update(pipeline_kwargs)
        self._feature_pipeline = FeaturePipeline(**_pipeline_kwargs)
        self._feature_pipeline.fit(raw_split.train_df)

        # Transform each split independently
        X_train = self._feature_pipeline.transform(raw_split.train_df)
        X_val = self._feature_pipeline.transform(raw_split.val_df)
        X_test = self._feature_pipeline.transform(raw_split.test_df)

        # Remove zero-variance features (constants add noise, waste splits)
        zero_var_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
        if zero_var_cols:
            print(f"Removing {len(zero_var_cols)} zero-variance features: {zero_var_cols}")
            X_train = X_train.drop(columns=zero_var_cols)
            X_val = X_val.drop(columns=zero_var_cols)
            X_test = X_test.drop(columns=zero_var_cols)

        # Log-transform price-based features to align with log-transformed target
        price_cols = [
            c
            for c in X_train.columns
            if "price" in c.lower() or "avg" in c.lower() or "median" in c.lower()
        ]
        if price_cols:
            print(f"Log-transforming {len(price_cols)} price features")
            for c in price_cols:
                X_train[c] = np.log1p(X_train[c].clip(lower=0))
                X_val[c] = np.log1p(X_val[c].clip(lower=0))
                X_test[c] = np.log1p(X_test[c].clip(lower=0))

        # Log-transform target for better handling of skewed price distribution
        y_train = np.log1p(raw_split.train_df[self._target_col].values)
        y_val = np.log1p(raw_split.val_df[self._target_col].values)
        y_test_raw = raw_split.test_df[self._target_col].values  # keep raw for evaluation

        print(f"Feature shape: {X_train.shape}")
        print(f"Features: {list(X_train.columns)}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Compute sample weights for training set
        sample_weight: np.ndarray[Any, Any] | None = None
        if sample_weight_strategy == "inverse_artist_freq":
            if "artist_or_team" in raw_split.train_df.columns:
                artist_counts = raw_split.train_df["artist_or_team"].value_counts()
                weights = raw_split.train_df["artist_or_team"].map(lambda a: 1.0 / artist_counts[a])
                # Normalize so weights sum to n_train (keeps effective scale)
                weights = weights / weights.sum() * len(weights)
                sample_weight = weights.values
                print(
                    f"Sample weights: inverse_artist_freq "
                    f"(min={sample_weight.min():.3f}, max={sample_weight.max():.3f})"
                )
            else:
                print(
                    "WARNING: sample_weight_strategy='inverse_artist_freq' requested "
                    "but 'artist_or_team' column not found — using uniform weights"
                )

        # Train model
        print("Training model...")
        start_time = time.time()

        self._model = self._create_model(params=params)
        try:
            self._model.fit(  # type: ignore[call-arg]
                X_train,
                pd.Series(y_train),
                X_val,
                pd.Series(y_val),
                sample_weight=sample_weight,
            )
        except TypeError:
            # Model does not support sample_weight (e.g. BaselineModel) — fit without it
            if sample_weight is not None:
                print("WARNING: model does not support sample_weight — fitting without weights")
            self._model.fit(
                X_train,
                pd.Series(y_train),
                X_val,
                pd.Series(y_val),
            )

        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.1f}s")

        # Convergence diagnostic: best_iteration=0 is expected for DART (uses all trees),
        # but for GBDT it suggests early stopping fired at round 0 (hyperparameter issue).
        best_iter = getattr(self._model, "best_iteration", None)
        if best_iter is not None and best_iter == 0:
            boosting = (params or {}).get(
                "boosting_type", LightGBMModel.DEFAULT_PARAMS.get("boosting_type", "dart")
            )
            if boosting != "dart":
                print(
                    f"WARNING: best_iteration=0 with boosting_type={boosting!r}. "
                    "Early stopping fired at round 0 — check learning_rate and early_stopping_rounds."
                )

        # Evaluate: inverse-transform predictions back to raw price scale
        print("Evaluating model...")
        self._metrics = ModelEvaluator.evaluate_model(
            self._model,
            X_test,
            y_test_raw,
            n_train=len(X_train),
            n_val=len(X_val),
            training_time=training_time,
            model_version=self._model_version,
            log_target=True,
        )

        ModelEvaluator.print_metrics(self._metrics)

        return self._metrics

    @staticmethod
    def _normalize_artist_names(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize artist name aliases to canonical forms.

        Merges variant names (e.g. 'BTS - Bangtan Boys' → 'BTS') so that
        artist-level statistics are computed on the full sample rather than
        being split across aliases.
        """
        aliases: dict[str, str] = {
            "BTS - Bangtan Boys": "BTS",
            "Rush - Rock Band": "Rush",
        }
        mapped = df["artist_or_team"].map(aliases)
        changed = mapped.notna()
        if changed.any():
            df = df.copy()
            df.loc[changed, "artist_or_team"] = mapped[changed]
            n = changed.sum()
            print(f"Normalized {n} artist aliases ({len(aliases)} rules)")
        return df

    @staticmethod
    def _deduplicate_listings(df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate listings, keeping the latest timestamp per unique listing.

        Deduplication is based on the combination of event, section, row, and price.
        When a timestamp column is available, the most recent entry is kept.

        Args:
            df: Input DataFrame

        Returns:
            DataFrame with duplicates removed
        """
        dedup_cols = ["event_id", "section", "row", "listing_price"]
        available_cols = [c for c in dedup_cols if c in df.columns]

        if len(available_cols) < 2:
            return df

        before = len(df)
        if "timestamp" in df.columns:
            df = df.sort_values("timestamp", ascending=False)
        df = df.drop_duplicates(subset=available_cols, keep="first")
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} duplicate listings ({removed / before * 100:.1f}%)")
        return df

    @staticmethod
    def _filter_invalid_prices(
        df: pd.DataFrame,
        target_col: str = "listing_price",
        min_price: float = 10.0,
    ) -> pd.DataFrame:
        """Remove rows with obviously invalid prices.

        Args:
            df: Input DataFrame
            target_col: Price column to filter
            min_price: Minimum valid price

        Returns:
            DataFrame with invalid prices removed
        """
        if target_col not in df.columns:
            return df
        before = len(df)
        df = df[df[target_col] >= min_price].copy()
        removed = before - len(df)
        if removed > 0:
            print(f"Removed {removed} rows with price < ${min_price:.0f}")
        return df

    @staticmethod
    def _cap_price_outliers(
        df: pd.DataFrame,
        target_col: str = "listing_price",
        percentile: float = 95.0,
    ) -> pd.DataFrame:
        """Cap extreme price outliers before splitting.

        Uses a single global percentile cap. This keeps training and test
        distributions aligned and avoids the tail-error inflation caused by
        per-zone caps that preserve very high VIP prices.

        Args:
            df: Input DataFrame
            target_col: Price column to cap
            percentile: Upper percentile to cap at

        Returns:
            DataFrame with capped prices
        """
        if target_col not in df.columns:
            return df
        df = df.copy()
        df[target_col] = df[target_col].astype(float)

        cap = float(df[target_col].quantile(percentile / 100.0))
        df[target_col] = df[target_col].clip(upper=cap)
        print(f"Price outlier cap at ${cap:.2f} (p{percentile:.0f})")
        return df

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
