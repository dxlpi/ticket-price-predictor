"""Model training utilities."""

import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.models.baseline import BaselineModel
from ticket_price_predictor.ml.models.catboost_model import CatBoostModel
from ticket_price_predictor.ml.models.lightgbm_model import LightGBMModel, QuantileLightGBMModel
from ticket_price_predictor.ml.models.residual import ResidualModel
from ticket_price_predictor.ml.models.stacking import StackingEnsemble
from ticket_price_predictor.ml.models.stacking_v2 import StackingEnsembleV2
from ticket_price_predictor.ml.models.xgboost_model import XGBoostModel
from ticket_price_predictor.ml.schemas import TrainingMetrics
from ticket_price_predictor.ml.training.evaluator import ModelEvaluator
from ticket_price_predictor.ml.training.splitter import DataSplit, TimeBasedSplitter
from ticket_price_predictor.ml.training.target_transforms import (
    TargetTransform,
    create_target_transform,
)

ModelType = Literal[
    "baseline",
    "lightgbm",
    "quantile",
    "xgboost",
    "catboost",
    "stacking",
    "stacking_v2",
    "residual",
]

OutlierStrategy = Literal["global_p95", "zone_winsorize", "none"]

SampleWeightStrategy = Literal[
    "none", "inverse_artist_freq", "sqrt_price", "log_price", "inverse_price_quartile"
]

# Suffixes that indicate scale-independent derived statistics.
# Columns with these suffixes are excluded from log-transformation even when
# their name contains "price", "avg", or "median", because log-transforming
# std/cv/ratio/count values is statistically inappropriate.
_LOG_EXCLUDE_SUFFIXES = (
    "_std",
    "_cv",
    "_ratio",
    "_count",
    "_change",
    "_rate",
    "_skewness",
    "_deviation",
)


def _join_snapshot_features(
    listings_df: pd.DataFrame,
    snapshot_df: pd.DataFrame,
    zone_mapper: Any,
) -> pd.DataFrame:
    """Join temporal snapshot features to listings.

    MUST be called AFTER split_raw(), once per split independently, to
    prevent cross-split information leakage (snapshots contain price_avg
    derived from listing prices).

    Uses pd.merge_asof() to find the nearest prior snapshot per
    (event_id, seat_zone) for each listing row. Temporal delta features
    are computed from the earliest and matched (latest prior) snapshot.

    Args:
        listings_df: One split's raw listing DataFrame
        snapshot_df: Raw snapshot DataFrame from DataLoader.load_snapshots()
        zone_mapper: Fitted SeatZoneMapper instance

    Returns:
        listings_df enriched with _snap_* columns (same row count)
    """
    result = listings_df.copy()

    # Early return when no snapshot data is available
    if snapshot_df.empty:
        for col in [
            "_snap_inventory_change_rate",
            "_snap_zone_price_trend",
            "_snap_count",
            "_snap_price_range",
        ]:
            result[col] = 0.0
        return result

    result["_pos"] = range(len(result))

    # Map raw section names → normalized zone strings
    result["_zone"] = result["section"].apply(
        lambda s: zone_mapper.normalize_zone_name(str(s)).value
    )
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)

    # Prepare snapshots (seat_zone is already a string; SeatZone is StrEnum)
    snap = snapshot_df.copy()
    snap["_zone"] = snap["seat_zone"]
    snap["timestamp"] = pd.to_datetime(snap["timestamp"], utc=True)
    snap = snap.sort_values("timestamp")

    # Precompute per-(event_id, _zone): earliest snapshot and total count
    snap_agg = (
        snap.groupby(["event_id", "_zone"])
        .agg(
            _earliest_price_avg=("price_avg", "first"),
            _earliest_inv=("inventory_remaining", "first"),
            _earliest_ts=("timestamp", "first"),
            _snap_count_raw=("timestamp", "count"),
        )
        .reset_index()
    )

    # Rename snapshot columns before merge_asof to avoid collision with listing columns
    snap_for_merge = snap[
        [
            "event_id",
            "_zone",
            "timestamp",
            "price_avg",
            "price_min",
            "price_max",
            "inventory_remaining",
        ]
    ].rename(
        columns={
            "timestamp": "_snap_ts",
            "price_avg": "_snap_price_avg",
            "price_min": "_snap_price_min",
            "price_max": "_snap_price_max",
            "inventory_remaining": "_snap_inv",
        }
    )

    # merge_asof: for each listing, find latest snapshot with _snap_ts <= listing.timestamp
    result_sorted = result.sort_values("timestamp")
    merged = pd.merge_asof(
        result_sorted,
        snap_for_merge,
        left_on="timestamp",
        right_on="_snap_ts",
        by=["event_id", "_zone"],
        direction="backward",
    )

    # Join per-group aggregates
    merged = merged.merge(snap_agg, on=["event_id", "_zone"], how="left")

    # --- Compute temporal delta features ---

    # Hours between earliest and matched snapshot
    delta_hours = (
        (merged["_snap_ts"] - merged["_earliest_ts"]).dt.total_seconds().div(3600).fillna(0.0)
    )

    # Inventory change rate: tickets/hour (negative = selling)
    inv_delta = merged["_snap_inv"].fillna(0.0) - merged["_earliest_inv"].fillna(0.0)
    merged["_snap_inventory_change_rate"] = inv_delta / delta_hours.clip(lower=1.0)

    # Zone price trend: fractional price change from earliest to matched snapshot
    snap_price = merged["_snap_price_avg"].fillna(merged["_snap_price_min"].fillna(0.0))
    earliest_price = merged["_earliest_price_avg"].fillna(0.0)
    merged["_snap_zone_price_trend"] = (snap_price - earliest_price) / earliest_price.clip(
        lower=1.0
    )

    # Snapshot count: log1p(number of snapshots for this event-zone)
    merged["_snap_count"] = np.log1p(merged["_snap_count_raw"].fillna(0.0))

    # Price range width at matched snapshot: (max - min) / avg
    snap_avg = merged["_snap_price_avg"].fillna(merged["_snap_price_min"].fillna(0.0))
    snap_min = merged["_snap_price_min"].fillna(0.0)
    snap_max = merged["_snap_price_max"].fillna(snap_avg)
    merged["_snap_price_range"] = (snap_max - snap_min) / snap_avg.clip(lower=1.0)

    # Zero out features for rows where merge_asof found no backward match.
    # Without this, snap_agg aggregates produce nonsensical deltas against
    # the NaN-filled matched snapshot values.
    no_match = merged["_snap_ts"].isna()
    for col in [
        "_snap_inventory_change_rate",
        "_snap_zone_price_trend",
        "_snap_count",
        "_snap_price_range",
    ]:
        merged.loc[no_match, col] = 0.0
        merged[col] = merged[col].fillna(0.0)

    # Drop temporary join columns
    drop_cols = [
        "_zone",
        "_snap_ts",
        "_snap_price_avg",
        "_snap_price_min",
        "_snap_price_max",
        "_snap_inv",
        "_earliest_ts",
        "_earliest_price_avg",
        "_earliest_inv",
        "_snap_count_raw",
    ]
    merged = merged.drop(columns=[c for c in drop_cols if c in merged.columns])

    # Restore original row order
    merged = merged.sort_values("_pos").drop(columns=["_pos"])
    merged.index = listings_df.index
    return merged


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
        self._log_transformed_cols: list[str] = []
        self._target_transform: TargetTransform | None = None

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
        elif self._model_type == "xgboost":
            return XGBoostModel(params=params) if params else XGBoostModel()
        elif self._model_type == "catboost":
            return CatBoostModel(params=params) if params else CatBoostModel()
        elif self._model_type == "stacking":
            return StackingEnsemble()
        elif self._model_type == "stacking_v2":
            return StackingEnsembleV2()
        elif self._model_type == "residual":
            return ResidualModel()
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
        sample_weight_strategy: SampleWeightStrategy = "none",
        snapshot_df: pd.DataFrame | None = None,
        outlier_strategy: OutlierStrategy = "global_p95",
        target_transform: str = "log",
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
                "sqrt_price": weight = sqrt(price/median_price), upweights expensive tickets.
                "log_price": weight = log1p(price)/mean(log1p(prices)), moderate upweight.
                "inverse_price_quartile": weight by inverse quartile frequency so each
                price quartile contributes equally to the loss.
            snapshot_df: Optional raw snapshot DataFrame from DataLoader.load_snapshots().
                When provided, temporal delta features are joined to each split
                independently AFTER split_raw() to prevent leakage.
            outlier_strategy: How to handle price outliers before splitting.
                "global_p95" (default): global 95th percentile cap (backward-compatible).
                "zone_winsorize": per-zone Winsorization at (p2, p98) with fallback.
                "none": no outlier capping.
            target_transform: Target transform strategy.
                "log" (default): np.log1p/np.expm1 (backward-compatible).
                "boxcox": scipy Box-Cox with fitted lambda.
                "sqrt": square root transform (gentler compression).

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
        if outlier_strategy == "global_p95":
            df = self._cap_price_outliers(df)
        elif outlier_strategy == "zone_winsorize":
            df = self._winsorize_by_zone(df)
        elif outlier_strategy == "none":
            print("Outlier capping: disabled")
        else:
            raise ValueError(f"Unknown outlier_strategy: {outlier_strategy!r}")

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

        # Enrich each split independently with snapshot features (AFTER split to prevent leakage)
        train_df = raw_split.train_df
        val_df = raw_split.val_df
        test_df = raw_split.test_df

        if snapshot_df is not None and not snapshot_df.empty:
            from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper

            zone_mapper = SeatZoneMapper()
            print(f"Joining snapshot features ({len(snapshot_df)} snapshots)...")
            train_df = _join_snapshot_features(train_df, snapshot_df, zone_mapper)
            val_df = _join_snapshot_features(val_df, snapshot_df, zone_mapper)
            test_df = _join_snapshot_features(test_df, snapshot_df, zone_mapper)

        # Fit feature pipeline on training data ONLY
        print("Extracting features (fit on train only)...")
        _pipeline_kwargs: dict[str, Any] = {
            "include_momentum": False,
            "popularity_service": popularity_service,
        }
        if pipeline_kwargs:
            _pipeline_kwargs.update(pipeline_kwargs)
        self._feature_pipeline = FeaturePipeline(**_pipeline_kwargs)
        self._feature_pipeline.fit(train_df)

        # Transform each split independently
        X_train = self._feature_pipeline.transform(train_df)
        X_val = self._feature_pipeline.transform(val_df)
        X_test = self._feature_pipeline.transform(test_df)

        # Remove zero-variance features (constants add noise, waste splits)
        zero_var_cols = [c for c in X_train.columns if X_train[c].nunique() <= 1]
        if zero_var_cols:
            print(f"Removing {len(zero_var_cols)} zero-variance features: {zero_var_cols}")
            X_train = X_train.drop(columns=zero_var_cols)
            X_val = X_val.drop(columns=zero_var_cols)
            X_test = X_test.drop(columns=zero_var_cols)

        # Log-transform price-level features to align with log-transformed target.
        # Exclude derived statistics (_std, _cv, _ratio, etc.) — these are already
        # scale-independent and log-transforming them is statistically inappropriate.
        price_cols = [
            c
            for c in X_train.columns
            if ("price" in c.lower() or "avg" in c.lower() or "median" in c.lower())
            and not any(c.lower().endswith(suffix) for suffix in _LOG_EXCLUDE_SUFFIXES)
        ]
        self._log_transformed_cols = price_cols
        if price_cols:
            print(f"Log-transforming {len(price_cols)} price features")
            for c in price_cols:
                X_train[c] = np.log1p(X_train[c].clip(lower=0))
                X_val[c] = np.log1p(X_val[c].clip(lower=0))
                X_test[c] = np.log1p(X_test[c].clip(lower=0))

        # Transform target for better handling of skewed price distribution
        tt = create_target_transform(target_transform)
        tt.fit(train_df[self._target_col].values)
        self._target_transform = tt
        print(f"Target transform: {tt.name}")

        y_train = tt.transform(train_df[self._target_col].values)
        y_val = tt.transform(val_df[self._target_col].values)
        y_test_raw = test_df[self._target_col].values  # keep raw for evaluation

        print(f"Feature shape: {X_train.shape}")
        print(f"Features: {list(X_train.columns)}")
        print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        # Compute sample weights for training set
        sample_weight: np.ndarray[Any, Any] | None = None
        if sample_weight_strategy == "inverse_artist_freq":
            if "artist_or_team" in train_df.columns:
                artist_counts = train_df["artist_or_team"].value_counts()
                weights = train_df["artist_or_team"].map(lambda a: 1.0 / artist_counts[a])
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
        elif sample_weight_strategy == "sqrt_price":
            prices = train_df[self._target_col].values.astype(float)
            median_price = np.median(prices)
            raw_weights = np.sqrt(prices / max(median_price, 1.0))
            # Normalize so weights sum to n_train
            sample_weight = raw_weights / raw_weights.sum() * len(raw_weights)
            print(
                f"Sample weights: sqrt_price "
                f"(min={sample_weight.min():.3f}, max={sample_weight.max():.3f})"
            )
        elif sample_weight_strategy == "log_price":
            prices = train_df[self._target_col].values.astype(float)
            log_prices = np.log1p(prices)
            mean_log = np.mean(log_prices)
            raw_weights = log_prices / max(mean_log, 1e-6)
            # Normalize so weights sum to n_train
            sample_weight = raw_weights / raw_weights.sum() * len(raw_weights)
            print(
                f"Sample weights: log_price "
                f"(min={sample_weight.min():.3f}, max={sample_weight.max():.3f})"
            )
        elif sample_weight_strategy == "inverse_price_quartile":
            prices = train_df[self._target_col].values.astype(float)
            q25, q50, q75 = np.percentile(prices, [25, 50, 75])
            # Assign quartile labels
            quartiles = np.where(
                prices <= q25,
                1,
                np.where(prices <= q50, 2, np.where(prices <= q75, 3, 4)),
            )
            # Weight = 1 / count_in_quartile, so each quartile contributes equally
            unique, counts = np.unique(quartiles, return_counts=True)
            quartile_counts = dict(zip(unique, counts, strict=False))
            raw_weights = np.array([1.0 / quartile_counts[q] for q in quartiles])
            # Normalize so weights sum to n_train
            sample_weight = raw_weights / raw_weights.sum() * len(raw_weights)
            # Cap max weight ratio at 4:1 to avoid extreme imbalance
            max_ratio = sample_weight.max() / max(sample_weight.min(), 1e-10)
            if max_ratio > 4.0:
                sample_weight = np.clip(
                    sample_weight, sample_weight.min(), sample_weight.min() * 4.0
                )
                sample_weight = sample_weight / sample_weight.sum() * len(sample_weight)
            print(
                f"Sample weights: inverse_price_quartile "
                f"(min={sample_weight.min():.3f}, max={sample_weight.max():.3f})"
            )

        # Train model
        print("Training model...")
        start_time = time.time()

        self._model = self._create_model(params=params)

        # stacking_v2 needs explicit timestamps because upstream artist-stratified
        # splitter does not produce globally time-sorted rows.
        extra_fit_kwargs: dict[str, Any] = {}
        if self._model_type == "stacking_v2" and "timestamp" in train_df.columns:
            extra_fit_kwargs["timestamps"] = train_df["timestamp"]

        try:
            self._model.fit(  # type: ignore[call-arg]
                X_train,
                pd.Series(y_train),
                X_val,
                pd.Series(y_val),
                sample_weight=sample_weight,
                **extra_fit_kwargs,
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
                "boosting_type", LightGBMModel.DEFAULT_PARAMS.get("boosting_type", "gbdt")
            )
            if boosting != "dart":
                print(
                    f"WARNING: best_iteration=0 with boosting_type={boosting!r}. "
                    "Early stopping fired at round 0 — check learning_rate and early_stopping_rounds."
                )

        if len(y_test_raw) == 0:
            print("No test set (test_ratio=0.0) — skipping evaluation")
            self._metrics = TrainingMetrics(
                model_version=self._model_version,
                model_type=self._model_type,
                trained_at=datetime.now(UTC),
                n_train_samples=len(X_train),
                n_val_samples=len(X_val),
                n_test_samples=0,
                n_features=X_train.shape[1],
                training_time_seconds=training_time,
                mae=0.0,
                rmse=0.0,
                mape=0.0,
                r2=0.0,
                feature_importance=self._model.get_feature_importance(),
                best_iteration=getattr(self._model, "best_iteration", None),
            )
            return self._metrics

        # Derive zone labels for per-zone evaluation breakdown
        test_zones = None
        if "section" in test_df.columns:
            from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper

            _zone_mapper = SeatZoneMapper()
            test_zones = (
                test_df["section"]
                .apply(lambda s: _zone_mapper.normalize_zone_name(str(s)).value)
                .values
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
            log_target=(target_transform == "log"),
            target_transform=tt if target_transform != "log" else None,
            zones=test_zones,
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
            "BTS (방탄소년단)": "BTS",
            "Rush - Rock Band": "Rush",
            "Taylor Swift - Pop": "Taylor Swift",
            "Bad Bunny - Latin Trap": "Bad Bunny",
            "Drake - Hip Hop": "Drake",
            "Beyoncé - R&B": "Beyoncé",
            "The Weeknd - R&B": "The Weeknd",
            "Billie Eilish - Pop": "Billie Eilish",
            "Harry Styles - Pop": "Harry Styles",
            "Coldplay - Rock": "Coldplay",
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

    @staticmethod
    def _winsorize_by_zone(
        df: pd.DataFrame,
        target_col: str = "listing_price",
        lower_pct: float = 2.0,
        upper_pct: float = 98.0,
        min_zone_samples: int = 20,
    ) -> pd.DataFrame:
        """Apply zone-aware Winsorization to cap outliers per seat zone.

        Groups listings by normalized seat zone and clips prices at
        (lower_pct, upper_pct) percentiles per zone. Zones with fewer
        than min_zone_samples listings fall back to global (p5, p95).

        Args:
            df: Input DataFrame with section and target columns
            target_col: Price column to Winsorize
            lower_pct: Lower percentile for clipping (default 2.0)
            upper_pct: Upper percentile for clipping (default 98.0)
            min_zone_samples: Minimum samples per zone before falling back to global

        Returns:
            DataFrame with Winsorized prices
        """
        if target_col not in df.columns:
            return df

        df = df.copy()
        df[target_col] = df[target_col].astype(float)

        # Compute global fallback caps
        global_lower = float(df[target_col].quantile(5.0 / 100.0))
        global_upper = float(df[target_col].quantile(95.0 / 100.0))

        # Derive normalized zone from section if available
        if "section" in df.columns:
            from ticket_price_predictor.normalization.seat_zones import SeatZoneMapper

            mapper = SeatZoneMapper()
            df["_winsorize_zone"] = df["section"].apply(
                lambda s: mapper.normalize_zone_name(str(s)).value
            )
        else:
            # No section column — use global caps
            print(
                f"Zone Winsorization: no 'section' column, using global p{lower_pct:.0f}/p{upper_pct:.0f}"
            )
            lower_cap = float(df[target_col].quantile(lower_pct / 100.0))
            upper_cap = float(df[target_col].quantile(upper_pct / 100.0))
            before_clipped = ((df[target_col] < lower_cap) | (df[target_col] > upper_cap)).sum()
            df[target_col] = df[target_col].clip(lower=lower_cap, upper=upper_cap)
            print(f"  Global caps: ${lower_cap:.2f} - ${upper_cap:.2f} ({before_clipped} clipped)")
            return df

        total_clipped = 0
        for zone, group_idx in df.groupby("_winsorize_zone").groups.items():
            zone_prices = df.loc[group_idx, target_col]

            if len(zone_prices) < min_zone_samples:
                # Fall back to global caps for small zones
                lower_cap = global_lower
                upper_cap = global_upper
                source = "global fallback"
            else:
                lower_cap = float(zone_prices.quantile(lower_pct / 100.0))
                upper_cap = float(zone_prices.quantile(upper_pct / 100.0))
                source = f"p{lower_pct:.0f}/p{upper_pct:.0f}"

            clipped = ((zone_prices < lower_cap) | (zone_prices > upper_cap)).sum()
            total_clipped += clipped
            df.loc[group_idx, target_col] = zone_prices.clip(lower=lower_cap, upper=upper_cap)
            print(
                f"  Zone {zone}: ${lower_cap:.2f} - ${upper_cap:.2f} "
                f"({source}, {len(zone_prices)} samples, {clipped} clipped)"
            )

        df = df.drop(columns=["_winsorize_zone"])
        print(f"Zone Winsorization: {total_clipped} total values clipped")
        return df

    def train_with_split(
        self,
        split: DataSplit,
        log_target: bool = False,
    ) -> TrainingMetrics:
        """Train model with pre-computed split.

        Args:
            split: DataSplit object
            log_target: If True, inverse-transform predictions from log-space before evaluation.

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
            log_target=log_target,
        )

        ModelEvaluator.print_metrics(self._metrics)

        return self._metrics

    def train_with_params(
        self,
        split: DataSplit,
        params: dict[str, Any],
        log_target: bool = False,
    ) -> TrainingMetrics:
        """Train model with custom hyperparameters.

        Args:
            split: DataSplit object
            params: Hyperparameter dictionary
            log_target: If True, inverse-transform predictions from log-space before evaluation.

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
            log_target=log_target,
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

        # Save fitted feature pipeline (enables inference with trained statistics)
        if self._feature_pipeline is not None:
            pipeline_path = output_dir / f"{self._model_type}_{self._model_version}_pipeline.joblib"
            self._feature_pipeline.save(pipeline_path)
            print(f"Pipeline saved to: {pipeline_path}")

        # Save log-transformed column list and target transform info
        meta: dict[str, Any] = {}
        if self._log_transformed_cols:
            meta["log_transformed_cols"] = self._log_transformed_cols
        if self._target_transform is not None:
            meta["target_transform"] = self._target_transform.name
        if meta:
            meta_path = output_dir / f"{self._model_type}_{self._model_version}_meta.json"
            meta_path.write_text(json.dumps(meta))
            print(f"Meta saved to: {meta_path}")

        # Save target transform object for inference
        if self._target_transform is not None:
            import joblib

            tt_path = (
                output_dir / f"{self._model_type}_{self._model_version}_target_transform.joblib"
            )
            joblib.dump(self._target_transform, tt_path)
            print(f"Target transform saved to: {tt_path}")

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

    @staticmethod
    def prune_features_by_importance(
        X_train: pd.DataFrame,
        X_val: pd.DataFrame,
        X_test: pd.DataFrame,
        importance: dict[str, float],
        threshold: float = 0.001,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
        """Remove features below importance threshold.

        After initial training, prunes low-importance features to reduce noise
        and improve generalization on small datasets.

        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features
            importance: Feature importance dict (normalized, values sum to ~1.0)
            threshold: Minimum importance to keep (default 0.1%)

        Returns:
            Tuple of (X_train, X_val, X_test, removed_features)
        """
        features_to_remove = []
        for f in X_train.columns:
            imp = importance.get(f, 0.0)
            if imp < threshold:
                features_to_remove.append(f)

        if features_to_remove:
            print(
                f"Pruning {len(features_to_remove)} features "
                f"below {threshold * 100:.1f}% importance"
            )
            X_train = X_train.drop(columns=features_to_remove)
            X_val = X_val.drop(columns=features_to_remove)
            X_test = X_test.drop(columns=features_to_remove)
            print(f"  Remaining features: {X_train.shape[1]}")
        else:
            print("No features below importance threshold — skipping pruning")

        return X_train, X_val, X_test, features_to_remove

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
        elif model_type == "xgboost":
            return XGBoostModel.load(model_path)
        elif model_type == "catboost":
            return CatBoostModel.load(model_path)
        elif model_type == "stacking":
            return StackingEnsemble.load(model_path)
        elif model_type == "stacking_v2":
            return StackingEnsembleV2.load(model_path)
        elif model_type == "residual":
            return ResidualModel.load(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
