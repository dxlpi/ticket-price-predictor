"""Sale probability training pipeline.

Trains a binary LightGBM classifier to predict listing sale probability
(a CVR analogue for commerce-style ranking). Follows the split-before-fit
invariant: labels are built before splitting, features are fitted on the
training split only.

Two-stage architecture:
  Stage 1: PricePredictor (price regression) — optional dependency
  Stage 2: SaleProbabilityModel (binary classification)

Sale-specific features are computed per-split to prevent leakage:
  - relative_price_position: percentile rank within split's event+zone
  - price_vs_zone_median: ratio to split's event+zone median
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ticket_price_predictor.ml.features.pipeline import FeaturePipeline
from ticket_price_predictor.ml.models.sale_probability import SaleProbabilityModel
from ticket_price_predictor.ml.schemas import TrainingMetrics
from ticket_price_predictor.ml.training.label_builder import (
    InventoryDepletionLabeler,
    SaleLabelBuilder,
)
from ticket_price_predictor.ml.training.splitter import TimeBasedSplitter


def _compute_sale_features(
    df: pd.DataFrame,
    train_df: pd.DataFrame | None = None,
    price_predictor: Any | None = None,
) -> pd.DataFrame:
    """Compute sale-specific features for the classifier.

    All group statistics (medians, percentiles) are derived from train_df when
    provided (training split), or from df itself (for test/val splits using
    training-time statistics). This prevents leakage.

    Args:
        df: Current split's DataFrame (with listing_price, event_id, seat_zone cols)
        train_df: Training split DataFrame (used to compute group stats)
        price_predictor: Optional PricePredictor for price_vs_predicted feature

    Returns:
        df with added sale-specific feature columns
    """
    result = df.copy()
    ref = train_df if train_df is not None else df

    # relative_price_position: percentile rank of listing_price within event+zone
    # Computed relative to training distribution (ref) to prevent leakage
    if "listing_price" in df.columns and "event_id" in df.columns:
        group_col = "seat_zone" if "seat_zone" in df.columns else None
        group_keys = ["event_id", group_col] if group_col else ["event_id"]

        # Build group-level statistics from reference (training) data
        group_stats = (
            ref.groupby(group_keys)["listing_price"].agg(["median", "std", "count"]).reset_index()
        )
        group_stats.columns = [*group_keys, "_grp_median", "_grp_std", "_grp_count"]

        result = result.merge(group_stats, on=group_keys, how="left")

        # relative_price_position: z-score relative to group median/std
        result["relative_price_position"] = np.where(
            result["_grp_std"].fillna(0) > 0,
            (result["listing_price"] - result["_grp_median"]) / result["_grp_std"],
            0.0,
        )

        # price_vs_zone_median: ratio to training-split zone median
        result["price_vs_zone_median"] = np.where(
            result["_grp_median"].fillna(0) > 0,
            result["listing_price"] / result["_grp_median"],
            1.0,
        )

        result = result.drop(columns=["_grp_median", "_grp_std", "_grp_count"])

    # price_vs_predicted: requires a trained price predictor
    if price_predictor is not None and "listing_price" in df.columns:
        try:
            preds = price_predictor.predict_batch(df)
            predicted_prices = np.array([p.predicted_price for p in preds])
            result["price_vs_predicted"] = np.where(
                predicted_prices > 0,
                df["listing_price"].values / predicted_prices,
                1.0,
            )
        except Exception:  # noqa: BLE001
            # If prediction fails (e.g. missing features), skip this feature
            pass

    return result


class SaleProbabilityTrainer:
    """Trains and evaluates the sale probability binary classifier.

    Follows the split-before-fit invariant from the project's training pipeline.
    Sale-specific features are computed per-split to prevent distribution leakage.
    """

    def __init__(
        self,
        label_window_hours: int = 24,
        min_absent_scrapes: int = 2,
        pipeline_kwargs: dict[str, Any] | None = None,
        model_params: dict[str, Any] | None = None,
        label_strategy: str = "inventory_depletion",
    ) -> None:
        """Initialize trainer.

        Args:
            label_window_hours: Hours to look ahead for label construction
            min_absent_scrapes: Minimum consecutive absent scrapes for sold=1
                               (used only when label_strategy="disappearance")
            pipeline_kwargs: Kwargs for FeaturePipeline constructor
            model_params: Override params for SaleProbabilityModel
            label_strategy: Label construction strategy.
                "inventory_depletion" (default): uses aggregate inventory depletion
                    from snapshot data. Robust to partial scraping. Requires
                    snapshots_df to be passed to train().
                "disappearance": uses individual listing disappearance (legacy).
                    Unreliable with current scraper's partial captures.
        """
        if label_strategy not in ("inventory_depletion", "disappearance"):
            raise ValueError(
                f"label_strategy must be 'inventory_depletion' or 'disappearance', "
                f"got {label_strategy!r}"
            )
        self._label_window_hours = label_window_hours
        self._min_absent_scrapes = min_absent_scrapes
        self._pipeline_kwargs = pipeline_kwargs or {}
        self._model_params = model_params
        self._label_strategy = label_strategy

    def train(
        self,
        listings_df: pd.DataFrame,
        model_version: str = "v1",
        price_predictor: Any | None = None,
        snapshots_df: pd.DataFrame | None = None,
    ) -> tuple[SaleProbabilityModel, TrainingMetrics]:
        """Train the sale probability classifier.

        Steps:
        1. Build sold/not-sold labels (strategy from __init__)
        2. Split raw data temporally (split-before-fit)
        3. Fit feature pipeline on training split only
        4. Compute sale-specific features per split
        5. Train SaleProbabilityModel
        6. Evaluate with AUC-ROC, precision, recall, F1, calibration

        Args:
            listings_df: Raw listings DataFrame with temporal coverage
            model_version: Version string for metrics
            price_predictor: Optional fitted PricePredictor for price_vs_predicted
            snapshots_df: Snapshot DataFrame with inventory_remaining column.
                         Required when label_strategy="inventory_depletion".

        Returns:
            (trained_model, training_metrics)
        """
        start_time = time.time()

        # Step 1: Build labels (before split — labels are from historical observation)
        if self._label_strategy == "inventory_depletion":
            if snapshots_df is None:
                raise ValueError(
                    "snapshots_df is required when label_strategy='inventory_depletion'. "
                    "Pass snapshot data or use label_strategy='disappearance'."
                )
            labeler = InventoryDepletionLabeler(window_hours=self._label_window_hours)
            labeled_df = labeler.build_labels(listings_df, snapshots_df)
        else:
            label_builder = SaleLabelBuilder()
            label_builder.verify_listing_id_stability(listings_df)
            labeled_df = label_builder.build_labels(
                listings_df,
                window_hours=self._label_window_hours,
                min_absent_scrapes=self._min_absent_scrapes,
            )

        # Keep only rows with definitive labels
        labeled_df = labeled_df[labeled_df["sold"].notna()].copy()
        labeled_df["sold"] = labeled_df["sold"].astype(int)

        # Drop _label_depletion_rate to prevent label leakage into feature extraction
        if "_label_depletion_rate" in labeled_df.columns:
            labeled_df = labeled_df.drop(columns=["_label_depletion_rate"])

        # Step 2: Temporal split (reuse existing splitter)
        splitter = TimeBasedSplitter()
        split = splitter.split_raw(labeled_df)

        train_df = split.train_df
        val_df = split.val_df
        test_df = split.test_df

        # Step 3: Fit feature pipeline on training data only
        pipeline = FeaturePipeline(**self._pipeline_kwargs)
        pipeline.fit(train_df)

        X_train = pipeline.transform(train_df)
        X_val = pipeline.transform(val_df)
        X_test = pipeline.transform(test_df)

        # Step 4: Compute sale-specific features per-split
        # Medians/percentiles always derived from training split
        X_train_aug = _compute_sale_features(
            train_df, train_df=train_df, price_predictor=price_predictor
        )
        X_val_aug = _compute_sale_features(
            val_df, train_df=train_df, price_predictor=price_predictor
        )
        X_test_aug = _compute_sale_features(
            test_df, train_df=train_df, price_predictor=price_predictor
        )

        # Add sale-specific features to feature matrices
        sale_feature_cols = [
            c
            for c in ["relative_price_position", "price_vs_zone_median", "price_vs_predicted"]
            if c in X_train_aug.columns
        ]
        for col in sale_feature_cols:
            X_train[col] = X_train_aug[col].values
            X_val[col] = X_val_aug[col].values
            X_test[col] = X_test_aug[col].values

        y_train = train_df["sold"]
        y_val = val_df["sold"]
        y_test = test_df["sold"]

        # Step 5: Train model
        model = SaleProbabilityModel(params=self._model_params)
        model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

        # Step 6: Evaluate on test set
        y_pred_prob = model.predict(X_test)
        y_pred_class = model.predict_class(X_test, threshold=0.5)

        auc = float(roc_auc_score(y_test, y_pred_prob)) if y_test.nunique() > 1 else None
        prec = float(precision_score(y_test, y_pred_class, zero_division=0))
        rec = float(recall_score(y_test, y_pred_class, zero_division=0))
        f1 = float(f1_score(y_test, y_pred_class, zero_division=0))

        # Expected calibration error
        try:
            prob_true, prob_pred = calibration_curve(y_test, y_pred_prob, n_bins=10)
            ece = float(np.mean(np.abs(prob_true - prob_pred)))
        except Exception:  # noqa: BLE001
            ece = None

        elapsed = time.time() - start_time

        metrics = TrainingMetrics(
            model_version=model_version,
            model_type="sale_probability",
            n_train_samples=len(X_train),
            n_val_samples=len(X_val),
            n_test_samples=len(X_test),
            n_features=X_train.shape[1],
            # Regression fields set to 0 (not applicable for classifier)
            mae=0.0,
            rmse=0.0,
            mape=0.0,
            r2=0.0,
            # Classification fields
            auc_roc=auc,
            precision=prec,
            recall=rec,
            f1=f1,
            calibration_error=ece,
            training_time_seconds=elapsed,
            best_iteration=model.best_iteration,
            feature_importance=model.get_feature_importance(top_k=10),
        )

        return model, metrics

    def save(
        self,
        model: SaleProbabilityModel,
        metrics: TrainingMetrics,
        output_dir: Path,
        version: str = "v1",
    ) -> None:
        """Save model and metrics to disk.

        Args:
            model: Trained SaleProbabilityModel
            metrics: Training metrics
            output_dir: Directory to save to
            version: Version string
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        model_path = output_dir / f"sale_probability_{version}.joblib"
        metrics_path = output_dir / f"sale_probability_{version}_metrics.json"

        model.save(model_path)
        metrics_path.write_text(metrics.model_dump_json(indent=2))
