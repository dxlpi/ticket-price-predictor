"""Model evaluation utilities."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

import numpy as np
import numpy.typing as npt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.schemas import TrainingMetrics

if TYPE_CHECKING:
    from ticket_price_predictor.ml.training.target_transforms import TargetTransform


class ModelEvaluator:
    """Evaluate model performance."""

    @staticmethod
    def compute_metrics(
        y_true: npt.NDArray[Any],
        y_pred: npt.NDArray[Any],
        zones: npt.NDArray[Any] | None = None,
    ) -> dict[str, Any]:
        """Compute regression metrics with optional per-quartile and per-zone breakdowns.

        Args:
            y_true: True values (raw price scale)
            y_pred: Predicted values (raw price scale)
            zones: Optional array of zone labels (same length as y_true)

        Returns:
            Dictionary of metrics including quartile_mae and zone_mae
        """
        # Ensure numpy arrays
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        # Basic metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)

        # MAPE (with protection against division by zero)
        mask = y_true != 0
        if mask.any():
            mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        else:
            mape = 0.0

        # Per-quartile MAE
        quartile_mae: dict[str, float] = {}
        q25, q50, q75 = np.percentile(y_true, [25, 50, 75])
        for label, qmask in [
            ("Q1", y_true <= q25),
            ("Q2", (y_true > q25) & (y_true <= q50)),
            ("Q3", (y_true > q50) & (y_true <= q75)),
            ("Q4", y_true > q75),
        ]:
            if qmask.any():
                quartile_mae[label] = float(mean_absolute_error(y_true[qmask], y_pred[qmask]))

        # Per-zone MAE
        zone_mae: dict[str, float] = {}
        if zones is not None:
            zones_arr = np.asarray(zones)
            for zone in np.unique(zones_arr):
                zmask = zones_arr == zone
                if zmask.any():
                    zone_mae[str(zone)] = float(mean_absolute_error(y_true[zmask], y_pred[zmask]))

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
            "quartile_mae": quartile_mae,
            "zone_mae": zone_mae,
        }

    @staticmethod
    def evaluate_model(
        model: PriceModel,
        X_test: npt.NDArray[Any],
        y_test: npt.NDArray[Any],
        n_train: int = 0,
        n_val: int = 0,
        training_time: float = 0.0,
        model_version: str = "v1",
        log_target: bool = False,
        target_transform: TargetTransform | None = None,
        zones: npt.NDArray[Any] | None = None,
    ) -> TrainingMetrics:
        """Full model evaluation.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target (raw scale when log_target=True or target_transform set)
            n_train: Number of training samples
            n_val: Number of validation samples
            training_time: Training time in seconds
            model_version: Model version string
            log_target: If True, inverse-transform predictions from log-space
            target_transform: Optional fitted TargetTransform for inverse-transforming
                predictions. Takes precedence over log_target when set.
            zones: Optional zone label array for per-zone MAE breakdown

        Returns:
            TrainingMetrics object
        """
        y_pred = model.predict(X_test)

        if target_transform is not None:
            # Use the fitted transform's inverse
            y_pred = target_transform.inverse_transform(y_pred)
        elif log_target:
            # Model predicts in log-space; inverse-transform to raw prices
            y_pred = np.expm1(y_pred)
            # Clip negative predictions (can occur from expm1 on small values)
            y_pred = np.clip(y_pred, 0, None)

        metrics = ModelEvaluator.compute_metrics(y_test, y_pred, zones=zones)

        # Get feature importance
        importance = model.get_feature_importance()

        # Get best iteration if available
        best_iter = None
        if hasattr(model, "best_iteration"):
            best_iter = model.best_iteration

        return TrainingMetrics(
            model_version=model_version,
            model_type=model.name,
            trained_at=datetime.now(UTC),
            n_train_samples=n_train,
            n_val_samples=n_val,
            n_test_samples=len(y_test),
            n_features=X_test.shape[1] if hasattr(X_test, "shape") else 0,
            mae=metrics["mae"],
            rmse=metrics["rmse"],
            mape=metrics["mape"],
            r2=metrics["r2"],
            training_time_seconds=training_time,
            best_iteration=best_iter,
            feature_importance=importance,
            quartile_mae=metrics["quartile_mae"],
            zone_mae=metrics["zone_mae"],
        )

    @staticmethod
    def print_metrics(metrics: TrainingMetrics) -> None:
        """Print metrics in a readable format.

        Args:
            metrics: TrainingMetrics object
        """
        print("=" * 50)
        print(f"Model: {metrics.model_type} ({metrics.model_version})")
        print("=" * 50)
        print(f"Training samples:   {metrics.n_train_samples:,}")
        print(f"Validation samples: {metrics.n_val_samples:,}")
        print(f"Test samples:       {metrics.n_test_samples:,}")
        print(f"Features:           {metrics.n_features}")
        print()
        print("Metrics:")
        print(f"  MAE:  ${metrics.mae:.2f}")
        print(f"  RMSE: ${metrics.rmse:.2f}")
        print(f"  MAPE: {metrics.mape:.1f}%")
        print(f"  R²:   {metrics.r2:.4f}")
        print()
        print(f"Training time: {metrics.training_time_seconds:.1f}s")
        if metrics.best_iteration:
            print(f"Best iteration: {metrics.best_iteration}")
        print()
        if metrics.quartile_mae:
            print("Per-quartile MAE:")
            for q, mae_val in sorted(metrics.quartile_mae.items()):
                print(f"  {q}: ${mae_val:.2f}")
            print()
        if metrics.zone_mae:
            print("Per-zone MAE:")
            for zone, mae_val in sorted(metrics.zone_mae.items()):
                print(f"  {zone}: ${mae_val:.2f}")
            print()
        if metrics.feature_importance:
            print("Top features:")
            for i, (name, imp) in enumerate(list(metrics.feature_importance.items())[:10]):
                print(f"  {i + 1}. {name}: {imp:.4f}")
