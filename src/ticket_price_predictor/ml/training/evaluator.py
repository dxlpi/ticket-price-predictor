"""Model evaluation utilities."""

from datetime import datetime

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from ticket_price_predictor.ml.models.base import PriceModel
from ticket_price_predictor.ml.schemas import TrainingMetrics


class ModelEvaluator:
    """Evaluate model performance."""

    @staticmethod
    def compute_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> dict[str, float]:
        """Compute regression metrics.

        Args:
            y_true: True values
            y_pred: Predicted values

        Returns:
            Dictionary of metrics
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

        return {
            "mae": float(mae),
            "rmse": float(rmse),
            "mape": float(mape),
            "r2": float(r2),
        }

    @staticmethod
    def compute_direction_accuracy(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prev: np.ndarray | None = None,
    ) -> float:
        """Compute accuracy of direction predictions.

        Args:
            y_true: True values
            y_pred: Predicted values
            y_prev: Previous values (for computing actual direction)

        Returns:
            Direction accuracy (0-1)
        """
        if y_prev is None:
            # Assume comparing against mean
            y_prev = np.full_like(y_true, np.mean(y_true))

        true_direction = np.sign(y_true - y_prev)
        pred_direction = np.sign(y_pred - y_prev)

        return float(np.mean(true_direction == pred_direction))

    @staticmethod
    def evaluate_model(
        model: PriceModel,
        X_test: np.ndarray,
        y_test: np.ndarray,
        n_train: int = 0,
        n_val: int = 0,
        training_time: float = 0.0,
        model_version: str = "v1",
    ) -> TrainingMetrics:
        """Full model evaluation.

        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            n_train: Number of training samples
            n_val: Number of validation samples
            training_time: Training time in seconds
            model_version: Model version string

        Returns:
            TrainingMetrics object
        """
        y_pred = model.predict(X_test)
        metrics = ModelEvaluator.compute_metrics(y_test, y_pred)

        # Get feature importance
        importance = model.get_feature_importance()

        # Get best iteration if available
        best_iter = None
        if hasattr(model, "best_iteration"):
            best_iter = model.best_iteration

        return TrainingMetrics(
            model_version=model_version,
            model_type=model.name,
            trained_at=datetime.utcnow(),
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
        if metrics.feature_importance:
            print("Top features:")
            for i, (name, imp) in enumerate(list(metrics.feature_importance.items())[:10]):
                print(f"  {i+1}. {name}: {imp:.4f}")
