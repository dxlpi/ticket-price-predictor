"""Base model protocol for price prediction."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy.typing as npt
import pandas as pd


class PriceModel(ABC):
    """Abstract base class for price prediction models."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return model name."""
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """Return True if model has been fitted."""
        ...

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "PriceModel":
        """Fit the model.

        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)

        Returns:
            self
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> npt.NDArray[Any]:
        """Make predictions.

        Args:
            X: Feature DataFrame

        Returns:
            Array of predictions
        """
        ...

    def predict_with_uncertainty(
        self, X: pd.DataFrame
    ) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
        """Make predictions with uncertainty estimates.

        Default implementation returns point predictions with no uncertainty.

        Args:
            X: Feature DataFrame

        Returns:
            Tuple of (predictions, lower_bounds, upper_bounds)
        """
        preds = self.predict(X)
        # Default: no uncertainty (bounds equal predictions)
        return preds, preds, preds

    @abstractmethod
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        ...

    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk.

        Args:
            path: Path to save model
        """
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: Path) -> "PriceModel":
        """Load model from disk.

        Args:
            path: Path to load model from

        Returns:
            Loaded model
        """
        ...

    def get_params(self) -> dict[str, Any]:
        """Get model parameters."""
        return {}

    def set_params(self, **params: Any) -> "PriceModel":  # noqa: ARG002
        """Set model parameters."""
        return self
