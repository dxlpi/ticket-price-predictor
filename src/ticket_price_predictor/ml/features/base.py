"""Base classes for feature extraction."""

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class FeatureExtractor(ABC):
    """Base class for feature extractors."""

    @property
    @abstractmethod
    def feature_names(self) -> list[str]:
        """Return list of feature names this extractor produces."""
        ...

    @abstractmethod
    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from input DataFrame.

        Args:
            df: Input DataFrame with raw data

        Returns:
            DataFrame with extracted features (same number of rows)
        """
        ...

    def fit(self, df: pd.DataFrame) -> "FeatureExtractor":  # noqa: ARG002
        """Fit the extractor on training data (optional).

        Default implementation does nothing. Override for extractors
        that need to learn from training data (e.g., encoders).

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        return self

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters for serialization."""
        return {}

    def set_params(self, params: dict[str, Any]) -> None:  # noqa: ARG002, B027
        """Set extractor parameters from deserialization."""
        pass
