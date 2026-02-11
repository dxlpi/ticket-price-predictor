"""Feature pipeline orchestration."""

from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor
from ticket_price_predictor.ml.features.event import EventFeatureExtractor
from ticket_price_predictor.ml.features.performer import PerformerFeatureExtractor
from ticket_price_predictor.ml.features.seating import SeatingFeatureExtractor
from ticket_price_predictor.ml.features.timeseries import (
    MomentumFeatureExtractor,
    TimeSeriesFeatureExtractor,
)


class FeaturePipeline:
    """Orchestrates all feature extractors."""

    def __init__(
        self,
        include_momentum: bool = True,
    ) -> None:
        """Initialize feature pipeline.

        Args:
            include_momentum: Whether to include momentum features
        """
        self._extractors: list[FeatureExtractor] = [
            PerformerFeatureExtractor(),
            EventFeatureExtractor(),
            SeatingFeatureExtractor(),
            TimeSeriesFeatureExtractor(),
        ]

        if include_momentum:
            self._extractors.append(MomentumFeatureExtractor())

        self._fitted = False

    @property
    def feature_names(self) -> list[str]:
        """Return all feature names from all extractors."""
        names: list[str] = []
        for extractor in self._extractors:
            names.extend(extractor.feature_names)
        return names

    def fit(self, df: pd.DataFrame) -> "FeaturePipeline":
        """Fit all extractors on training data.

        Args:
            df: Training DataFrame

        Returns:
            self
        """
        for extractor in self._extractors:
            extractor.fit(df)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract all features from input DataFrame.

        Args:
            df: Input DataFrame with raw data

        Returns:
            DataFrame with all extracted features
        """
        features_list: list[pd.DataFrame] = []

        for extractor in self._extractors:
            features = extractor.extract(df)
            features_list.append(features)

        # Concatenate all feature DataFrames
        result = pd.concat(features_list, axis=1)

        return result

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit and transform in one step.

        Args:
            df: Training DataFrame

        Returns:
            DataFrame with all extracted features
        """
        self.fit(df)
        return self.transform(df)

    def get_params(self) -> dict[str, Any]:
        """Get parameters for all extractors."""
        return {
            f"extractor_{i}": extractor.get_params() for i, extractor in enumerate(self._extractors)
        }


def prepare_training_data(
    df: pd.DataFrame,
    target_col: str = "listing_price",
    include_momentum: bool = False,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare training data with features and target.

    Args:
        df: Raw listing DataFrame
        target_col: Column to use as target
        include_momentum: Whether to compute momentum features

    Returns:
        Tuple of (features DataFrame, target Series)
    """
    # Compute momentum if needed
    if include_momentum and "price_momentum_7d" not in df.columns:
        df = MomentumFeatureExtractor.compute_momentum_features(df)

    # Extract features
    pipeline = FeaturePipeline(include_momentum=include_momentum)
    X = pipeline.fit_transform(df)

    # Get target
    y = df[target_col].copy()

    return X, y
