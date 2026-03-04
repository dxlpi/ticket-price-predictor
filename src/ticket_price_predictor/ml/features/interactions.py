"""Interaction feature extraction (post-extraction stage)."""

from typing import Any

import numpy as np
import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor


class InteractionFeatureExtractor(FeatureExtractor):
    """Create cross-feature interactions from base features.

    This is a POST-extractor: it receives the concatenated base feature
    DataFrame (not the raw input data). It creates multiplicative and
    non-linear combinations that capture relationships between domains.
    """

    def __init__(self) -> None:
        """Initialize with a defensive global median default."""
        self._global_median: float = 150.0

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "artist_zone_price",
            "popularity_zone",
            "urgency_zone",
            "artist_city_tier",
            "days_to_event_log",
            "price_per_urgency",
            "artist_venue_price",
        ]

    def fit(self, df: pd.DataFrame) -> "InteractionFeatureExtractor":
        """Compute global venue median from training base features.

        Args:
            df: Concatenated base feature DataFrame (training split only)

        Returns:
            self
        """
        if "venue_median_price" in df.columns:
            valid = df["venue_median_price"].dropna()
            if len(valid) > 0:
                self._global_median = float(valid.median())
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract interaction features from base feature DataFrame.

        Safely retrieves base features with defaults when missing.

        Args:
            df: Concatenated base feature DataFrame

        Returns:
            DataFrame with interaction features
        """
        result = pd.DataFrame(index=df.index)

        # Safely get base features with defaults
        artist_avg = df.get("artist_avg_price", pd.Series(0.0, index=df.index))
        zone_ratio = df.get("zone_price_ratio", pd.Series(0.5, index=df.index))
        pop_score = df.get("popularity_score", pd.Series(0.0, index=df.index))
        urgency = df.get("urgency_bucket", pd.Series(2, index=df.index))
        city_tier = df.get("city_tier", pd.Series(3, index=df.index))
        days = df.get("days_to_event", pd.Series(30, index=df.index))
        venue_median = df.get("venue_median_price", pd.Series(0.0, index=df.index))

        result["artist_zone_price"] = artist_avg * zone_ratio
        result["popularity_zone"] = pop_score * zone_ratio
        result["urgency_zone"] = urgency * zone_ratio
        result["artist_city_tier"] = artist_avg * city_tier
        result["days_to_event_log"] = np.log1p(days.clip(lower=0))
        result["price_per_urgency"] = artist_avg / (days + 1)
        # Floor at 1.0 prevents divide-by-zero when global median is zero (unfitted or degenerate data)
        result["artist_venue_price"] = artist_avg * venue_median / max(self._global_median, 1.0)

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters."""
        return {}
