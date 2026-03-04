"""Venue-level feature extraction."""

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ticket_price_predictor.ml.features.base import FeatureExtractor


@dataclass
class VenueStats:
    """Statistics for a single venue."""

    venue_name: str
    avg_price: float
    median_price: float
    price_std: float
    listing_count: int


class VenueStatsCache:
    """Cache of Bayesian-smoothed venue statistics computed from listing data.

    Uses the same smoothing approach as RegionalStatsCache:
        smoothed = (n * group_stat + m * global_stat) / (n + m)
    where m = SMOOTHING_FACTOR. This pulls small-sample venues toward the
    global mean, preventing noisy estimates.
    """

    SMOOTHING_FACTOR = 75

    def __init__(self) -> None:
        """Initialize empty cache."""
        self._stats: dict[str, VenueStats] = {}
        self._global_avg: float = 0.0
        self._global_median: float = 0.0
        self._global_std: float = 0.0
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Return True if cache has been computed."""
        return self._fitted

    def fit(self, df: pd.DataFrame) -> "VenueStatsCache":
        """Compute venue statistics from listing data.

        Args:
            df: DataFrame with columns: venue_name, listing_price

        Returns:
            self
        """
        if "venue_name" not in df.columns or "listing_price" not in df.columns:
            self._fitted = True
            return self

        self._global_avg = df["listing_price"].mean()
        self._global_median = df["listing_price"].median()
        self._global_std = df["listing_price"].std()
        m = self.SMOOTHING_FACTOR

        for venue, group in df.groupby("venue_name"):
            venue_key = str(venue).lower().strip()
            n = len(group)
            group_mean = group["listing_price"].mean()
            group_median = group["listing_price"].median()
            group_std = group["listing_price"].std() if n > 1 else self._global_std

            smoothed_avg = (n * group_mean + m * self._global_avg) / (n + m)
            smoothed_median = (n * group_median + m * self._global_median) / (n + m)
            smoothed_std = (n * group_std + m * self._global_std) / (n + m)

            self._stats[venue_key] = VenueStats(
                venue_name=str(venue),
                avg_price=smoothed_avg,
                median_price=smoothed_median,
                price_std=smoothed_std,
                listing_count=n,
            )

        self._fitted = True
        return self

    def get_stats(self, venue: str) -> VenueStats:
        """Get statistics for a venue.

        Args:
            venue: Venue name

        Returns:
            VenueStats (or global defaults if venue not found)
        """
        if not self._fitted:
            return VenueStats(venue, 150.0, 150.0, 50.0, 0)

        venue_key = venue.lower().strip()
        if venue_key in self._stats:
            return self._stats[venue_key]

        return VenueStats(venue, self._global_avg, self._global_median, self._global_std, 0)

    def is_known_venue(self, venue: str) -> bool:
        """Check if venue is in the cache."""
        return venue.lower().strip() in self._stats


class VenueFeatureExtractor(FeatureExtractor):
    """Extract venue-level features using Bayesian-smoothed statistics."""

    def __init__(self) -> None:
        """Initialize extractor with empty venue stats cache."""
        self._cache = VenueStatsCache()

    @property
    def feature_names(self) -> list[str]:
        """Return list of feature names."""
        return [
            "venue_avg_price",
            "venue_median_price",
            "venue_price_std",
            "is_known_venue",
        ]

    def fit(self, df: pd.DataFrame) -> "VenueFeatureExtractor":
        """Fit extractor by computing venue statistics.

        Args:
            df: Training DataFrame with venue_name and listing_price columns

        Returns:
            self
        """
        self._cache.fit(df)
        return self

    def extract(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract venue features.

        Expects 'venue_name' column in input DataFrame.
        Falls back to zeros if column is missing.
        """
        result = pd.DataFrame(index=df.index)

        if "venue_name" not in df.columns:
            for name in self.feature_names:
                result[name] = 0.0
            return result

        stats_list = df["venue_name"].apply(self._cache.get_stats)
        result["venue_avg_price"] = stats_list.apply(lambda s: s.avg_price)
        result["venue_median_price"] = stats_list.apply(lambda s: s.median_price)
        result["venue_price_std"] = stats_list.apply(lambda s: s.price_std)
        result["is_known_venue"] = df["venue_name"].apply(
            lambda v: 1.0 if self._cache.is_known_venue(v) else 0.0
        )

        return result

    def get_params(self) -> dict[str, Any]:
        """Return extractor parameters."""
        return {"fitted": self._cache.is_fitted}
